using Connectomes
using ADNIDatasets
using CSV, DataFrames
using DrWatson: projectdir
using OrdinaryDiffEq
using Distributions
using Turing
using Serialization
using DelimitedFiles
using LinearAlgebra
using Random
using LinearAlgebra
using SparseArrays
include(projectdir("functions.jl"))
#-------------------------------------------------------------------------------
# Connectome and ROIs
#-------------------------------------------------------------------------------
connectome_path = Connectomes.connectome_path()
all_c = filter(Connectome(connectome_path; norm=true, weight_function = (n, l) -> n), 1e-2);

subcortex = filter(x -> get_lobe(x) == "subcortex", all_c.parc);
cortex = filter(x -> get_lobe(x) != "subcortex", all_c.parc);

c = slice(all_c, cortex; norm=true) |> filter

mtl_regions = ["entorhinal", "Left-Amygdala", "Right-Amygdala"]
mtl = findall(x -> x ∈ mtl_regions, get_label.(cortex))
neo_regions = ["inferiortemporal", "middletemporal"]
neo = findall(x -> x ∈ neo_regions, get_label.(cortex))
#-------------------------------------------------------------------------------
# Data 
#-------------------------------------------------------------------------------
sub_data_path = projectdir("adni/data/new_new_data/UCBERKELEY_TAU_6MM_18Dec2023_AB_STATUS.csv")
alldf = CSV.read(sub_data_path, DataFrame)

posdf = filter(x -> x.AB_Status == 1, alldf)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in get_node_id.(cortex)]

insample_data = ADNIDataset(posdf, dktnames; min_scans=3, qc=true)
insample_n_data = length(insample_data)
mtl_cutoff = 1.375
neo_cutoff = 1.395

insample_mtl_pos = filter(x -> regional_mean(insample_data, mtl, x) >= mtl_cutoff, 1:insample_n_data)
insample_neo_pos = filter(x -> regional_mean(insample_data, neo, x) >= neo_cutoff, 1:insample_n_data)

insample_tau_pos = findall(x -> x ∈ unique([insample_mtl_pos; insample_neo_pos]), 1:insample_n_data)
insample_tau_neg = findall(x -> x ∉ insample_tau_pos, 1:insample_n_data)

insample_n_pos = length(insample_tau_pos)
insample_n_neg = length(insample_tau_neg)

gmm_moments = CSV.read(projectdir("adni/data/component_moments.csv"), DataFrame)
ubase, upath = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(ubase)
cc = quantile.(upath, .99)
#-------------------------------------------------------------------------------
# Pos data 
#-------------------------------------------------------------------------------
_insample_subdata = [calc_suvr(insample_data, i) for i in insample_tau_pos]
insample_pos_data = [normalise(sd, u0, cc) for sd in _insample_subdata]

min_suvr = minimum(u0)
max_suvr = maximum(cc)

insample_pos_initial_conditions = [sd[:,1] for sd in insample_pos_data]
insample_pos_times =  [get_times(insample_data, i) for i in insample_tau_pos]
insample_max_t = maximum(reduce(vcat, insample_pos_times))

#-------------------------------------------------------------------------------
# Out of sample pos data 
#-------------------------------------------------------------------------------
outsample_data = ADNIDataset(posdf, dktnames; min_scans=2, max_scans=2, qc=true)
outsample_n_data = length(outsample_data)

outsample_mtl_pos = filter(x -> regional_mean(outsample_data, mtl, x) >= mtl_cutoff, 1:outsample_n_data)
outsample_neo_pos = filter(x -> regional_mean(outsample_data, neo, x) >= neo_cutoff, 1:outsample_n_data)

outsample_tau_pos = findall(x -> x ∈ unique([outsample_mtl_pos; outsample_neo_pos]), 1:outsample_n_data)
outsample_tau_neg = findall(x -> x ∉ outsample_tau_pos, 1:outsample_n_data)

outsample_n_pos = length(outsample_tau_pos)
outsample_n_neg = length(outsample_tau_neg)

_outsample_subdata = [calc_suvr(outsample_data, i) for i in outsample_tau_pos];
outsample_subdata = [normalise(sd, u0, cc) for sd in _outsample_subdata];
outsample_initial_conditions = [sd[:,1] for sd in outsample_subdata];
outsample_times =  [get_times(outsample_data, i) for i in outsample_tau_pos];
outsample_second_times = [t[2] for t in outsample_times]

#-------------------------------------------------------------------------------
# Models
#-------------------------------------------------------------------------------
L = laplacian_matrix(c)

vols = [get_vol(insample_data, i) for i in insample_tau_pos]
init_vols = [v[:,1] for v in vols]
max_norm_vols = reduce(hcat, [v ./ maximum(v) for v in init_vols])
mean_norm_vols = vec(mean(max_norm_vols, dims=2))
Lv = sparse(inv(diagm(mean_norm_vols)) * L)

function NetworkLocalFKPP(du, u, p, t; Lv = Lv, u0 = u0, cc = cc)
    du .= -p[1] * Lv * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

# function NetworkGlobalFKPP(du, u, p, t; Lv = Lv)
#     du .= -p[1] * Lv * u .+ p[2] .* u .* (1 .- ( u ./ p[3]))
# end
function NetworkGlobalFKPP(du, u, p, t; Lv = Lv)
    du .= -p[1] * Lv * (u .- p[3]) .+ p[2] .* (u .- p[3]) .* ((p[4] .- p[3]) .- (u .- p[3]))
end

function NetworkDiffusion(du, u, p, t; Lv = Lv, u0=u0)
    du .= -p[1] * L * (u .- u0)
end

function NetworkLogistic(du, u, p, t; Lv = Lv)
    du .= p[1] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

#-------------------------------------------------------------------------------
# Posteriors
#-------------------------------------------------------------------------------
local_pst = mean(deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-taupos-4x2000.jls")));
global_pst = mean(deserialize(projectdir("adni/new-chains/global-fkpp/length-free/pst-taupos-4x2000.jls")));
diffusion_pst = mean(deserialize(projectdir("adni/new-chains/diffusion/length-free/pst-taupos-4x2000.jls")));
logistic_pst = mean(deserialize(projectdir("adni/new-chains/logistic/pst-taupos-4x2000.jls")));
#-------------------------------------------------------------------------------
# Local model
#-------------------------------------------------------------------------------
ρs = [local_pst["ρ[$i]", :mean] for i in 1:insample_n_pos]
αs = [local_pst["α[$i]", :mean] for i in 1:insample_n_pos]

function simulate(f, initial_conditions, params, times)
    max_t = maximum(reduce(vcat, times))
    [solve(
        ODEProblem(
            f, inits, (0, max_t), p
        ), 
        Tsit5(), saveat=t
    )
    for (inits, p, t) in zip(initial_conditions, params, times)
    ]
end

local_sols = simulate(NetworkLocalFKPP, insample_pos_initial_conditions, collect(zip(ρs, αs)), insample_pos_times);
#-------------------------------------------------------------------------------
# Global model
#-------------------------------------------------------------------------------
ρs = [global_pst["ρ[$i]", :mean] for i in 1:insample_n_pos]
αs = [global_pst["α[$i]", :mean] for i in 1:insample_n_pos]

global_sols = simulate(NetworkGlobalFKPP, 
                       insample_pos_initial_conditions, 
                       collect(zip(ρs, αs, ones(insample_n_pos) * min_suvr, ones(insample_n_pos) * max_suvr)), 
                       insample_pos_times);
#-------------------------------------------------------------------------------
# Diffusion model
#-------------------------------------------------------------------------------
ρs = [diffusion_pst["ρ[$i]", :mean] for i in 1:insample_n_pos]

diffusion_sols = simulate(NetworkDiffusion, 
                       insample_pos_initial_conditions, 
                       ρs, 
                       insample_pos_times);

#-------------------------------------------------------------------------------
# Logistic model
#-------------------------------------------------------------------------------
αs = [logistic_pst["α[$i]", :mean] for i in 1:insample_n_pos]

logistic_sols = simulate(NetworkLogistic, insample_pos_initial_conditions, αs, insample_pos_times);
#-------------------------------------------------------------------------------
# Some convenience functions
#-------------------------------------------------------------------------------
using CairoMakie, ColorSchemes, Colors

function getdiff(d, n)
    d[:,n] .- d[:,1]
end

function getdiff(d, n, n2)
    d[:,n] .- d[:,n2]
end

function getdiff(d)
    d[:,end] .- d[:,1]
end

function get_sol_t(sols, n)
    asols = Array.(sols)
    n_sols = Vector{Vector{Float64}}()
    for sol in asols
        if size(sol, 2) >= n
            push!(n_sols, sol[:,n])
        end
    end
    reduce(hcat, n_sols)
end

function get_sol_mean_t(sols, n)
    mean(get_sol_t(sols, n), dims=2) |> vec
end

function get_sol_t_end(sols)
    asols = Array.(sols)
    n_sols = Vector{Vector{Float64}}()
    for sol in asols
        push!(n_sols, sol[:,end])
    end
    reduce(hcat, n_sols)
end

#-------------------------------------------------------------------------------
# Tau Positive Prediction Plot
#-------------------------------------------------------------------------------
titles = ["Local FKPP", "Global FKPP", "Diffusion", "Logistic"]
begin 
    cols = ColorSchemes.seaborn_colorblind[1:3]
    titlesize = 40
    xlabelsize = 25 
    ylabelsize = 25
    xticklabelsize = 20 
    yticklabelsize = 20
    f = Figure(resolution=(2000, 1100), fontsize=40);
    g = [f[i, j] = GridLayout() for i in 1:2, j in 1:4]
    gl = f[3,:] = GridLayout()
    for (i, sol) in enumerate([local_sols, global_sols, diffusion_sols, logistic_sols])
        start = 1.0
        stop = 4.0
        border = 0.25
        ax = Axis(g[1, i][1, 1],  
                xlabel="SUVR", 
                ylabel="Prediction", 
                title=titles[i],
                titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
                xminorgridvisible=true,yminorgridvisible=true,
                xminorticksvisible=true, yminorticksvisible=true,
                xminorticks=collect(start:0.5:stop),yminorticks=collect(start:0.5:stop),
                xticks=start:1.0:stop, yticks=start:1.0:stop, 
                xtickformat = "{:.1f}", ytickformat = "{:.1f}")
        if i > 1
            hideydecorations!(ax, minorgrid=false, minorticks=false, ticks=false, grid=false)
        end
        xlims!(ax, start - border, stop + border)
        ylims!(ax, start - border, stop + border)
        lines!(start:0.1:stop + 0.1, start:0.1:stop + 0.1, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
        # for j in 1:insample_n_pos
        for (scan, col) in zip(collect(2:4), alphacolor.(cols, [1.0,0.5,0.25]))
            for j in 1:insample_n_pos
                if size(insample_pos_data[j], 2) >= scan
                    scatter!(insample_pos_data[j][:,scan], sol[j][:,scan], color=col, markersize=15, marker='o', label="Scan $(scan)")
                end
            end
        end
        # end
        start = -1.0
        stop = 1.0
        ax = Axis(g[2, i][1,1], 
                xlabel="Δ SUVR",
                ylabel="Δ Prediction",
                titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize, 
                xminorgridvisible=true,yminorgridvisible=true,
                xminorticksvisible=true, yminorticksvisible=true,
                xminorticks=collect(start:0.25:stop),yminorticks=collect(start:0.25:stop),
                xticks=start:0.5:stop, yticks=start:0.5:stop, 
                xtickformat = "{:.1f}", ytickformat = "{:.1f}")
        if i > 1
            hideydecorations!(ax, minorgrid=false, minorticks=false, ticks=false, grid=false)
        end

        xlims!(ax, start, stop)
        ylims!(ax, start, stop)
        lines!(start:0.1:stop, start:0.1:stop, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
        for (scan, col) in zip(collect(2:4), alphacolor.(cols, [1.0,0.5,0.25]))
            if scan < 4
                diffs = getdiff.(insample_pos_data, scan)
                soldiff = getdiff.(sol, scan)
            else
                idx = findall(x -> size(x,2) == scan, insample_pos_data)
                diffs = getdiff.(insample_pos_data[idx], scan)
                soldiff = getdiff.(sol[idx], scan)
            end

            for j in eachindex(diffs)
                scatter!(diffs[j], soldiff[j], color=col, markersize=15, marker='o', label="Scan $(scan)")
            end
        end

        Legend(gl[1, 1],
                [MarkerElement(color = col, marker= '●', markersize=20) for col in cols],
                ["Scan 2", "Scan 3", "Scan 4"],
                patchsize = (35, 35), rowgap = 10, orientation = :horizontal, framevisible=false)


    end
    f
end
save(projectdir("visualisation/inference/model-selection/output/model-fits.pdf"), f)

#-------------------------------------------------------------------------------
# Regional average
#-------------------------------------------------------------------------------
begin
    cols = ColorSchemes.seaborn_colorblind[1:3]
    titlesize = 50
    xlabelsize = 40
    ylabelsize = 40
    xticklabelsize = 30 
    yticklabelsize = 30
    f = Figure(size=(2000, 1100), fontsize=40);
    g = [f[i, j] = GridLayout() for i in 1:2, j in 1:4]
    gl = f[3,:] = GridLayout()
    for (i, _sol) in enumerate([local_sols, global_sols, diffusion_sols, logistic_sols])
        start = 1.0
        stop = 1.915
        border = 0.025
        ax = Axis(g[1, i][1, 1],  
                xlabel="SUVR", 
                ylabel="Prediction", 
                title=titles[i],
                titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
                xticks=start:0.20:stop, yticks=start:0.20:stop, 
                xminorticks=start:0.10:stop+border, xminorticksvisible=true, xminorgridvisible=true,
                xminorgridcolor=RGBAf(0, 0, 0, 0.15), xgridcolor=RGBAf(0, 0, 0, 0.15),
                yminorticks=start:0.10:stop+border, yminorticksvisible=true, yminorgridvisible=true,
                yminorgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
                xtickformat = "{:.2f}", ytickformat = "{:.2f}")
        if i > 1
            hideydecorations!(ax, minorgrid=false, minorticks=false, ticks=false, grid=false)
        end
        xlims!(ax, start, stop + border)
        ylims!(ax, start, stop + border)
        lines!(start:0.05:stop+border, start:0.05:stop+border, color=(:grey, 0.75), linewidth=5, linestyle=:dash)

        preds = reduce(hcat, [get_sol_mean_t(_sol, j) for j in 1:4])
        obs = reduce(hcat, [get_sol_mean_t(insample_pos_data, j) for j in 1:4])
        fidx = findall(x -> length(x) > 3, insample_pos_times)
        fpreds = reduce(hcat, [get_sol_mean_t(_sol[fidx], j) for j in 1:4])
        fobs = reduce(hcat, [get_sol_mean_t(insample_pos_data[fidx], j) for j in 1:4])

        # for j in 1:insample_n_pos
        for (scan, col) in zip(collect(2:4), alphacolor.(cols, [1.0,0.5,0.3]))
            scatter!(obs[:,scan], preds[:,scan], color=col, markersize=15, marker='●', label="Scan $(scan)")
        end
        # end

        start = -0.025
        stop = 0.265
        border = 0.05
        ax = Axis(g[2, i][1,1], 
                xlabel="Δ SUVR",
                ylabel="Δ Prediction",
                titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize, 
                xticks=0:0.1:stop+border, yticks=0:0.1:stop+border, 
                xminorticks=0:0.05:stop+border, xminorticksvisible=true, xminorgridvisible=true,
                xminorgridcolor=RGBAf(0, 0, 0, 0.15), xgridcolor=RGBAf(0, 0, 0, 0.15),
                yminorticks=0:0.05:stop+border, yminorticksvisible=true, yminorgridvisible=true,
                yminorgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
                xtickformat = "{:.2f}", ytickformat = "{:.2f}")
        if i > 1
            hideydecorations!(ax, minorgrid=false, minorticks=false, ticks=false, grid=false)
        end
        xlims!(ax, start, stop + border)
        ylims!(ax, start, stop + border)
        lines!(start:0.01:stop + border, start:0.01:stop + border, color=(:grey, 0.75), linewidth=2, linestyle=:dash)

        for (scan, col) in zip(2:3, alphacolor.(cols, [1.0,0.5,0.5]))
            diffs = getdiff(obs, scan)
            soldiff = getdiff(preds, scan)
            scatter!(diffs, soldiff, color=col, markersize=20, marker='●', label="Scan $(scan)")
        end
        fdiff = getdiff(fobs, 4)
        fsoldiff = getdiff(fpreds, 4)
        scatter!(fdiff, fsoldiff, color=(cols[3], 0.5), markersize=20, marker='●', label="Scan 4")

        Legend(gl[1, 1],
                [MarkerElement(color = col, marker= '●', markersize=30) for col in cols],
                ["Scan 2", "Scan 3", "Scan 4"],
                patchsize = (35, 35), rowgap = 10, orientation = :horizontal, framevisible=false)
    end
    f
end
save(projectdir("visualisation/inference/model-selection/output/model-fits-roi-average.pdf"), f)

begin
    cols = Makie.wong_colors()[1:3]
    titlesize = 30
    xlabelsize = 30
    ylabelsize = 30
    xticklabelsize = 25
    yticklabelsize = 25
    f = Figure(size=(600, 1000), fontsize=40);
    g = f[1, 1] = GridLayout()
    # gl = f[2,1] = GridLayout()
    g2 = f[2,1] = GridLayout()
    for (i, _sol) in enumerate([local_sols])
       
        preds = reduce(hcat, [get_sol_mean_t(_sol, j) for j in 1:4])
        obs = reduce(hcat, [get_sol_mean_t(insample_pos_data, j) for j in 1:4])
        fidx = findall(x -> length(x) > 3, insample_pos_times)
        fpreds = reduce(hcat, [get_sol_mean_t(_sol[fidx], j) for j in 1:4])
        fobs = reduce(hcat, [get_sol_mean_t(insample_pos_data[fidx], j) for j in 1:4])

        start = -0.025
        stop = 0.265
        border = 0.05
        ax = Axis(g[1, i][1,1], 
                xlabel="Δ SUVR",
                ylabel="Δ Prediction",
                titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize, 
                xticks=0:0.1:stop+border, yticks=0:0.1:stop+border, 
                xminorticks=0:0.05:stop+border, xminorticksvisible=true, xminorgridvisible=true,
                xminorgridcolor=RGBAf(0, 0, 0, 0.15), xgridcolor=RGBAf(0, 0, 0, 0.15),
                yminorticks=0:0.05:stop+border, yminorticksvisible=true, yminorgridvisible=true,
                yminorgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
                xtickformat = "{:.2f}", ytickformat = "{:.2f}", xticksize=10,  yticksize=10)
        if i > 1
            hideydecorations!(ax, minorgrid=false, minorticks=false, ticks=false, grid=false)
        end
        xlims!(ax, start, stop + border)
        ylims!(ax, start, stop + border)
        lines!(start:0.01:stop + border, start:0.01:stop + border, color=(:grey, 0.75), linewidth=5, linestyle=:dash)

        for (scan, col) in zip(2:3, alphacolor.(cols, [1.0,0.5,0.5]))
            diffs = getdiff(obs, scan)
            soldiff = getdiff(preds, scan)
            scatter!(diffs, soldiff, color=col, markersize=20, marker='●', label="Scan $(scan)")
        end
        fdiff = getdiff(fobs, 4)
        fsoldiff = getdiff(fpreds, 4)
        scatter!(fdiff, fsoldiff, color=(cols[3], 0.5), markersize=20, marker='●', label="Scan 4")
    end
    Legend(g[2, 1],
    [MarkerElement(color = col, marker= '●', markersize=20) for col in cols],
    ["Scan 2", "Scan 3", "Scan 4"], labelsize=30, tellheight=true, tellwidth=false,
    patchsize = (35, 35), rowgap = 10, orientation = :horizontal, framevisible=false)

    start = -0.025
    stop = 0.265
    border = 0.05
    ax = Axis(g2[1, 1], 
            xlabel="Δ SUVR",
            ylabel="Δ Prediction",
            titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
            xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize, 
            xticks=0:0.1:stop+border, yticks=0:0.1:stop+border, 
            xminorticks=0:0.05:stop+border, xminorticksvisible=true, xminorgridvisible=true,
            xminorgridcolor=RGBAf(0, 0, 0, 0.15), xgridcolor=RGBAf(0, 0, 0, 0.15),
            yminorticks=0:0.05:stop+border, yminorticksvisible=true, yminorgridvisible=true,
            yminorgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
            xtickformat = "{:.2f}", ytickformat = "{:.2f}", xticksize=10,  yticksize=10)

    scatter!(mean_final_scan .- mean_first_scan, mean_final_pred .- mean_first_scan, 
    markersize=20, color=(Makie.wong_colors()[4], 0.75));
    xlims!(ax, start, stop + border)
    ylims!(ax, start, stop + border)
    lines!(start:0.01:stop + border, start:0.01:stop + border, color=(:grey, 0.75), linewidth=5, linestyle=:dash)

    Legend(g2[2, 1],
    [MarkerElement(color = Makie.wong_colors()[4], marker= '●', markersize=20)],
    ["Out of Sample"], labelsize=30, tellheight=true, tellwidth=false,
    patchsize = (35, 35), rowgap = 10, orientation = :horizontal, framevisible=false)

    f
end
save(projectdir("visualisation/inference/model-selection/output/model-fits-roi-average-local-fkpp.pdf"), f)

begin 
    CairoMakie.activate!()
    col = ColorSchemes.seaborn_colorblind[1]
    titlesize = 50
    xlabelsize = 40
    ylabelsize = 40
    xticklabelsize = 30 
    yticklabelsize = 30
    f = Figure(resolution=(2000, 1000), fontsize=40);
    g = [f[i, j] = GridLayout() for i in 1:2, j in 1:4]
    for (i, _sol) in enumerate([local_sols, global_sols, diffusion_sols, logistic_sols])
        start = 1.0
        stop = 1.70
        border = 0.025
        ax = Axis(g[1, i][1, 1],  
                xlabel="SUVR", 
                ylabel="Prediction", 
                title=titles[i],
                titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
                xticks=start:0.20:stop, yticks=start:0.20:stop, 
                xminorticks=start:0.10:stop, xminorticksvisible=true, xminorgridvisible=true,
                xminorgridcolor=RGBAf(0, 0, 0, 0.15), xgridcolor=RGBAf(0, 0, 0, 0.15),
                yminorticks=start:0.10:stop, yminorticksvisible=true, yminorgridvisible=true,
                yminorgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
                xtickformat = "{:.2f}", ytickformat = "{:.2f}")
        if i > 1
            hideydecorations!(ax, minorgrid=false, minorticks=false, ticks=false, grid=false)
        end
        xlims!(ax, start, stop + border)
        ylims!(ax, start, stop + border)
        lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 0.75), linewidth=5, linestyle=:dash)


        final_obs = mean(get_sol_t_end(insample_pos_data), dims=2) |> vec
        final_preds = mean(get_sol_t_end(_sol), dims=2) |> vec
        scatter!(final_obs, final_preds, color=(col, 0.5), markersize=20, marker='●')

        # end
        start = -0.03
        stop = 0.2
        border = 0.05
        ax = Axis(g[2, i][1,1], 
                xlabel="Δ SUVR",
                ylabel="Δ Prediction",
                titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize, 
                xticks=start:0.05:stop, yticks=start:0.05:stop, 
                xminorticks=start:0.025:stop, xminorticksvisible=true, xminorgridvisible=true,
                xminorgridcolor=RGBAf(0, 0, 0, 0.15), xgridcolor=RGBAf(0, 0, 0, 0.15),
                yminorticks=start:0.025:stop, yminorticksvisible=true, yminorgridvisible=true,
                yminorgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
                xtickformat = "{:.2f}", ytickformat = "{:.2f}")
        if i > 1
            hideydecorations!(ax, minorgrid=false, minorticks=false, ticks=false, grid=false)
        end
        xlims!(ax, start, stop + border)
        ylims!(ax, start, stop + border)
        lines!(start:0.1:stop, start:0.1:stop, color=(:grey, 0.75), linewidth=2, linestyle=:dash)

        final_diffs = mean(getdiff.(insample_pos_data))
        final_soldiffs =  mean(getdiff.(_sol))
        scatter!(final_diffs, final_soldiffs, color=(col, 0.5), markersize=20, marker='●')
    end
    f
end
save(projectdir("visualisation/inference/model-selection/output/model-fits-roi-average-final-scan.pdf"), f)

begin 
    CairoMakie.activate!()
    _titles = ["Local FKPP", "Logistic"]

    col = ColorSchemes.seaborn_colorblind[1]
    titlesize = 25
    xlabelsize = 25 
    ylabelsize = 25
    xticklabelsize = 20 
    yticklabelsize = 20
    f = Figure(resolution=(500, 600), fontsize=40);
    g = [f[i, 1] = GridLayout() for i in 1:2]
    for (i, _sol) in enumerate([local_sols, logistic_sols])
        start = 0.0
        stop = 0.2
        border = 0.03
        ax = Axis(g[i][1,1], 
                xlabel="Δ SUVR",
                ylabel="Δ Prediction",
                titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize, xticksize=15, yticksize=15,
                xticks=start:0.05:stop, yticks=start:0.05:stop, 
                xminorticks=start:0.025:stop, xminorticksvisible=true, xminorgridvisible=true,
                xminorgridcolor=RGBAf(0, 0, 0, 0.15), xgridcolor=RGBAf(0, 0, 0, 0.15),
                yminorticks=start:0.025:stop, yminorticksvisible=true, yminorgridvisible=true,
                yminorgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
                xtickformat = "{:.2f}", ytickformat = "{:.2f}")
        if i == 1
            hidexdecorations!(ax, minorgrid=false, minorticks=false, ticks=false, grid=false)
        end
        xlims!(ax, start, stop + border)
        ylims!(ax, start, stop + border)
        lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 0.75), linewidth=5, linestyle=:dash)

        final_diffs = mean(getdiff.(insample_pos_data))
        final_soldiffs =  mean(getdiff.(_sol))
        scatter!(final_diffs, final_soldiffs, color=(col, 0.5), markersize=15, marker='●')
    end
    Label(f[1, 0], "Local FKPP", rotation=pi/2, tellheight=false, fontsize=30, font=:bold)
    Label(f[2, 0], "Logistic", rotation=pi/2, tellheight=false, fontsize=30, font=:bold)
    f
end
save(projectdir("visualisation/inference/model-selection/output/regional_mean_error_final.pdf"), f)
#-------------------------------------------------------------------------------
# Error visualisation
#-------------------------------------------------------------------------------
using ColorSchemes, GLMakie; GLMakie.activate!()
cmap = reverse(ColorSchemes.RdBu);
local_meanerror = vec(mean(get_sol_t_end(local_sols) .- get_sol_t_end(insample_pos_data), dims=2))
logistic_meanerror = vec(mean(get_sol_t_end(logistic_sols) .- get_sol_t_end(insample_pos_data), dims=2))

right_cortex = filter(x -> get_hemisphere(x) == "right", cortex)
left_cortex = filter(x -> get_hemisphere(x) == "left", cortex)
lims = maximum(abs.([local_meanerror; logistic_meanerror]))
begin
    GLMakie.activate!()
    f = Figure(resolution=(1000, 300))
    titlesize = 25
    xlabelsize = 25 
    ylabelsize = 25
    xticklabelsize = 20 
    yticklabelsize = 20
    # for (i, _sol) in enumerate([local_sols, logistic_sols])
    #     start = 0.0
    #     stop = 0.2
    #     border = 0.03
    #     ax = Axis(f[i,1], 
    #             xlabel="Δ SUVR",
    #             ylabel="Δ Prediction",
    #             titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
    #             xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize, xticksize=15, yticksize=15,
    #             xticks=start:0.05:stop, yticks=start:0.05:stop, 
    #             xminorticks=start:0.025:stop, xminorticksvisible=true, xminorgridvisible=true,
    #             xminorgridcolor=RGBAf(0, 0, 0, 0.15), xgridcolor=RGBAf(0, 0, 0, 0.15),
    #             yminorticks=start:0.025:stop, yminorticksvisible=true, yminorgridvisible=true,
    #             yminorgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
    #             xtickformat = "{:.2f}", ytickformat = "{:.2f}")
    #     if i == 1
    #         hidexdecorations!(ax, minorgrid=false, minorticks=false, ticks=false, grid=false)
    #     end
    #     xlims!(ax, start, stop + border)
    #     ylims!(ax, start, stop + border)
    #     lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 0.75), linewidth=5, linestyle=:dash)

    #     final_diffs = mean(getdiff.(insample_pos_data))
    #     final_soldiffs =  mean(getdiff.(_sol))
    #     scatter!(final_diffs, final_soldiffs, color=(col, 0.5), markersize=20, marker='●')
    # end
    Label(f[1, 0], "Local FKPP", rotation=pi/2, tellheight=false, fontsize=20, font=:bold)
    Label(f[2, 0], "Logistic", rotation=pi/2, tellheight=false, fontsize=20, font=:bold)

    ax = Axis3(f[1,1 ], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax)
    hidespines!(ax)
    for (i, j) in enumerate(get_node_id.(right_cortex))
        plot_roi!(j, get(cmap, local_meanerror[i], (-lims, lims)))
    end

    ax = Axis3(f[1,2 ], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax)
    hidespines!(ax)
    for (i, j) in enumerate(get_node_id.(right_cortex))
        plot_roi!(j, get(cmap, local_meanerror[i], (-lims, lims)))
    end

    ax = Axis3(f[1,3 ], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax)
    hidespines!(ax)
    for (i, j) in enumerate(get_node_id.(left_cortex))
        plot_roi!(j, get(cmap, local_meanerror[i], (-lims, lims)))
    end

    ax = Axis3(f[1,4 ], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax)
    hidespines!(ax)
    for (i, j) in enumerate(get_node_id.(left_cortex))
        plot_roi!(j, get(cmap, local_meanerror[i], (-lims, lims)))
    end

    ax = Axis3(f[2,1 ], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax)
    hidespines!(ax)
    for (i, j) in enumerate(get_node_id.(right_cortex))
        plot_roi!(j, get(cmap, logistic_meanerror[i], (-lims, lims)))
    end

    ax = Axis3(f[2,2 ], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax)
    hidespines!(ax)
    for (i, j) in enumerate(get_node_id.(right_cortex))
        plot_roi!(j, get(cmap, logistic_meanerror[i], (-lims, lims)))
    end

    ax = Axis3(f[2,3 ], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax)
    hidespines!(ax)
    for (i, j) in enumerate(get_node_id.(left_cortex))
        plot_roi!(j, get(cmap, logistic_meanerror[i], (-lims, lims)))
    end

    ax = Axis3(f[2,4 ], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax)
    hidespines!(ax)
    for (i, j) in enumerate(get_node_id.(left_cortex))
        plot_roi!(j, get(cmap, logistic_meanerror[i], (-lims, lims)))
    end

    # Label(f[1, 0], "Local FKPP", rotation=pi/2, tellheight=false, fontsize=30, font=:bold)
    # Label(f[2, 0], "Logistic", rotation=pi/2, tellheight=false, fontsize=30, font=:bold)
    _tickformat(x) = x < 0 ? "$x" : " $x"
    Colorbar(f[1:2,5], 
    limits = (-lims, lims),  ticks=-0.08:0.02:0.08, colormap = cmap, vertical=true, flipaxis=true, label="Mean Error", 
    labelrotation=-pi/2, labelsize=25, ticklabelsize=20, tickformat = xs -> [_tickformat(x) for x in xs])
    f
end
save(projectdir("visualisation/inference/model-selection/output/regional_mean_error_tau_pos.jpeg"), f)

begin
    CairoMakie.activate!()
    cols = Makie.wong_colors();
    mean_data = vec(mean(get_sol_t_end(insample_pos_data), dims=2))
    mean_local_error = vec(mean(get_sol_t_end(local_sols) .- get_sol_t_end(insample_pos_data), dims=2))
    mean_global_error = vec(mean(get_sol_t_end(global_sols) .- get_sol_t_end(insample_pos_data), dims=2))
    mean_logistic_error = vec(mean(get_sol_t_end(logistic_sols) .- get_sol_t_end(insample_pos_data), dims=2))
    f = Figure(size=(1000, 600), fontsize=20)
    ax = Axis(f[1, 1:3], yticksize=15, ylabel="Mean Residual", ylabelsize=20)
    hidexdecorations!(ax, grid=false)
    ylims!(-0.15, 0.15)
    for (i, j) in zip(mean_data, mean_local_error)
        linesegments!([i, i], [0, j], color=(cols[1], 0.5))
    end
    scatter!(mean_data, 
            mean_local_error, 
            markersize=15, color=(cols[1], 0.8))
    hlines!(ax, 0, color=cols[1])
    hlines!(ax, mean(mean_local_error), color=:black, linestyle=:dash)
    
    ax = Axis(f[1, 4])
    hideydecorations!(ax, ticks=false)
    hidexdecorations!(ax)
    hidespines!(ax, :b, :t, :r)
    ylims!(-0.15, 0.15)
    density!(mean_local_error, direction=:y)
    hlines!(ax, 0, color=cols[1])
    hlines!(ax, mean(mean_local_error),  color=:black, linestyle=:dash)

    ax = Axis(f[2, 1:3], yticksize=15, ylabel="Mean Residual", xlabel="SUVR", ylabelsize=20, xlabelsize=20)
    hidexdecorations!(ax)
    ylims!(-0.15, 0.15)
    for (i, j) in zip(mean_data, mean_global_error)
        linesegments!([i, i], [0, j], color=(cols[1], 0.5))
    end
    scatter!(mean_data, 
            mean_global_error, 
            markersize=15, color=(cols[1], 0.8))
    hlines!(ax, 0, color=cols[1])
    hlines!(ax, mean(mean_global_error),  color=:black, linestyle=:dash)

    ax = Axis(f[2, 4])
    hideydecorations!(ax, ticks=false)
    hidexdecorations!(ax)
    hidespines!(ax, :b, :t, :r)
    ylims!(-0.15, 0.15)
    density!(mean_global_error, direction=:y)
    hlines!(ax, 0, color=cols[1])
    hlines!(ax, mean(mean_global_error),  color=:black, linestyle=:dash)


    ax = Axis(f[3, 1:3], yticksize=15, ylabel="Mean Residual", xlabel="SUVR", ylabelsize=20, xlabelsize=20)
    ylims!(-0.15, 0.15)
    for (i, j) in zip(mean_data, mean_logistic_error)
        linesegments!([i, i], [0, j], color=(cols[1], 0.5))
    end
    scatter!(mean_data, 
            mean_logistic_error, 
            markersize=15, color=(cols[1], 0.8))
    hlines!(ax, 0, color=cols[1])
    hlines!(ax, mean(mean_logistic_error),  color=:black, linestyle=:dash)

    ax = Axis(f[3, 4])
    hideydecorations!(ax, ticks=false)
    hidexdecorations!(ax)
    hidespines!(ax, :b, :t, :r)
    ylims!(-0.15, 0.15)
    density!(mean_logistic_error, direction=:y)
    hlines!(ax, 0, color=cols[1])
    hlines!(ax, mean(mean_logistic_error),  color=:black, linestyle=:dash)

    colgap!(f.layout, 10)
    Label(f[1, 0], "Local FKPP", rotation=pi/2, tellheight=false, fontsize=20, font=:bold)
    Label(f[2, 0], "Global FKPP", rotation=pi/2, tellheight=false, fontsize=20, font=:bold)
    Label(f[3, 0], "Logistic", rotation=pi/2, tellheight=false, fontsize=20, font=:bold)
    f
end
save(projectdir("visualisation/inference/model-selection/output/regional_residuals_tau_pos.pdf"), f)

begin
    CairoMakie.activate!()
    cols = Makie.wong_colors();
    mean_data = vec(get_sol_t_end(insample_pos_data))
    mean_local_error = vec(get_sol_t_end(local_sols) .- get_sol_t_end(insample_pos_data))
    mean_global_error = vec(get_sol_t_end(global_sols) .- get_sol_t_end(insample_pos_data))
    mean_logistic_error = vec(get_sol_t_end(logistic_sols) .- get_sol_t_end(insample_pos_data))
    f = Figure(size=(1000, 600), fontsize=20)
    ax = Axis(f[1, 1:3], yticksize=15, ylabel="Residual", ylabelsize=20)
    hidexdecorations!(ax, grid=false)
    ylims!(-1., 1.)
    scatter!(mean_local_error, 
            markersize=15, color=(cols[1], 0.8))
    hlines!(ax, 0, color=cols[1])
    hlines!(ax, mean(mean_local_error), color=:black, linestyle=:dash)
    
    ax = Axis(f[1, 4])
    hideydecorations!(ax, ticks=false)
    hidexdecorations!(ax)
    hidespines!(ax, :b, :t, :r)
    ylims!(-1., 1.)
    xlims!(0, 10)
    hist!(mean_local_error,  direction=:x, bins=100, normalization=:pdf)
    lines!([pdf(Normal(0, local_pst[:σ, :mean]), x) for x in -0.5:0.001:0.5], -0.5:0.001:0.5, linewidth=5, color=:orange)
    hlines!(ax, 0, color=cols[1])
    hlines!(ax, mean(mean_local_error),  color=:black, linestyle=:dash)

    ax = Axis(f[2, 1:3], yticksize=15, ylabel="Residual", ylabelsize=20, xlabelsize=20)
    hidexdecorations!(ax, grid=false)
    ylims!(-1., 1.)
    scatter!(mean_global_error, 
            markersize=15, color=(cols[1], 0.8))
    hlines!(ax, 0, color=cols[1])
    hlines!(ax, mean(mean_global_error),  color=:black, linestyle=:dash)

    ax = Axis(f[2, 4])
    hideydecorations!(ax, ticks=false)
    hidexdecorations!(ax)
    hidespines!(ax, :b, :t, :r)
    ylims!(-1., 1.)
    xlims!(0, 10)
    hist!(mean_global_error,  direction=:x, bins=100, normalization=:pdf)
    lines!([pdf(Normal(0, global_pst[:σ, :mean]), x) for x in -0.5:0.001:0.5], -0.5:0.001:0.5, linewidth=5, color=:orange)
    hlines!(ax, 0, color=cols[1])
    hlines!(ax, mean(mean_global_error),  color=:black, linestyle=:dash)

    ax = Axis(f[3, 1:3], yticksize=15, ylabel="Residual", ylabelsize=20, xlabelsize=20)
    hidexdecorations!(ax, grid=false)
    ylims!(-1., 1.)
    scatter!(mean_logistic_error, 
            markersize=15, color=(cols[1], 0.8))
    hlines!(ax, 0, color=cols[1])
    hlines!(ax, mean(mean_logistic_error),  color=:black, linestyle=:dash)

    ax = Axis(f[3, 4])
    hideydecorations!(ax, ticks=false)
    hidexdecorations!(ax)
    hidespines!(ax, :b, :t, :r)
    ylims!(-1., 1.)
    xlims!(0, 10)
    hist!(mean_logistic_error,  direction=:x, bins=100, normalization=:pdf)
    lines!([pdf(Normal(0, logistic_pst[:σ, :mean]), x) for x in -0.5:0.001:0.5], -0.5:0.001:0.5, linewidth=5, color=:orange)
    hlines!(ax, 0, color=cols[1])
    hlines!(ax, mean(mean_logistic_error),  color=:black, linestyle=:dash)

    colgap!(f.layout, 10)
    Label(f[1, 0], "Local FKPP", rotation=pi/2, tellheight=false, fontsize=20, font=:bold)
    Label(f[2, 0], "Global FKPP", rotation=pi/2, tellheight=false, fontsize=20, font=:bold)
    Label(f[3, 0], "Logistic", rotation=pi/2, tellheight=false, fontsize=20, font=:bold)
    f
end
save(projectdir("visualisation/inference/model-selection/output/all_residuals_tau_pos.pdf"), f)

#-------------------------------------------------------------------------------
# Out of Sample Models
#-------------------------------------------------------------------------------
L = laplacian_matrix(c)

vols = [get_vol(outsample_data, i) for i in outsample_tau_pos]
init_vols = [v[:,1] for v in vols]
max_norm_vols = reduce(hcat, [v ./ maximum(v) for v in init_vols])
mean_norm_vols = vec(mean(max_norm_vols, dims=2))
Lv = sparse(inv(diagm(mean_norm_vols)) * L)

function NetworkLocalFKPP(du, u, p, t; Lv = Lv, u0 = u0, cc = cc)
    du .= -p[1] * Lv * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

# function NetworkGlobalFKPP(du, u, p, t; Lv = Lv)
#     du .= -p[1] * Lv * u .+ p[2] .* u .* (1 .- ( u ./ p[3]))
# end
function NetworkGlobalFKPP(du, u, p, t; Lv = Lv)
    du .= -p[1] * Lv * (u .- p[3]) .+ p[2] .* (u .- p[3]) .* ((p[4] .- p[3]) .- (u .- p[3]))
end

function NetworkDiffusion(du, u, p, t; Lv = Lv)
    du .= -p[1] * Lv * u
end

function NetworkLogistic(du, u, p, t; Lv = Lv)
    du .= p[1] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end


#-------------------------------------------------------------------------------
# Out of Sample Predictions
#-------------------------------------------------------------------------------
p, a = local_pst[:Pm, :mean], local_pst[:Am, :mean]
outsample_local_sols = simulate(NetworkLocalFKPP, 
                                outsample_initial_conditions, 
                                collect(zip(ones(outsample_n_pos) .* p, ones(outsample_n_pos) .* a)), 
                                outsample_times);

p, a = global_pst[:Pm, :mean], global_pst[:Am, :mean]
outsample_global_sols = simulate(NetworkGlobalFKPP, 
                                outsample_initial_conditions, 
                                collect(zip(ones(outsample_n_pos) .* p, 
                                            ones(outsample_n_pos) .* a, 
                                            ones(outsample_n_pos) .* min_suvr,
                                            ones(outsample_n_pos) .* max_suvr)), 
                                outsample_times);
                                
p  = diffusion_pst[:Pm, :mean]
outsample_diffusion_sols = simulate(NetworkDiffusion, 
                                outsample_initial_conditions, 
                                ones(outsample_n_pos) .* p, 
                                outsample_times);

a = logistic_pst[:Am, :mean]
outsample_logistic_sols = simulate(NetworkLogistic, 
                                outsample_initial_conditions, 
                                ones(outsample_n_pos) .* a, 
                                outsample_times);

#-------------------------------------------------------------------------------
# Out of Sample figure
#-------------------------------------------------------------------------------
begin
    cols = ColorSchemes.seaborn_colorblind[1:3]
    titlesize = 30
    xlabelsize = 25 
    ylabelsize = 25
    xticklabelsize = 20 
    yticklabelsize = 20
    f = Figure(resolution=(2000, 1000), fontsize=40);
    g = [f[i, j] = GridLayout() for i in 1:2, j in 1:4]
    for (i, sol) in enumerate([outsample_local_sols, outsample_global_sols, outsample_diffusion_sols, outsample_logistic_sols])
        ax = Axis(g[1, i][1, 1],  
                xlabel="SUVR", 
                ylabel="Prediction", 
                title=titles[i],
                titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,  
                xticks=0.0:0.5:4.0, yticks=0.0:0.5:4.0, 
                xtickformat = "{:.1f}", ytickformat = "{:.1f}")
        if i > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        xlims!(ax, 0.8, 4.0)
        ylims!(ax, 0.8, 4.0)
        lines!(0.8:0.1:4.0, 0.8:0.1:4.0, color=(:grey, 0.75), linewidth=2, linestyle=:dash)

        for j in 1:outsample_n_pos
            scatter!(outsample_subdata[j][:,end], sol[j][:,end], color=(:grey ,0.25), markersize=20, marker='o')
        end

        start = -0.5
        stop = 0.5
        ax = Axis(g[2, i][1,1], 
                xlabel="Δ SUVR",
                ylabel="Δ Prediction",
                titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize, 
                xticks=collect(start:0.1:stop), yticks=collect(start:0.1:stop),
                xtickformat = "{:.1f}", ytickformat = "{:.1f}")
        if i > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        xlims!(ax, start + (start * 0.05), stop + (stop * 0.05))
        ylims!(ax, start + (start * 0.05), stop + (stop * 0.05))
        lines!(start:0.1:stop, start:0.1:stop, color=(:grey, 0.75), linewidth=2, linestyle=:dash)

        for j in 1:outsample_n_pos
            diffs = getdiff(outsample_subdata[j], 2)
            soldiff = getdiff(sol[j], 2)
            scatter!(diffs, soldiff, color=(:grey,0.25), markersize=20, marker='o')
        end
    end
    f
end

begin 
    col = ColorSchemes.seaborn_colorblind[1]
    titlesize = 40
    xlabelsize = 25 
    ylabelsize = 25
    xticklabelsize = 20 
    yticklabelsize = 20
    f = Figure(resolution=(2000, 1000), fontsize=40);
    g = [f[i, j] = GridLayout() for i in 1:2, j in 1:4]
    for (i, sol) in enumerate([outsample_local_sols, outsample_global_sols, outsample_diffusion_sols, outsample_logistic_sols])
        start = 1.0
        stop = 1.80
        ax = Axis(g[1, i][1, 1],  
                xlabel="SUVR", 
                ylabel="Prediction", 
                title=titles[i],
                titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
                xminorgridvisible=true, yminorgridvisible=true,
                xminorticksvisible=true, yminorticksvisible=true,
                xminorticks=collect(start:0.1:stop),yminorticks=collect(start:0.1:stop),
                xticks=start:0.2:stop, yticks=start:0.2:stop, 
                xtickformat = "{:.1f}", ytickformat = "{:.1f}")
        if i > 1
            hideydecorations!(ax, minorgrid=false, minorticks=false, ticks=false, grid=false)
        end
        xlims!(ax, start, stop + 0.05)
        ylims!(ax, start, stop + 0.05)
        lines!(start:0.1:1.9, start:0.1:1.9, color=(:grey, 0.75), linewidth=5, linestyle=:dash)

        preds = reduce(hcat, [get_sol_mean_t(sol, i) for i in 1:2])
        obs = reduce(hcat, [get_sol_mean_t(outsample_subdata, i) for i in 1:2])
        
        scatter!(obs[:,end], preds[:,end], color=Colors.alphacolor(col, 0.5), markersize=20)

        start = -0.025
        stop = 0.155
        ax = Axis(g[2, i][1,1], 
                xlabel="Δ SUVR",
                ylabel="Δ Prediction",
                titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize, 
                xticks=collect(0:0.05:0.15), yticks=collect(0:0.05:0.1),
                xminorgridvisible=true,yminorgridvisible=true,
                xminorticksvisible=true, yminorticksvisible=true,
                xminorticks=collect(start:0.025:stop),yminorticks=collect(start:0.025:stop),
                xtickformat = "{:.2f}", ytickformat = "{:.2f}")
        if i > 1
            hideydecorations!(ax, minorgrid=false, minorticks=false, ticks=false, grid=false)
        end
        xlims!(ax, start, stop + (stop * 0.1))
        ylims!(ax, start, stop + (stop * 0.1))
        lines!(start:0.01:stop + (stop * 0.1), start:0.01:stop + (stop * 0.1), color=(:grey, 0.75), linewidth=2, linestyle=:dash)

        diffs = getdiff(obs, 2)
        soldiff = getdiff(preds, 2)
        scatter!(diffs, soldiff, color=Colors.alphacolor(col, 0.5), markersize=20)
    end
    f
end
save(projectdir("visualisation/inference/model-selection/output/outsample-model-fits-roi-average.pdf"), f)

