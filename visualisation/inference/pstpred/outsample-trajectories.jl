using Connectomes
using ADNIDatasets
using CSV, DataFrames
using DrWatson: projectdir
using DifferentialEquations
using SciMLSensitivity
using Zygote
using Turing
using AdvancedHMC
using Distributions
using Serialization
using DelimitedFiles, LinearAlgebra
using Random
using LinearAlgebra
using SparseArrays
include(projectdir("functions.jl"))
#-------------------------------------------------------------------------------
# Connectome and ROIs
#-------------------------------------------------------------------------------
connectome_path = Connectomes.connectome_path()
all_c = filter(Connectome(connectome_path; norm=true), 1e-2);

subcortex = filter(x -> x.Lobe == "subcortex", all_c.parc);
cortex = filter(x -> x.Lobe != "subcortex", all_c.parc);

c = slice(all_c, cortex) |> filter

mtl_regions = ["entorhinal", "Left-Amygdala", "Right-Amygdala"]
mtl = findall(x -> x ∈ mtl_regions, cortex.Label)
neo_regions = ["inferiortemporal", "middletemporal"]
neo = findall(x -> x ∈ neo_regions, cortex.Label)
#-------------------------------------------------------------------------------
# Data 
#-----------------------------------------------------------------------------
sub_data_path = projectdir("adni/data/AV1451_Diagnosis-STATUS-STIME-braak-regions.csv")
alldf = CSV.read(sub_data_path, DataFrame)

posdf = filter(x -> x.STATUS == "POS", alldf)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in cortex.ID]

data = ADNIDataset(posdf, dktnames; min_scans=3)

# Ask Jake where we got these cutoffs from? 
mtl_cutoff = 1.375
neo_cutoff = 1.395

mtl_pos = filter(x -> regional_mean(data, mtl, x) >= mtl_cutoff, 1:50)
neo_pos = filter(x -> regional_mean(data, neo, x) >= neo_cutoff, 1:50)

tau_pos = findall(x -> x ∈ unique([mtl_pos; neo_pos]), 1:50)
tau_neg = findall(x -> x ∉ tau_pos, 1:50)

n_pos = length(tau_pos)
n_neg = length(tau_neg)

gmm_moments = CSV.read(projectdir("adni/data/component_moments.csv"), DataFrame)
#gmm_moments2 = CSV.read(projectdir("data/adni-data/component_moments-bothcomps.csv"), DataFrame)
ubase, upath = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(ubase)
cc = quantile.(upath, .99)
#-------------------------------------------------------------------------------
# Connectome + ODEE
#-------------------------------------------------------------------------------
L = laplacian_matrix(c)

vols = [get_vol(data, i) for i in tau_pos]
init_vols = [v[:,1] for v in vols]
max_norm_vols = reduce(hcat, [v ./ maximum(v) for v in init_vols])
mean_norm_vols = vec(mean(max_norm_vols, dims=2))
Lv = sparse(inv(diagm(mean_norm_vols)) * L)

function NetworkLocalFKPP(du, u, p, t; L = L, u0 = u0, cc = cc)
    du .= -p[1] * L * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

_subdata = [calc_suvr(data, i) for i in tau_pos]
subdata = [normalise(sd, u0, cc) for sd in _subdata]

foursubs_idx = findall(x -> size(x, 2) == 4, subdata)
foursubs = subdata[foursubs_idx]

initial_conditions = [sd[:,1] for sd in subdata]

times =  [get_times(data, i) for i in tau_pos]

prob = ODEProblem(NetworkLocalFKPP,
                  initial_conditions[1], 
                  (0.,15.), 
                  [1.0,1.0])
                  
sol = solve(prob, Tsit5())

#-------------------------------------------------------------------------------
# Posterior simulations
#-------------------------------------------------------------------------------

pst = deserialize(projectdir("adni/chains/local-fkpp/pst-taupos-4x2000-three.jls"));

meanpst = mean(pst);
params = [[meanpst[Symbol("ρ[$i]"), :mean], meanpst[Symbol("α[$i]"), :mean]] for i in foursubs_idx];
meansols = [solve(
    ODEProblem(NetworkLocalFKPP, 
               init, (0.0,15.0), p), 
               Tsit5(), saveat=0.1) 
    for (init, t, p) in zip(initial_conditions[foursubs_idx], times[foursubs_idx], params)];


sols = Vector{Vector{Array{Float64}}}();

for i in foursubs_idx
    isols = Vector{Array{Float64}}()
    for s in 1:8:8000
        params = [pst[Symbol("ρ[$i]")][s], pst[Symbol("α[$i]")][s]]
        σ = pst[:σ][s]
        sol = solve(ODEProblem(NetworkLocalFKPP, initial_conditions[i], (0.0,15.0), params), Tsit5(), saveat=0.1)
        noise = (randn(size(Array(sol))) .* σ)
        push!(isols, Array(sol) .+ noise)
    end
    push!(sols, isols)
end

mean_all_sols = mean.(meansols, dims=1)
mean_sols = [mean.(sols[i], dims=1) for i in 1:3]
mean_data = mean.(foursubs, dims=1)


function get_quantiles(mean_sols)
    [vec(mapslices(x -> quantile(x, q), mean_sols, dims = 2)) for q in [0.975, 0.025, 0.5]]
end

using CairoMakie; CairoMakie.activate!()

begin
    nodes = zip([27, 25, 29],["Entorhinal Ctx", "Fusiform Gyrus", "Lateral Temporal"])
    cols = Makie.wong_colors()

    samples = 1000
    f = Figure(resolution=(2000,1500))
    g1 = [f[1,i] = GridLayout() for i in 1:3]
    g2 = [f[2,i] = GridLayout() for i in 1:3]
    g3 = [f[3,i] = GridLayout() for i in 1:3]
    gs = [g1, g2, g3]
    ga = [f[4,i] = GridLayout() for i in 1:3]
    gl = f[:, 4] = GridLayout()
    for (i, node) in enumerate(nodes)
        for sub in 1:3
            q1, q2, q3 = get_quantiles(reduce(hcat, [sols[sub][i][node[1], :] for i in 1:samples]))

            ax = Axis(gs[i][sub][1, 1], 
                    ylabel="SUVR", ylabelsize=40,
                    yticks = 1.0:0.5:3.5, yticksize=20, yticklabelsize=30,
                    yminorticks = 1.0:0.25:3.25, yminorticksize=15,
                    yminorgridvisible=true, yminorticksvisible=true,
                    yminorgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
                    xlabel="Time / years", xlabelsize=40,
                    xticks = 0.0:5.0:15.0, xticksize=20, xticklabelsize=30,
                    xminorticks = 0.0:2.5:15.0, xminorticksize=15,
                    xminorgridvisible=true, xminorticksvisible=true, 
                    xminorgridcolor=RGBAf(0, 0, 0, 0.15), xgridcolor=RGBAf(0, 0, 0, 0.15))

            ylims!(ax, 1.0,3.25)
            xlims!(ax, -1.0,15)
            # if i < 3
                hidexdecorations!(ax, minorgrid=false, grid=false)
            # end
            if sub > 1
                hideydecorations!(ax, minorgrid=false, grid=false)
            end
            band!(0.0:0.1:15.0, q1, q2, color=(:grey, 0.5), label=" %95 quantile")
            lines!(meansols[sub].t, meansols[sub][node[1],:], color=(:red, 0.8), linewidth=3, label="Mean")
            scatter!(times[foursubs_idx][sub][1:3], subdata[foursubs_idx][sub][node[1],1:3], 
                    markersize=30, color=(cols[1], 0.7),label="Insample data")
            scatter!(times[foursubs_idx][sub][4], subdata[foursubs_idx][sub][node[1],4], 
                    markersize=30, color=(cols[2], 0.7),label="Outsample data")
            Label(f[i,0], node[2], tellheight=false, rotation=pi/2, fontsize=40)
        end
    end
    for sub in 1:3
        q1, q2, q3 = get_quantiles(transpose(reduce(vcat, mean_sols[sub])))

        ax = Axis(ga[sub][1, 1], 
                ylabel="SUVR", ylabelsize=40,
                yticks = 1.0:0.5:3.5, yticksize=20, yticklabelsize=30,
                yminorticks = 80.0:15.0:200.0, yminorticksize=15,
                yminorgridvisible=true, yminorticksvisible=true,
                yminorgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
                xlabel="Time / years", xlabelsize=40,
                xticks = 0.0:5.0:15.0, xticksize=20, xticklabelsize=30,
                xminorticks = 0.0:2.5:15.0, xminorticksize=15,
                xminorgridvisible=true, xminorticksvisible=true, 
                xminorgridcolor=RGBAf(0, 0, 0, 0.15), xgridcolor=RGBAf(0, 0, 0, 0.15))

        ylims!(ax, 1.0,3.25)
        xlims!(ax, -1.0,15)
        if sub > 1
            hideydecorations!(ax, minorgrid=false, grid=false)
        end
        band!(0.0:0.1:15.0, q1, q2, color=(:grey, 0.5), label=" %95 quantile")
        lines!(meansols[sub].t, mean_all_sols[sub][1,:], color=(:red, 0.8), linewidth=3, label="Mean")
        scatter!(times[foursubs_idx][sub][1:3], mean_data[sub][1,1:3], 
                markersize=30, color=(cols[1], 0.7),label="Insample data")
        scatter!(times[foursubs_idx][sub][4], mean_data[sub][1,4], 
                markersize=30, color=(cols[2], 0.7),label="Outsample data")
        Label(f[4,0], "Total", tellheight=false, rotation=pi/2, fontsize=40)
    end



    elem_1 = LineElement(color = (:red, 0.6), linewidth=5)
    elem_2 = PolyElement(color = (:lightgrey, 1.0))
    elem_3 = MarkerElement(color = (cols[1], 0.7), marker='●', markersize=30)
    elem_4 = MarkerElement(color = (cols[2], 0.7), marker='●', markersize=30)
    legend = Legend(gl[1,1],
           [elem_1, elem_2, elem_3, elem_4],
           [" Mean Predictions", " 95% Quantile", "In-sample data", "Out-sample data"],
           patchsize = (30, 30), rowgap = 10, labelsize=40, framevisible=false)
    f
end
save(projectdir("visualisation/inference/pstpred/output/pstpred-outsample.pdf"), f)

begin
    nodes = zip([27, 25, 29],["Entorhinal Ctx", "Fusiform Gyrus", "Lateral Temporal"])
    cols = ColorSchemes.seaborn_bright

    samples = 1000
    f = Figure(resolution=(2000,1250))
    g1 = [f[i,1] = GridLayout() for i in 1:3]
    g2 = [f[i,2] = GridLayout() for i in 1:3]
    g3 = [f[i,3] = GridLayout() for i in 1:3]
    gs = [g1, g2, g3]
    ga = [f[i,4] = GridLayout() for i in 1:3]
    gl = f[:, 5] = GridLayout()
    for (i, node) in enumerate(nodes)
        for sub in 1:3
            q1, q2, q3 = get_quantiles(reduce(hcat, [sols[sub][i][node[1], :] for i in 1:samples]))

            ax = Axis(gs[i][sub][1, 1], 
                    # ylabel="SUVR", ylabelsize=40,
                    yticks = 1.0:0.5:3.5, yticksize=20, yticklabelsize=30,
                    yminorticks = 1.0:0.25:3.25, yminorticksize=15,
                    yminorgridvisible=true, yminorticksvisible=true,
                    yminorgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
                    # xlabel="Time / years", xlabelsize=40,
                    xticks = 0.0:5.0:15.0, xticksize=20, xticklabelsize=30,
                    xminorticks = 0.0:2.5:15.0, xminorticksize=15,
                    xminorgridvisible=true, xminorticksvisible=true, 
                    xminorgridcolor=RGBAf(0, 0, 0, 0.15), xgridcolor=RGBAf(0, 0, 0, 0.15))

            ylims!(ax, 1.0,3.25)
            xlims!(ax, -1.0,15)
            if sub < 3
                hidexdecorations!(ax, minorgrid=false, grid=false)
            end
            if i > 1
                hideydecorations!(ax, minorgrid=false, grid=false)
            end
            band!(0.0:0.1:15.0, q1, q2, color=(:grey, 0.5), label=" %95 quantile")
            lines!(meansols[sub].t, meansols[sub][node[1],:], color=(:black, 0.8), linewidth=3, label="Mean")
            scatter!(times[foursubs_idx][sub][1:3], subdata[foursubs_idx][sub][node[1],1:3], 
                    markersize=30, color=(cols[1], 0.7),label="Insample data")
            scatter!(times[foursubs_idx][sub][4], subdata[foursubs_idx][sub][node[1],4], 
                    markersize=30, color=(cols[4], 0.7),label="Outsample data")
            Label(f[0,i], node[2], tellwidth=false,rotation=0, fontsize=40)
        end
    end
    Label(f[1:3,0], "SUVR", tellheight=false, rotation=pi/2, fontsize=40)

    for sub in 1:3
        q1, q2, q3 = get_quantiles(transpose(reduce(vcat, mean_sols[sub])))

        ax = Axis(ga[sub][1, 1], 
                ylabel="SUVR", ylabelsize=40,
                yticks = 1.0:0.5:3.5, yticksize=20, yticklabelsize=30,
                yminorticks = 1.0:0.25:3.25, yminorticksize=15,
                yminorgridvisible=true, yminorticksvisible=true,
                yminorgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
                # xlabel="Time / years", xlabelsize=40,
                xticks = 0.0:5.0:15.0, xticksize=20, xticklabelsize=30,
                xminorticks = 0.0:2.5:15.0, xminorticksize=15,
                xminorgridvisible=true, xminorticksvisible=true, 
                xminorgridcolor=RGBAf(0, 0, 0, 0.15), xgridcolor=RGBAf(0, 0, 0, 0.15))

        ylims!(ax, 1.0,3.25)
        xlims!(ax, -1.0,15)
        hideydecorations!(ax, minorgrid=false, grid=false)
        if sub < 3
            hidexdecorations!(ax, minorgrid=false, grid=false)
        end
        band!(0.0:0.1:15.0, q1, q2, color=(:grey, 0.5), label=" %95 quantile")
        lines!(meansols[sub].t, mean_all_sols[sub][1,:], color=(:black, 0.8), linewidth=3, label="Mean")
        scatter!(times[foursubs_idx][sub][1:3], mean_data[sub][1,1:3], 
                markersize=30, color=(cols[1], 0.7),label="Insample data")
        scatter!(times[foursubs_idx][sub][4], mean_data[sub][1,4], 
                markersize=30, color=(cols[4], 0.7),label="Outsample data")
        Label(f[0,4], "Mean", tellwidth=false, rotation=0, fontsize=40)
    end
    Label(f[4,1:4], "Time / years", tellwidth=false, rotation=0, fontsize=40)

    elem_1 = LineElement(color = (:black, 0.6), linewidth=5)
    elem_2 = PolyElement(color = (:lightgrey, 1.0))
    elem_3 = MarkerElement(color = (cols[1], 0.7), marker='●', markersize=30)
    elem_4 = MarkerElement(color = (cols[4], 0.7), marker='●', markersize=30)
    legend = Legend(gl[1,1],
           [elem_1, elem_2, elem_3, elem_4],
           [" Mean Predictions", " 95% Quantile", "In-sample data", "Out-sample data"],
           patchsize = (30, 30), rowgap = 10, labelsize=40, framevisible=false)
    f
end
save(projectdir("visualisation/inference/pstpred/output/pstpred-outsample-transpose.pdf"), f)