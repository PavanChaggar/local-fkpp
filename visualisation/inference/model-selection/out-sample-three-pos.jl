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
using LinearAlgebra, SparseArrays
include(projectdir("functions.jl"))
#-------------------------------------------------------------------------------
# Connectome and ROIs
#-------------------------------------------------------------------------------
connectome_path = Connectomes.connectome_path()
all_c = filter(Connectome(connectome_path; norm=true, weight_function = (n, l) -> n ./ l), 1e-2);

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
sub_data_path = projectdir("adni/data/new_data/UCBERKELEYAV1451_8mm_02_17_23_AB_Status.csv")
alldf = CSV.read(sub_data_path, DataFrame)

posdf = filter(x -> x.AB_Status == 1, alldf)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in cortex.ID]

data = ADNIDataset(posdf, dktnames; min_scans=3)
n_data = length(data)
# Ask Jake where we got these cutoffs from? 
mtl_cutoff = 1.375
neo_cutoff = 1.395

mtl_pos = filter(x -> regional_mean(data, mtl, x) >= mtl_cutoff, 1:n_data)
neo_pos = filter(x -> regional_mean(data, neo, x) >= neo_cutoff, 1:n_data)

tau_pos = findall(x -> x ∈ unique([mtl_pos; neo_pos]), 1:n_data)
tau_neg = findall(x -> x ∉ tau_pos, 1:n_data)

n_pos = length(tau_pos)
n_neg = length(tau_neg)

gmm_moments = CSV.read(projectdir("adni/data/component_moments.csv"), DataFrame)
#gmm_moments2 = CSV.read(projectdir("data/adni-data/component_moments-bothcomps.csv"), DataFrame)
ubase, upath = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(ubase)
cc = quantile.(upath, .99)
#-------------------------------------------------------------------------------
# Pos data 
#-------------------------------------------------------------------------------
_subdata = [calc_suvr(data, i) for i in tau_pos]
[normalise!(_subdata[i], u0, cc) for i in 1:n_pos]

outsample_idx = findall(x -> size(x, 2) > 3, _subdata)

four_subdata = _subdata[outsample_idx]

insample_subdata = [sd[:, 1:3] for sd in _subdata]
insample_four_subdata = insample_subdata[outsample_idx]
insample_inits = [d[:,1] for d in insample_four_subdata]

outsample_subdata = [sd[:, 4:end] for sd in _subdata[outsample_idx]]

max_suvr = maximum(reduce(vcat, reduce(hcat, insample_subdata)))

_times =  [get_times(data, i) for i in tau_pos]
times = _times[outsample_idx]
insample_times = [t[1:3] for t in _times]

outsample_times = [t[4:end] for t in _times[outsample_idx]]

#-------------------------------------------------------------------------------
# Models
#-------------------------------------------------------------------------------
L = laplacian_matrix(c)

vols = [get_vol(data, i) for i in tau_pos]
init_vols = [v[:,1] for v in vols]
max_norm_vols = reduce(hcat, [v ./ maximum(v) for v in init_vols])
mean_norm_vols = vec(mean(max_norm_vols, dims=2))
Lv = sparse(inv(diagm(mean_norm_vols)) * L)

function NetworkLocalFKPP(du, u, p, t; Lv = Lv, u0 = u0, cc = cc)
    du .= -p[1] * Lv * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

function NetworkGlobalFKPP(du, u, p, t; Lv = Lv)
    du .= -p[1] * Lv * u .+ p[2] .* u .* (1 .- ( u ./ p[3]))
end

function NetworkDiffusion(du, u, p, t; Lv = Lv)
    du .= -p[1] * Lv * u
end

function NetworkLogistic(du, u, p, t; Lv = Lv)
    du .= p[1] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

#-------------------------------------------------------------------------------
# Posteriors
#-------------------------------------------------------------------------------
local_pst = deserialize(projectdir("adni/chains/local-fkpp/pst-taupos-1x2000-three.jls"));
global_pst = deserialize(projectdir("adni/chains/global-fkpp/pst-taupos-1x2000-three.jls"));
diffusion_pst = deserialize(projectdir("adni/chains/diffusion/pst-taupos-1x2000-three.jls"));
logistic_pst = deserialize(projectdir("adni/chains/logistic/pst-taupos-1x2000-three.jls"));

[sum(p[:numerical_error]) for p in [local_pst, global_pst, diffusion_pst, logistic_pst]]
#-------------------------------------------------------------------------------
# Local model
#-------------------------------------------------------------------------------
local_meanpst = mean(local_pst);

local_ps = [Array(local_pst[Symbol("ρ[$i]")]) for i in outsample_idx];
local_as = [Array(local_pst[Symbol("α[$i]")]) for i in outsample_idx];
local_ss = Array(local_pst[Symbol("σ")]);
local_params = [[local_meanpst[Symbol("ρ[$i]"), :mean], local_meanpst[Symbol("α[$i]"), :mean]] for i in outsample_idx];

function simulate(f, initial_conditions, params, times)
    max_t = maximum(reduce(vcat, times))
    [solve(
        ODEProblem(
            f, inits, (0, max_t), p
        ), 
        Tsit5(), abstol=1e-9, reltol=1e-9, saveat=t
    )
    for (inits, p, t) in zip(initial_conditions, params, times)
    ]
end

local_preds = simulate(NetworkLocalFKPP, insample_inits, local_params, times);
#-------------------------------------------------------------------------------
# Global model
#-------------------------------------------------------------------------------
global_meanpst = mean(global_pst);

global_params = [[global_meanpst[Symbol("ρ[$i]"), :mean], global_meanpst[Symbol("α[$i]"), :mean]] for i in outsample_idx];

global_preds = simulate(NetworkGlobalFKPP, insample_inits, vcat.(global_params, max_suvr), times);

#-------------------------------------------------------------------------------
# Diffusion model
#-------------------------------------------------------------------------------
diffusion_meanpst = mean(diffusion_pst);

diffusion_params = [[diffusion_meanpst[Symbol("ρ[$i]"), :mean]] for i in outsample_idx];

diffusion_preds = simulate(NetworkDiffusion, insample_inits, diffusion_params, times);

#-------------------------------------------------------------------------------
# Logistic model
#-------------------------------------------------------------------------------
logistic_meanpst = mean(logistic_pst);

logistic_params = [[logistic_meanpst[Symbol("α[$i]"), :mean]] for i in outsample_idx];

logistic_preds = simulate(NetworkLogistic, insample_inits, logistic_params, times);
#-------------------------------------------------------------------------------
# Tau Positive Prediction Plot
#-------------------------------------------------------------------------------
using CairoMakie, ColorSchemes, Colors

function getdiff(d, n)
    d[:,n] .- d[:,1]
end

function getdiff(d)
    d[:,end] .- d[:,1]
end

function get_quantiles(mean_sols)
    [vec(mapslices(x -> quantile(x, q), mean_sols, dims = 2)) for q in [0.975, 0.025, 0.5]]
end

five_pred_mean = solve(ODEProblem(NetworkLocalFKPP, insample_inits[3], (0.0,20.0), local_params[3]), Tsit5(), saveat=collect(0.0:0.1:20.0))
five_preds = [solve(
                ODEProblem(NetworkLocalFKPP, insample_inits[3], (0.0,20.0), [p, a]), 
                Tsit5(), saveat=collect(0.0:0.1:20.0)) .+ (randn(72,201) .* σ) for (p, a, σ) in 
                zip(local_ps[3], local_as[3], local_ss)];

begin
    cols = ColorSchemes.seaborn_colorblind[1:10]
    scan = 4
    titlesize = 40
    xlabelsize = 30
    ylabelsize = 30
    xticklabelsize = 20 
    yticklabelsize = 20
    f = Figure(resolution = (2000, 1000), fontsize=30)
    gt = f[1,1] = GridLayout()
    gl = f[2, 1] = GridLayout()
    gb = f[3, 1] = GridLayout()
    gl2 = f[4, 1] = GridLayout()
    for (i, (preds, title)) in enumerate(
                    zip([local_preds, global_preds, diffusion_preds, logistic_preds], 
                    ["Local FKPP", "Global FKPP", "Diffusion", "Logistic"]))
        start = 1.0
        stop = 2.5
        ax = Axis(gt[1,i], title=title, titlesize=titlesize,
                    # xlabel="SUVR", 
                    ylabel="Prediction", 
                    xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                    xticks = 1.0:0.5:2.5, yticks = 1.0:0.5:2.5,
                    xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
                    xminorgridvisible=true,yminorgridvisible=true,
                    xminorticksvisible=true, yminorticksvisible=true,
                    xminorticks=collect(start:0.25:stop),yminorticks=collect(start:0.25:stop))
        if i > 1
        hideydecorations!(ax, grid=false, ticks=false, minorgrid=false, minorticks=false) 
        end
        for j in 1:10
            scatter!(Array(four_subdata[j][:, scan]), Array(preds[j])[:, scan],
                     markersize=15, color=(cols[j], 0.75));
            xlims!(ax, 0.9,2.7)
            ylims!(ax, 0.9,2.7)
        end
        lines!(0.9:0.1:2.7, 0.9:0.1:2.7, color=:grey)
        ax = Axis(gb[1,i], 
                # xlabel="Δ SUVR",
                ylabel="Δ Prediction", 
                xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
                yticks = -0.25:0.25:2.5, xticks = -0.25:0.25:2.5,
                xminorgridvisible=true,yminorgridvisible=true,
                xminorticksvisible=true, yminorticksvisible=true,)
        if i > 1
            hideydecorations!(ax, grid=false, ticks=false) 
        end
        for j in 1:10
            _data = getdiff(four_subdata[j], scan)
            _preds = getdiff(preds[j], scan)
            scatter!(_data, _preds, markersize=15, color=(cols[j], 0.75));
            xlims!(ax, -0.3,1.05)
            ylims!(ax, -0.3,1.05)
        end
        lines!(-0.3:0.01:1.05, -0.3:0.01:1.05, color=:grey)
    end
    Label(gl[1,1:4], "SUVR", tellwidth=false, rotation=0, padding = (0, 0, -10, -20), fontsize=30)
    Label(gl2[1,1:4], "Δ SUVR", tellwidth=false, rotation=0, padding = (0, 0, -10, -20), fontsize=30)
    f
end
save(projectdir("visualisation/inference/model-selection/output/out-sample-fit.pdf"), f)

left = [25, 27, 29]
right = [61, 63, 65]
begin
    cols = ColorSchemes.seaborn_bright
    f = Figure(resolution = (2000, 550), fontsize=20)
    for (i, (node, title)) in enumerate(zip(right, ["Fusiform", "Entorhinal", "Inf. Temporal"]))
        ax = Axis(f[1,i], title=title, titlesize=40, ylabel="SUVR", ylabelsize=30,
                yticks = 1.0:0.5:3.5, yticksize=20, yticklabelsize=30,
                yminorticks = 1.0:0.25:3.25, yminorticksize=15,
                yminorgridvisible=true, yminorticksvisible=true,
                yminorgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
                # xlabel="Time / years", xlabelsize=40,
                xticks = 0.0:5.0:20.0, xticksize=20, xticklabelsize=30,
                xminorticks = 0.0:2.5:20.0, xminorticksize=15,
                xminorgridvisible=true, xminorticksvisible=true, 
                xminorgridcolor=RGBAf(0, 0, 0, 0.15), xgridcolor=RGBAf(0, 0, 0, 0.15))
        ylims!(ax, 1.0, 3.5)
        if i == 1
            hidexdecorations!(ax, grid=false, ticks=false, ticklabels=false, 
                            minorticks=false, minorgrid=false)
        else
            hideydecorations!(ax, grid=false, ticks=false, 
                            minorticks=false, minorgrid=false)
            hidexdecorations!(ax, grid=false, ticks=false, ticklabels=false, 
                            minorticks=false, minorgrid=false)
        end

        q1, q2, q3 = get_quantiles(reduce(hcat, [five_preds[i][node, :] for i in 1:2000]))
        band!(0.0:0.1:20.0, q1, q2, color=(:grey, 0.5))
        lines!(five_pred_mean.t, five_pred_mean[node, :], color=:black)
        scatter!(times[3][1:3], four_subdata[3][node,1:3], color=cols[1], markersize=25)
        scatter!(times[3][4:end], four_subdata[3][node,4:end], color=cols[4], markersize=25)
    end
    ax = Axis(f[1, 4], title="Mean", titlesize=40,
                yticks = 1.0:0.5:3.5, yticksize=20, yticklabelsize=30,
                yminorticks = 1.0:0.25:3.25, yminorticksize=15,
                yminorgridvisible=true, yminorticksvisible=true,
                yminorgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
                # xlabel="Time / years", xlabelsize=40,
                xticks = 0.0:5.0:20.0, xticksize=20, xticklabelsize=30,
                xminorticks = 0.0:2.5:20.0, xminorticksize=15,
                xminorgridvisible=true, xminorticksvisible=true, 
                xminorgridcolor=RGBAf(0, 0, 0, 0.15), xgridcolor=RGBAf(0, 0, 0, 0.15))
    hideydecorations!(ax, grid=false, ticks=false, 
                      minorticks=false, minorgrid=false)
    hidexdecorations!(ax, grid=false, ticks=false, ticklabels=false, 
                            minorticks=false, minorgrid=false)
    ylims!(ax, 1.0, 3.5)
    q1, q2, q3 = get_quantiles(transpose(reduce(vcat, [mean(five_preds[i], dims=1) for i in 1:2000])))
    band!(0.0:0.1:20.0, q1, q2, color=(:grey, 0.5), label="95% C.I.")
    lines!(five_pred_mean.t, vec(mean(five_pred_mean, dims=1)), color=:black, label="Mean pred.")
    scatter!(times[3][1:3], vec(mean(four_subdata[3][:,1:3], dims=1)), color=cols[1], markersize=25, label="In-sample data")
    scatter!(times[3][4:end], vec(mean(four_subdata[3][:,4:end], dims=1)), color=cols[4], markersize=25, label="Out-sample data")
    axislegend()
    Label(f[2,1:4], "Time / years", tellwidth=false, rotation=0, fontsize=30, padding=(0, 0, 0, -20))

    f
end
save(projectdir("visualisation/inference/model-selection/output/out-sample-trajectories.pdf"), f)


# out_sample_fourth_scans = [sd[:,4] for sd in four_subdata]
# mean_fourth_scan = mean(reduce(hcat, out_sample_fourth_scans), dims=2) |> vec
# mean_insample_inits = mean(reduce(hcat, [sd[:,1] for sd in four_subdata]), dims=2)

# begin
#     f = Figure(resolution = (2000, 1000))
#     for (i, _preds) in enumerate([local_preds, global_preds, diffusion_preds, logistic_preds])
#         ax = Axis(f[1,i])

#         _out_sample_fourth_preds = [p[:,4] for p in _preds]
#         _mean_fourth_preds = mean(reduce(hcat, _out_sample_fourth_preds), dims=2) |> vec

#         scatter!(mean_fourth_scan, _mean_fourth_preds)    
#         xlims!(ax, 1.0,2.5)
#         ylims!(ax, 1.0,2.5)
#         lines!(1.0:0.1:2.5, 1.0:0.1:2.5, color=:grey)

#         ax = Axis(f[2,i])
        
#         _out_sample_inits = [p[:,1] for p in _preds]
#         _mean_init_preds = mean(reduce(hcat, _out_sample_inits), dims=2) |> vec

#         scatter!(vec(mean_fourth_scan .- mean_insample_inits), 
#         vec(_mean_fourth_preds .- _mean_init_preds))   
#         xlims!(ax, -0.025,0.3)
#         ylims!(ax, -0.025,0.3)
#         lines!(-0.1:0.01:0.5, -0.1:0.01:0.5, color=:grey)
#     end
#     f
# end