using Connectomes
using ADNIDatasets
using CSV, DataFrames
using DrWatson: projectdir
using DifferentialEquations
using Turing
using Distributions
using Serialization
using DelimitedFiles, LinearAlgebra
using Random
using LinearAlgebra, SparseArrays
include(projectdir("functions.jl"))
#-------------------------------------------------------------------------------
# Connectome and ROIs
#-------------------------------------------------------------------------------
include(projectdir("adni/inference/inference-preamble.jl"))
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

min_suvr = minimum(u0)
max_suvr = maximum(cc)

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

# function NetworkGlobalFKPP(du, u, p, t; Lv = Lv)
#     du .= -p[1] * Lv * u .+ p[2] .* u .* (1 .- ( u ./ p[3]))
# end
function NetworkGlobalFKPP(du, u, p, t; Lv = Lv)
    du .= -p[1] * Lv * (u .- p[3]) .+ p[2] .* (u .- p[3]) .* ((p[4] .- p[3]) .- (u .- p[3]))
end


function NetworkDiffusion(du, u, p, t; Lv = Lv)
    du .= -p[1] * Lv * (u .- u0)
end

function NetworkLogistic(du, u, p, t; Lv = Lv)
    du .= p[1] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

#-------------------------------------------------------------------------------
# Posteriors
#-------------------------------------------------------------------------------
local_pst = deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-taupos-1x2000-three.jls"));
global_pst = deserialize(projectdir("adni/new-chains/global-fkpp/scaled/pst-taupos-1x2000-three.jls"));
diffusion_pst = deserialize(projectdir("adni/new-chains/diffusion/length-free/pst-taupos-1x2000-three.jls"));
logistic_pst = deserialize(projectdir("adni/new-chains/logistic/pst-taupos-1x2000-three.jls"));

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

global_preds = simulate(NetworkGlobalFKPP, insample_inits, vcat.(global_params, min_suvr, max_suvr), times);

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
CairoMakie.activate!()
function getdiff(d, n)
    d[:,n] .- d[:,1]
end

function getdiff(d)
    d[:,end] .- d[:,1]
end

function get_quantiles(mean_sols)
    [vec(mapslices(x -> quantile(x, q), mean_sols, dims = 2)) for q in [0.975, 0.025, 0.5]]
end

begin
    cols = ColorSchemes.seaborn_colorblind[1:10]
    scan = 4
    titlesize = 40
    xlabelsize = 30
    ylabelsize = 30
    xticklabelsize = 20 
    yticklabelsize = 20
    f = Figure(size = (2000, 1000), fontsize=30)
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
            scatter!(Array(four_subdata[j][:, end]), Array(preds[j])[:, end],
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
            _data = getdiff(four_subdata[j])
            _preds = getdiff(preds[j])
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

mean_first_scan = vec(mean(reduce(hcat, [sd[:, 1] for sd in four_subdata]), dims=2))
mean_final_scan = vec(mean(reduce(hcat, [sd[:, end] for sd in four_subdata]), dims=2))
mean_final_pred = vec(mean(reduce(hcat, [sd[:, end] for sd in local_preds]), dims=2))

begin
    cols = ColorSchemes.seaborn_colorblind[1:10]
    scan = 4
    titlesize = 40
    xlabelsize = 30
    ylabelsize = 30
    xticklabelsize = 20 
    yticklabelsize = 20
    f = Figure(resolution = (1000, 500), fontsize=30)
    g1 = f[1,1] = GridLayout()
    start = 1.0
    stop = 2.0
    ax = Axis(g1[1,1],
                xlabel="Observed", 
                ylabel="Predicted", 
                xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                xticks = 1.0:0.5:2.5, yticks = 1.0:0.5:2.5,
                xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
                xminorgridvisible=true,yminorgridvisible=true,
                xminorticksvisible=true, yminorticksvisible=true,
                xminorticks=collect(start:0.25:stop),yminorticks=collect(start:0.25:stop))
    for j in 1:10
        scatter!(Array(four_subdata[j][:, end]), Array(local_preds[j])[:, end],
                    markersize=15, color=(cols[j], 0.75));
        xlims!(ax, 0.9,2.7)
        ylims!(ax, 0.9,2.7)
    end
    lines!(0.9:0.1:2.7, 0.9:0.1:2.7, color=:grey)
    ax = Axis(g1[1,2], 
                xlabel="Δ Observed",
                ylabel="Δ Predicted", 
                xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
                yticks = -0.25:0.25:2.5, xticks = -0.25:0.25:2.5,
                xminorgridvisible=true,yminorgridvisible=true,
                xminorticksvisible=true, yminorticksvisible=true,)
    for j in 1:10
        _data = getdiff(four_subdata[j])
        _preds = getdiff(local_preds[j])
        scatter!(_data, _preds, markersize=15, color=(cols[j], 0.75));
        xlims!(ax, -0.3,1.1)
        ylims!(ax, -0.3,1.1)
    end
    lines!(-0.3:0.01:1.05, -0.3:0.01:1.05, color=:grey)

    Label(g1[0,:], "Individual prediction", font=:bold,
    tellwidth=false, rotation=0, padding = (0, 0, 0, 0), fontsize=40)
    f
end
save(projectdir("visualisation/inference/model-selection/output/out-sample-fit-individual.pdf"), f)


begin
    cols = ColorSchemes.seaborn_colorblind[1:10]
    scan = 4
    titlesize = 50
    xlabelsize = 40
    ylabelsize = 40
    xticklabelsize = 30 
    yticklabelsize = 30
    f = Figure(size = (2000, 1000), fontsize=50)
    gt = f[1,1] = GridLayout()
    gl = f[2, 1] = GridLayout()
    gb = f[3, 1] = GridLayout()
    gl2 = f[4, 1] = GridLayout()
    for (i, (preds, title)) in enumerate(
        zip([local_preds, global_preds, diffusion_preds, logistic_preds], 
        ["Local FKPP", "Global FKPP", "Diffusion", "Logistic"]))
        mean_final_pred = vec(mean(reduce(hcat, [sd[:, end] for sd in preds]), dims=2))

        start = 1.0
        stop = 1.75
        ax = Axis(gt[1,i],
                ylabel="Predicted", 
                title=title, titlesize=titlesize,
                xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                xticks = 1.0:0.25:stop, yticks = 1.0:0.25:stop,
                xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
                xminorgridvisible=true,yminorgridvisible=true,
                xminorticksvisible=true, yminorticksvisible=true,
                xminorticks=collect(start:0.125:stop),yminorticks=collect(start:0.125:stop))
        scatter!(mean_final_scan, mean_final_pred,
                markersize=15, color=(cols[1], 0.75));
        if i > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        xlims!(ax, 0.95,stop + 0.05)
        ylims!(ax, 0.95,stop + 0.05)
        lines!(0.9:0.1:2.7, 0.9:0.1:2.7, color=:grey)

        ax = Axis(gb[1,i], 
                    ylabel="Δ Predicted", 
                    xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                    xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
                    yticks = -0.0:0.1:0.3, xticks =-0.0:0.1:0.3,
                    xminorgridvisible=true,yminorgridvisible=true,
                    xminorticksvisible=true, yminorticksvisible=true,)

        scatter!(mean_final_scan .- mean_first_scan, mean_final_pred .- mean_first_scan, 
                markersize=15, color=(cols[1], 0.75));
        if i > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        
        xlims!(ax, -0.025,0.325)
        ylims!(ax, -0.025,0.325)
        lines!(-0.3:0.01:1.05, -0.3:0.01:1.05, color=:grey)
    end
    Label(gl[1,1:4], "SUVR", tellwidth=false, rotation=0, padding = (0, 0, -10, -20), fontsize=30)
    Label(gl2[1,1:4], "Δ SUVR", tellwidth=false, rotation=0, padding = (0, 0, -10, -20), fontsize=30)
    # Label(f[0,:], "Regional prediction", font=:bold,
    # tellwidth=false, rotation=0, padding = (0, 0, 0, 0), fontsize=40)
    f
end
save(projectdir("visualisation/inference/model-selection/output/out-sample-fit-regional-average.pdf"), f)

begin
    cols = ColorSchemes.seaborn_colorblind[1:10]
    scan = 4
    titlesize = 40
    xlabelsize = 30
    ylabelsize = 30
    xticklabelsize = 20 
    yticklabelsize = 20
    f = Figure(resolution = (1000, 500), fontsize=30)
    g1 = f[1,1] = GridLayout()
    start = 1.0
    stop = 2.0
    ax = Axis(g1[1,1],
            xlabel="Observed", 
            ylabel="Predicted", 
            xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
            xticks = 1.0:0.25:2., yticks = 1.0:0.25:2.,
            xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
            xminorgridvisible=true,yminorgridvisible=true,
            xminorticksvisible=true, yminorticksvisible=true,
            xminorticks=collect(start:0.125:stop),yminorticks=collect(start:0.125:stop))
    scatter!(mean_final_scan, mean_final_pred,
            markersize=15, color=(cols[1], 0.75));
    xlims!(ax, 0.95,2.)
    ylims!(ax, 0.95,2.)
    lines!(0.9:0.1:2.7, 0.9:0.1:2.7, color=:grey)

    ax = Axis(g1[1,2], 
                xlabel="Δ Observed",
                ylabel="Δ Predicted", 
                xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
                yticks = -0.0:0.1:0.3, xticks =-0.0:0.1:0.3,
                xminorgridvisible=true,yminorgridvisible=true,
                xminorticksvisible=true, yminorticksvisible=true,)

    scatter!(mean_final_scan .- mean_first_scan, mean_final_pred .- mean_first_scan, 
            markersize=15, color=(cols[1], 0.75));
    xlims!(ax, -0.025,0.3)
    ylims!(ax, -0.025,0.3)
    lines!(-0.3:0.01:1.05, -0.3:0.01:1.05, color=:grey)

    Label(g1[0,:], "Regional prediction", font=:bold,
    tellwidth=false, rotation=0, padding = (0, 0, 0, 0), fontsize=40)
    f
end

sub_idx = findall(x -> size(x,2) == 2, outsample_subdata)[1]
five_pred_mean = solve(ODEProblem(NetworkLocalFKPP, insample_inits[sub_idx], (0.0,20.0), local_params[sub_idx]), Tsit5(), saveat=collect(0.0:0.1:20.0))
five_preds = [solve(
                ODEProblem(NetworkLocalFKPP, insample_inits[sub_idx], (0.0,20.0), [p, a]), 
                Tsit5(), saveat=collect(0.0:0.1:20.0)) .+ (randn(72,201) .* σ) for (p, a, σ) in 
                zip(local_ps[sub_idx], local_as[sub_idx], local_ss)];


right = [25, 27, 29]
left = [61, 63, 65]
begin
    cols = ColorSchemes.seaborn_bright
    f = Figure(size = (2000, 550), fontsize=20)
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
        scatter!(times[sub_idx][1:3], four_subdata[sub_idx][node,1:3], color=cols[1], markersize=25)
        scatter!(times[sub_idx][4:end], four_subdata[sub_idx][node,4:end], color=cols[4], markersize=25)
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
    scatter!(times[sub_idx][1:3], vec(mean(four_subdata[sub_idx][:,1:3], dims=1)), color=cols[1], markersize=25, label="In-sample data")
    scatter!(times[sub_idx][4:end], vec(mean(four_subdata[sub_idx][:,4:end], dims=1)), color=cols[4], markersize=25, label="Out-sample data")
    axislegend()
    Label(f[2,1:4], "Time / years", tellwidth=false, rotation=0, fontsize=30, padding=(0, 0, 0, -20))

    f
end
save(projectdir("visualisation/inference/model-selection/output/out-sample-trajectories.pdf"), f)

right = [25, 27, 29]
left = [61, 63, 65]
begin
    cols = ColorSchemes.seaborn_bright
    f = Figure(resolution = (2000, 700), fontsize=20)
    for (i, (left, right, title)) in enumerate(zip(left, right, ["Fusiform", "Entorhinal", "Inf. Temporal"]))
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

        #right
        q1, q2, q3 = get_quantiles(reduce(hcat, [five_preds[i][right, :] for i in 1:2000]))
        band!(0.0:0.1:20.0, q1, q2, color=(cols[4], 0.25))
        lines!(five_pred_mean.t, five_pred_mean[right, :], color=cols[4])
        scatter!(times[sub_idx][1:3], four_subdata[sub_idx][right,1:3], color=cols[4], markersize=25)
        scatter!(times[sub_idx][4:end], four_subdata[sub_idx][right,4:end], color=cols[4], markersize=25, marker=:rect)

        #left 
        q1, q2, q3 = get_quantiles(reduce(hcat, [five_preds[i][left, :] for i in 1:2000]))
        band!(0.0:0.1:20.0, q1, q2, color=(cols[1], 0.25))
        lines!(five_pred_mean.t, five_pred_mean[left, :], color=cols[1])
        scatter!(times[sub_idx][1:3], four_subdata[sub_idx][left,1:3], color=cols[1], markersize=25)
        scatter!(times[sub_idx][4:end], four_subdata[sub_idx][left,4:end], color=cols[1], markersize=25, marker=:rect)

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
    q1, q2, q3 = get_quantiles(transpose(reduce(vcat, [mean(five_preds[i][1:36,:], dims=1) for i in 1:2000])))
    band!(0.0:0.1:20.0, q1, q2, color=(cols[4], 0.25))
    lines!(five_pred_mean.t, vec(mean(five_pred_mean[1:36, :], dims=1)), color=cols[4])
    scatter!(times[sub_idx][1:3], vec(mean(four_subdata[sub_idx][1:36,1:3], dims=1)), color=cols[4], markersize=25)
    scatter!(times[sub_idx][4:end], vec(mean(four_subdata[sub_idx][1:36,4:end], dims=1)), color=cols[4], markersize=25, marker=:rect)

    q1, q2, q3 = get_quantiles(transpose(reduce(vcat, [mean(five_preds[i][37:72,:], dims=1) for i in 1:2000])))
    band!(0.0:0.1:20.0, q1, q2, color=(cols[1], 0.25))
    lines!(five_pred_mean.t, vec(mean(five_pred_mean[37:72,:], dims=1)), color=cols[1])
    scatter!(times[sub_idx][1:3], vec(mean(four_subdata[sub_idx][37:72,1:3], dims=1)), color=cols[1], markersize=25)
    scatter!(times[sub_idx][4:end], vec(mean(four_subdata[sub_idx][37:72,4:end], dims=1)), color=cols[1], markersize=25, marker=:rect)

    Label(f[2,1:4], "Time / years", tellwidth=false, rotation=0, fontsize=30, padding=(0, 0, 0, -20))

    elem_1 = MarkerElement(color = (cols[1], 0.7), marker='●', markersize=30)
    elem_2 = MarkerElement(color = (cols[1], 0.7), marker=:rect, markersize=30)
    elem_3 = MarkerElement(color = (cols[4], 0.7), marker='●', markersize=30)
    elem_4 = MarkerElement(color = (cols[4], 0.7), marker=:rect, markersize=30)
    elem_5 = LineElement(color = (cols[1], 0.6), linewidth=5)
    elem_6 = PolyElement(color = (cols[1], 0.5))
    elem_7 = LineElement(color = (cols[4], 0.6), linewidth=5)
    elem_8 = PolyElement(color = (cols[4], 0.5))

    legend = Legend(f[3,:],
           [elem_1, elem_2, 
            elem_5, elem_6 ,
           elem_3, elem_4, 
           elem_7, elem_8],
           ["Left In-sample", "Left Out-sample", 
            "Left Mean Pred.", "Left 95th Quantile", 
           "Right In-sample", "Right Out-sample", 
           "Right Mean Pred.", "Right 95th Quantile"],
           patchsize = (30, 20), rowgap = 10, colgap=20, 
           labelsize=40, framevisible=false, orientation=:horizontal,
           nbanks=2)
    f
end
save(projectdir("visualisation/inference/model-selection/output/out-sample-trajectories-hemi.pdf"), f)
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


right_nodes = [25, 35, 29]
left_nodes = [61, 71, 65]
cols = ColorSchemes.seaborn_bright

begin
f = Figure(resolution = (1500, 2000), fontsize=20)
for (j, s) in enumerate(1:8)
    g = f[j, :] = GridLayout()
    five_pred_mean = solve(ODEProblem(NetworkLocalFKPP, insample_inits[s], (0.0,20.0), local_params[s]), Tsit5(), saveat=collect(0.0:0.1:20.0))
    five_preds = [solve(
                    ODEProblem(NetworkLocalFKPP, insample_inits[s], (0.0,20.0), [p, a]), 
                    Tsit5(), saveat=collect(0.0:0.1:20.0)) .+ (randn(72,201) .* σ) for (p, a, σ) in 
                    zip(local_ps[s], local_as[s], local_ss)];

    for (i, (left_nodes, right_nodes, title)) in enumerate(zip(left_nodes, right_nodes, ["Fusiform", "Entorhinal", "Inf. Temporal"]))
        
        if j == 1
            ax = Axis(g[1,i], title = title, titlesize=40, ylabel="SUVR", ylabelsize=30,
            yticks = 1.0:0.5:3.5, yticksize=10, yticklabelsize=30,
            yminorticks = 1.0:0.25:3.25, yminorticksize=5,
            yminorgridvisible=true, yminorticksvisible=true,
            yminorgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
            xlabel="Time / years", xlabelsize=30,
            xticks = 0.0:5.0:20.0, xticksize=5, xticklabelsize=30,
            xminorticks = 0.0:2.5:20.0, xminorticksize=5,
            xminorgridvisible=true, xminorticksvisible=true, 
            xminorgridcolor=RGBAf(0, 0, 0, 0.15), xgridcolor=RGBAf(0, 0, 0, 0.15))
            ylims!(ax, 1.0, 3.4)
        else
        ax = Axis(g[1,i], ylabel="SUVR", ylabelsize=30,
                yticks = 1.0:0.5:3.5, yticksize=10, yticklabelsize=30,
                yminorticks = 1.0:0.25:3.25, yminorticksize=5,
                yminorgridvisible=true, yminorticksvisible=true,
                yminorgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
                xlabel="Time / years", xlabelsize=30,
                xticks = 0.0:5.0:20.0, xticksize=5, xticklabelsize=30,
                xminorticks = 0.0:2.5:20.0, xminorticksize=5,
                xminorgridvisible=true, xminorticksvisible=true, 
                xminorgridcolor=RGBAf(0, 0, 0, 0.15), xgridcolor=RGBAf(0, 0, 0, 0.15))
        ylims!(ax, 1.0, 3.4)
        end

        if i == 1
            if j < 8 
                hidexdecorations!(ax, grid=false, ticks=false,
                minorticks=false, minorgrid=false)
            end
        else
            hideydecorations!(ax, grid=false, ticks=false, 
                            minorticks=false, minorgrid=false)
            if j < 8 
                hidexdecorations!(ax, grid=false, ticks=false,
                minorticks=false, minorgrid=false)
            end
        end

        #right_nodes
        q1, q2, q3 = get_quantiles(reduce(hcat, [five_preds[i][right_nodes, :] for i in 1:2000]))
        band!(0.0:0.1:20.0, q1, q2, color=(cols[4], 0.25))
        lines!(five_pred_mean.t, five_pred_mean[right_nodes, :], color=cols[4])
        scatter!(times[s][1:3], four_subdata[s][right_nodes,1:3], color=cols[4], markersize=20)
        scatter!(times[s][4:end], four_subdata[s][right_nodes,4:end], color=cols[4], markersize=20, marker=:rect)

        #left_nodes 
        q1, q2, q3 = get_quantiles(reduce(hcat, [five_preds[i][left_nodes, :] for i in 1:2000]))
        band!(0.0:0.1:20.0, q1, q2, color=(cols[1], 0.25))
        lines!(five_pred_mean.t, five_pred_mean[left_nodes, :], color=cols[1])
        scatter!(times[s][1:3], four_subdata[s][left_nodes,1:3], color=cols[1], markersize=20)
        scatter!(times[s][4:end], four_subdata[s][left_nodes,4:end], color=cols[1], markersize=20, marker=:rect)
    end
    if j == 1
        ax = Axis(g[1, 4],  title="Mean", titlesize=40, ylabel="SUVR", ylabelsize=30,
        yticks = 1.0:0.5:3.5, yticksize=10, yticklabelsize=30,
        yminorticks = 1.0:0.25:3.25, yminorticksize=5,
        yminorgridvisible=true, yminorticksvisible=true,
        yminorgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
        xlabel="Time / years", xlabelsize=30,
        xticks = 0.0:5.0:20.0, xticksize=5, xticklabelsize=30,
        xminorticks = 0.0:2.5:20.0, xminorticksize=5,
        xminorgridvisible=true, xminorticksvisible=true, 
        xminorgridcolor=RGBAf(0, 0, 0, 0.15), xgridcolor=RGBAf(0, 0, 0, 0.15))
    else
    ax = Axis(g[1, 4],  ylabel="SUVR", ylabelsize=30,
                yticks = 1.0:0.5:3.5, yticksize=10, yticklabelsize=30,
                yminorticks = 1.0:0.25:3.25, yminorticksize=5,
                yminorgridvisible=true, yminorticksvisible=true,
                yminorgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
                xlabel="Time / years", xlabelsize=30,
                xticks = 0.0:5.0:20.0, xticksize=5, xticklabelsize=30,
                xminorticks = 0.0:2.5:20.0, xminorticksize=5,
                xminorgridvisible=true, xminorticksvisible=true, 
                xminorgridcolor=RGBAf(0, 0, 0, 0.15), xgridcolor=RGBAf(0, 0, 0, 0.15))
    end
    hideydecorations!(ax, grid=false, ticks=false, 
                        minorticks=false, minorgrid=false)
    if s == 16
    hidexdecorations!(ax, grid=false, ticks=false, ticklabels=false, label=false,
                            minorticks=false, minorgrid=false)
    else
        hidexdecorations!(ax, grid=false, ticks=false,
                            minorticks=false, minorgrid=false)
    end
    ylims!(ax, 1.0, 3.4)
    q1, q2, q3 = get_quantiles(transpose(reduce(vcat, [mean(five_preds[i][1:36,:], dims=1) for i in 1:2000])))
    band!(0.0:0.1:20.0, q1, q2, color=(cols[4], 0.25))
    lines!(five_pred_mean.t, vec(mean(five_pred_mean[1:36, :], dims=1)), color=cols[4])
    scatter!(times[s][1:3], vec(mean(four_subdata[s][1:36,1:3], dims=1)), color=cols[4], markersize=20)
    scatter!(times[s][4:end], vec(mean(four_subdata[s][1:36,4:end], dims=1)), color=cols[4], markersize=20, marker=:rect)

    q1, q2, q3 = get_quantiles(transpose(reduce(vcat, [mean(five_preds[i][37:72,:], dims=1) for i in 1:2000])))
    band!(0.0:0.1:20.0, q1, q2, color=(cols[1], 0.25))
    lines!(five_pred_mean.t, vec(mean(five_pred_mean[37:72,:], dims=1)), color=cols[1])
    scatter!(times[s][1:3], vec(mean(four_subdata[s][37:72,1:3], dims=1)), color=cols[1], markersize=20)
    scatter!(times[s][4:end], vec(mean(four_subdata[s][37:72,4:end], dims=1)), color=cols[1], markersize=20, marker=:rect)
    colgap!(g, 10)
    rowgap!(g, 5)
    # Label(f[2,1:4], "Time / years", tellwidth=false, rotation=0, fontsize=30, padding=(0, 0, 0, -20))

    # elem_1 = MarkerElement(color = (cols[1], 0.7), marker='●', markersize=30)
    # elem_2 = MarkerElement(color = (cols[1], 0.7), marker=:rect, markersize=30)
    # elem_3 = MarkerElement(color = (cols[4], 0.7), marker='●', markersize=30)
    # elem_4 = MarkerElement(color = (cols[4], 0.7), marker=:rect, markersize=30)
    # elem_5 = LineElement(color = (cols[1], 0.6), linewidth=5)
    # elem_6 = PolyElement(color = (cols[1], 0.5))
    # elem_7 = LineElement(color = (cols[4], 0.6), linewidth=5)
    # elem_8 = PolyElement(color = (cols[4], 0.5))

    # legend = Legend(f[3,:],
    #         [elem_1, elem_2, 
    #         elem_5, elem_6 ,
    #         elem_3, elem_4, 
    #         elem_7, elem_8],
    #         ["Left In-sample", "Left Out-sample", 
    #         "Left Mean Pred.", "Left 95th Quantile", 
    #         "Right In-sample", "Right Out-sample", 
    #         "Right Mean Pred.", "Right 95th Quantile"],
    #         patchsize = (30, 20), rowgap = 10, colgap=20, 
    #         labelsize=40, framevisible=false, orientation=:horizontal,
    #         nbanks=2)
    elem_1 = MarkerElement(color = (cols[1], 0.7), marker='●', markersize=30)
    elem_2 = MarkerElement(color = (cols[1], 0.7), marker=:rect, markersize=30)
    elem_3 = MarkerElement(color = (cols[4], 0.7), marker='●', markersize=30)
    elem_4 = MarkerElement(color = (cols[4], 0.7), marker=:rect, markersize=30)
    elem_5 = LineElement(color = (cols[1], 0.6), linewidth=5)
    elem_6 = PolyElement(color = (cols[1], 0.5))
    elem_7 = LineElement(color = (cols[4], 0.6), linewidth=5)
    elem_8 = PolyElement(color = (cols[4], 0.5))

    legend = Legend(f[9,:],
           [elem_1, elem_2, 
            elem_5, elem_6 ,
           elem_3, elem_4, 
           elem_7, elem_8],
           ["Left In-sample", "Left Out-sample", 
            "Left Mean Pred.", "Left 95th Quantile", 
           "Right In-sample", "Right Out-sample", 
           "Right Mean Pred.", "Right 95th Quantile"],
           patchsize = (30, 20), rowgap = 10, colgap=20, 
           labelsize=25, framevisible=false, orientation=:horizontal,
           nbanks=2) 
end
f
end
save(projectdir("visualisation/inference/model-selection/output/out-sample-trajectories-9-16.pdf"), f)
