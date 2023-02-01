using Connectomes
using ADNIDatasets
using CSV, DataFrames
using DrWatson: projectdir
using DifferentialEquations
using Turing
using Distributions
using Serialization
using DelimitedFiles
using Random
using LinearAlgebra, SparseArrays
include(projectdir("functions.jl"))

#-------------------------------------------------------------------------------
# Connectome and ROIs
#-------------------------------------------------------------------------------
connectome_path = Connectomes.connectome_path()
all_c = filter(Connectome(connectome_path; norm=true), 1e-2);

subcortex = filter(x -> x.Lobe == "subcortex", all_c.parc)
cortex = filter(x -> x.Lobe != "subcortex", all_c.parc)

c = slice(all_c, cortex) |> filter

mtl_regions = ["entorhinal", "Left-Amygdala", "Right-Amygdala"]
mtl = findall(x -> x ∈ mtl_regions, cortex.Label)
neo_regions = ["inferiortemporal", "middletemporal"]
neo = findall(x -> x ∈ neo_regions, cortex.Label)
#-------------------------------------------------------------------------------
# Data 
#-------------------------------------------------------------------------------
sub_data_path = projectdir("adni/data/AV1451_Diagnosis-STATUS-STIME-braak-regions.csv")
alldf = CSV.read(sub_data_path, DataFrame)

posdf = filter(x -> x.STATUS == "POS", alldf)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in cortex.ID]

data = ADNIDataset(posdf, dktnames; min_scans=3)

function regional_mean(data, rois, sub)
    subsuvr = calc_suvr(data, sub)
    mean(subsuvr[rois,end])
end

mtl_cutoff = 1.375
neo_cutoff = 1.395

mtl_pos = filter(x -> regional_mean(data, mtl, x) >= mtl_cutoff, 1:50)
neo_pos = filter(x -> regional_mean(data, neo, x) >= neo_cutoff, 1:50)

tau_pos = findall(x -> x ∈ unique([mtl_pos; neo_pos]), 1:50)
tau_neg = findall(x -> x ∉ tau_pos, 1:50)

n_pos = length(tau_pos)
n_neg = length(tau_neg)

neo_only = findall(x -> x ∈ setdiff(neo_pos, mtl_pos), tau_pos)
mtl_only = findall(x -> x ∈ setdiff(tau_pos, setdiff(neo_pos, mtl_pos)), tau_pos)

gmm_moments = CSV.read(projectdir("adni/data/component_moments.csv"), DataFrame)
ubase, upath = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(ubase)
cc = quantile.(upath, .99);

#-------------------------------------------------------------------------------
# Connectome
#-------------------------------------------------------------------------------
L = laplacian_matrix(c)

vols = [get_vol(data, i) for i in tau_pos]
init_vols = [v[:,1] for v in vols]
max_norm_vols = reduce(hcat, [v ./ maximum(v) for v in init_vols])
mean_norm_vols = vec(mean(max_norm_vols, dims=2))
Lv = sparse(inv(diagm(mean_norm_vols)) * L)

#-------------------------------------------------------------------------------
# Local FKPP
#-------------------------------------------------------------------------------
function NetworkLocalFKPP(du, u, p, t; Lv = Lv, u0 = u0, cc = cc)
    du .= -p[1] * Lv * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

subdata = [calc_suvr(data, i) for i in tau_pos];
norm_subdata = [normalise(sd, u0, cc) for sd in subdata];
norm_initial_conditions = [sd[:,1] for sd in norm_subdata];
times =  [get_times(data, i) for i in tau_pos];

local_pst = deserialize(projectdir("adni/chains/local-fkpp/pst-taupos-4x2000-vc.jls"));

local_meanpst = mean(local_pst);
local_params = [[local_meanpst[Symbol("ρ[$i]"), :mean], local_meanpst[Symbol("α[$i]"), :mean]] for i in 1:27];
local_sols = [solve(ODEProblem(NetworkLocalFKPP, init, (0.0,5.0), p), Tsit5(), saveat=t) for (init, t, p) in zip(norm_initial_conditions, times, local_params)];

#-------------------------------------------------------------------------------
# Global FKPP
#-------------------------------------------------------------------------------
function NetworkGlobalFKPP(du, u, p, t; Lv = Lv)
    du .= -p[1] * Lv * u .+ p[2] .* u .* (1 .- ( u ./ p[3]))
end

max_suvr = maximum(reduce(vcat, reduce(hcat, subdata)))

initial_conditions = [sd[:,1] for sd in subdata]

global_pst = deserialize(projectdir("adni/chains/global-fkpp/pst-taupos-4x2000-vc.jls"));

global_meanpst = mean(global_pst);
global_params = [[global_meanpst[Symbol("ρ[$i]"), :mean], global_meanpst[Symbol("α[$i]"), :mean], max_suvr] for i in 1:27];
global_sols = [solve(ODEProblem(NetworkGlobalFKPP, init, (0.0,5.0), p), Tsit5(), saveat=t) for (init, t, p) in zip(initial_conditions, times, global_params)];

#-------------------------------------------------------------------------------
# Diffusion
#-------------------------------------------------------------------------------
function NetworkDiffusion(du, u, p, t; Lv = Lv)
    du .= -p[1] * Lv * u
end

diffusion_pst = deserialize(projectdir("adni/chains/diffusion/pst-taupos-4x2000-vc.jls"));

diffusion_meanpst = mean(diffusion_pst);
diffusion_params = [diffusion_meanpst[Symbol("ρ[$i]"), :mean] for i in 1:27];
diffusion_sols = [solve(ODEProblem(NetworkDiffusion, init, (0.0,5.0), p), Tsit5(), saveat=t) for (init, t, p) in zip(initial_conditions, times, diffusion_params)];

#-------------------------------------------------------------------------------
# Logistic
#-------------------------------------------------------------------------------
function NetworkLogistic(du, u, p, t; Lv = Lv)
    du .= p[1] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

logistic_pst = deserialize(projectdir("adni/chains/logistic/pst-taupos-4x2000.jls"));

logistic_meanpst = mean(logistic_pst);
logistic_params = [logistic_meanpst[Symbol("α[$i]"), :mean] for i in 1:27];
logistic_sols = [solve(ODEProblem(NetworkLogistic, init, (0.0,5.0), p), Tsit5(), saveat=t) for (init, t, p) in zip(norm_initial_conditions, times, logistic_params)];

#-------------------------------------------------------------------------------
# Model comparison
#-------------------------------------------------------------------------------
function calc_aic(pst)
    k = length(pst.name_map.parameters)
    maxP = maximum(pst[:lp])
    return 2 * ( log(k) - log(maxP) )
end

calc_aic(local_pst)
calc_aic(logistic_pst)
calc_aic(global_pst)
calc_aic(diffusion_pst)

#-------------------------------------------------------------------------------
# Predictions
#-------------------------------------------------------------------------------
using CairoMakie; CairoMakie.activate!()

function getdiff(d, n)
    d[:,n] .- d[:,1]
end

sols = global_sols;
plot_data = subdata;
begin
    f = Figure(resolution=(1500, 1000))
    gl = [f[1, i] = GridLayout() for i in 1:3]
    for i in 1:3
        scan = i + 1
        ax = Axis(gl[i][1,1], 
                xlabel="SUVR", 
                ylabel="Prediction", 
                title="Scan: $scan", 
                titlesize=26, xlabelsize=20, ylabelsize=20)
        if i > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        xlims!(ax, 0.8, 4.0)
        ylims!(ax, 0.8, 4.0)
        lines!(0.8:0.1:4.0, 0.8:0.1:4.0, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
        for i in 1:27
            if size(plot_data[i], 2) >= scan
                scatter!(plot_data[i][:,scan], sols[i][:,scan], marker='o', markersize=15, color=(:grey, 0.5))
            end
        end
    end

    gl = [f[2, i] = GridLayout() for i in 1:3]
    for i in 1:3
        scan = i + 1

        if scan < 4
            diffs = getdiff.(plot_data, scan)
            soldiff = getdiff.(sols, scan)
        else
            idx = findall(x -> size(x,2) == scan, plot_data)
            diffs = getdiff.(plot_data[idx], scan)
            soldiff = getdiff.(sols[idx], scan)
        end

        ax = Axis(gl[i][1,1], 
                xlabel="Δ SUVR",
                ylabel="Δ Prediction",
                titlesize=26, xlabelsize=20, ylabelsize=20, 
                xticks=collect(-1:0.5:1))
        if i > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        start = -1.0
        stop = 1.0
        xlims!(ax, start, stop)
        ylims!(ax, start, stop)
        lines!(start:0.1:stop, start:0.1:stop, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
        for i in eachindex(diffs)
            scatter!(diffs[i], soldiff[i], marker='o', markersize=15, color=(:grey, 0.5))
        end
    end

    f
end
save(projectdir("visualisation/inference/model-selection/global-fkpp-model-preds.pdf"), f)

#-------------------------------------------------------------------------------
# Predicted trajectories
#-------------------------------------------------------------------------------
function get_quantiles(mean_sols)
    [vec(mapslices(x -> quantile(x, q), mean_sols, dims = 2)) for q in [0.975, 0.025, 0.5]]
end

meansols = [solve(ODEProblem(NetworkLogistic, 
                             init, (0.0,15.0), p), 
                             Tsit5(), saveat=0.05) 
            for (init, t, p) in 
            zip(norm_initial_conditions, times, logistic_params)];

sols = Vector{Vector{Array{Float64}}}();

for (i, j) in enumerate(tau_pos)
    isols = Vector{Array{Float64}}()
    for s in 1:40:8000
        params = [logistic_pst[Symbol("α[$i]")][s]]
        σ = logistic_pst[:σ][s]
        sol = solve(ODEProblem(NetworkLogistic, norm_initial_conditions[i], (0.0,15.0), params), Tsit5(), saveat=0.1)
        noise = (randn(size(Array(sol))) .* σ)
        push!(isols, Array(sol) .+ noise)
    end
    push!(sols, isols)
end

node = 10

begin
    f = Figure(resolution=(1500,1800))
    g = f[1:4,:] = GridLayout()
    g2 = f[5:6,:] = GridLayout()
    ylabelsize = 30
    xlabelsize = 30
    for j in 1:5
        sub = 0 + j
        q1, q2, q3 = get_quantiles(reduce(hcat, [sols[sub][i][node, :] for i in 1:200]))

        ax = Axis(g[1, sub], ylabel="SUVR", ylabelsize=ylabelsize)
        ylims!(ax, 1.0,3.5)
        hidexdecorations!(ax, ticks = false, grid=false)
        if j > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        band!(0.0:0.1:15.0, q1, q2, color=(:grey, 0.5))
        lines!(meansols[sub].t, meansols[sub][node,:], color=(:red, 0.8), linewidth=3)
        scatter!(times[sub], plot_data[sub][node,:], color=:navy)
    end
    f
    for j in 1:5
        sub = 5 + j
        q1, q2, q3 = get_quantiles(reduce(hcat, [sols[sub][i][node, :] for i in 1:200]))

        ax = Axis(g[2, j], ylabel="SUVR", ylabelsize=ylabelsize)
        hidexdecorations!(ax, ticks = false, grid=false)
        ylims!(ax, 1.0,3.5)
        if j > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        band!(0.0:0.1:15.0, q1, q2, color=(:grey, 0.5))
        lines!(meansols[sub].t, meansols[sub][node,:], color=(:red, 0.8), linewidth=3)
        scatter!(times[sub], plot_data[sub][node,:], color=:navy)
    end
    f
    for j in 1:5
        sub = 10 + j
        q1, q2, q3 = get_quantiles(reduce(hcat, [sols[sub][i][node, :] for i in 1:200]))

        ax = Axis(g[3, j], ylabel="SUVR", ylabelsize=ylabelsize)
        hidexdecorations!(ax, ticks = false, grid=false)
        ylims!(ax, 1.0,3.5)
        if j > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        band!(0.0:0.1:15.0, q1, q2, color=(:grey, 0.5))
        lines!(meansols[sub].t, meansols[sub][node,:], color=(:red, 0.8), linewidth=3)
        scatter!(times[sub], plot_data[sub][node,:], color=:navy)
    end
    for j in 1:5
        sub = 15 + j
        q1, q2, q3 = get_quantiles(reduce(hcat, [sols[sub][i][node, :] for i in 1:200]))

        ax = Axis(g[4, j], ylabel="SUVR", ylabelsize=ylabelsize)
        hidexdecorations!(ax, ticks = false, grid=false)
        if j > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        ylims!(ax, 1.0,3.5)
        band!(0.0:0.1:15.0, q1, q2, color=(:grey, 0.5))
        lines!(meansols[sub].t, meansols[sub][node,:], color=(:red, 0.8), linewidth=3)
        scatter!(times[sub], plot_data[sub][node,:], color=:navy)
    end
    for j in 1:5
        sub = 20 + j
        q1, q2, q3 = get_quantiles(reduce(hcat, [sols[sub][i][node, :] for i in 1:200]))

        ax = Axis(g2[1, j], ylabel="SUVR", xlabel="Time / Years", 
                  ylabelsize=ylabelsize, xlabelsize=xlabelsize)
        if j > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        if j < 3
            hidexdecorations!(ax, ticks=false, grid=false)
        end
        ylims!(ax, 1.0,3.5)
        band!(0.0:0.1:15.0, q1, q2, color=(:grey, 0.5))
        lines!(meansols[sub].t, meansols[sub][node,:], color=(:red, 0.8), linewidth=3)
        scatter!(times[sub], plot_data[sub][node,:], color=:navy)
    end

    for j in 1:2
        sub = 25 + j
        q1, q2, q3 = get_quantiles(reduce(hcat, [sols[sub][i][node, :] for i in 1:200]))

        ax = Axis(g2[2, j], ylabel="SUVR", xlabel="Time / Years",
                  ylabelsize=ylabelsize, xlabelsize=xlabelsize)
        if j > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        ylims!(ax, 1.0,3.5)
        band!(0.0:0.1:15.0, q1, q2, color=(:grey, 0.5))
        lines!(meansols[sub].t, meansols[sub][node,:], color=(:red, 0.8), linewidth=3)
        scatter!(times[sub], plot_data[sub][node,:], color=:navy)
    end

    elem_1 = LineElement(color = (:red, 0.6), linewidth=5)
    elem_2 = MarkerElement(color = (:navy, 0.6), marker=:circ, markersize=20)
    elem_3 = PolyElement(color = (:lightgrey, 1.0))
    legend = Legend(g2[2,4:5],
           [elem_1, elem_3, elem_2],
           [" Mean Predictions", " 95% Quantile", " Observations"],
           patchsize = (100, 50), rowgap = 10, labelsize=40)
    rowgap!(g, 15)
    rowgap!(f.layout,15)
    rowgap!(g2,-20)
    f
end
f
#-------------------------------------------------------------------------------
#  Local FKPP CV
#-------------------------------------------------------------------------------
@inline function allequal(x)
    length(x) < 2 && return true
    e1 = x[1]
    i = 2
    @inbounds for i=2:length(x)
        x[i] == e1 || return false
    end
    return true
end

function make_prob_func(initial_conditions, p, a, times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[i], p=[p[i], a[i]], saveat=times[i])
    end
end

function output_func(sol,i)
    (sol,false)
end

@model function localfkpp(data, prob, initial_conditions, times, n)
    σ ~ LogNormal(0, 1)
    
    Pm ~ LogNormal(0, 0.5)
    Ps ~ LogNormal(0, 0.5)

    Am ~ Normal(0, 1)
    As ~ LogNormal(0, 0.5)

    ρ ~ filldist(truncated(Normal(Pm, Ps), lower=0), n)
    α ~ filldist(Normal(Am, As), n)

    ensemble_prob = EnsembleProblem(prob, 
                                    prob_func=make_prob_func(initial_conditions, ρ, α, times), 
                                    output_func=output_func)

    ensemble_sol = solve(ensemble_prob, 
                         Tsit5(), 
                         abstol = 1e-9, 
                         reltol = 1e-9, 
                         trajectories=n)
    vecsol = reduce(vcat, ensemble_sol)
    # for i in 1:length(data)
    #     data[i] .~ Normal.(vec(ensemble_sol[i]), σ)
    # end
    data ~ MvNormal(vecsol, σ^2 * I)
end

vecsubdata = reduce(vcat, reduce(hcat, norm_subdata))
prob = ODEProblem(NetworkLocalFKPP, 
                  norm_initial_conditions[1], 
                  (0.,5.0), 
                  [1.0,1.0])

m = localfkpp(vecsubdata, prob, norm_initial_conditions, times, n_pos);
m()

chains_params = Turing.MCMCChains.get_sections(local_pst, :parameters)
loglikelihoods = pointwise_loglikelihoods(m, chains_params)
ℓ_mat = reduce(hcat, values(loglikelihoods));
ℓ_arr = reshape(ℓ_mat, 1, size(ℓ_mat)...)

data = ArviZ.from_mcmcchains(
    local_pst,
    library = "Turing",
    log_likelihood = Dict("y" => ℓ_arr)
)

local_loo = ArviZ.loo(data)

local_loo = psis_loo(m, local_pst);
scatter(local_loo.psis_object.pareto_k)
findall(x -> x > 0.7, local_loo.psis_object.pareto_k)

#-------------------------------------------------------------------------------
#  Logistic Model CV
#-------------------------------------------------------------------------------
function make_prob_func(initial_conditions, p, times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[i], p=[p[i]], saveat=times[i])
    end
end

@model function logistic(data, prob, initial_conditions, times, n)
    σ ~ LogNormal(0, 1)

    Am ~ Normal(0, 1)
    As ~ LogNormal(0, 0.5)

    α ~ filldist(Normal(Am, As), n)

    ensemble_prob = EnsembleProblem(prob, 
                                    prob_func=make_prob_func(initial_conditions, α, times), 
                                    output_func=output_func)

    ensemble_sol = solve(ensemble_prob, 
                         Tsit5(), 
                         abstol = 1e-9, 
                         reltol = 1e-9, 
                         trajectories=n)

    vecsol = reduce(vcat, ensemble_sol)

    for i in eachindex(data)
        data[i] ~ Normal(vecsol[i], σ)
    end
end

vecsubdata = reduce(vcat, reduce(hcat, norm_subdata))
prob = ODEProblem(NetworkLogistic, 
                  norm_initial_conditions[1], 
                  (0.,5.0), 
                  [1.0])

m = logistic(vecsubdata, prob, norm_initial_conditions, times, n_pos);
m()

chains_params = Turing.MCMCChains.get_sections(logistic_pst, :parameters)
loglikelihoods = pointwise_loglikelihoods(m, chains_params)
lls = loglikelihoods["data"]

k = length(logistic_pst.name_map.parameters)
2 * (log(k) - log(maximum(lls)))

logistic_loo = psis_loo(m, logistic_pst);
scatter(local_loo.psis_object.pareto_k)
findall(x -> x > 0.7, local_loo.psis_object.pareto_k)

#-------------------------------------------------------------------------------
#  Global FKPP CV
#-------------------------------------------------------------------------------
function make_prob_func(initial_conditions, p, a, p_max, times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[i], p=[p[i], a[i], p_max], saveat=times[i])
    end
end

@model function globalfkpp(data, prob, initial_conditions, max_suvr, times, n)
    σ ~ LogNormal(0, 1)
    
    Pm ~ LogNormal(0, 0.5)
    Ps ~ LogNormal(0, 0.5)

    Am ~ Normal(0, 1)
    As ~ LogNormal(0, 0.5)

    ρ ~ filldist(truncated(Normal(Pm, Ps), lower=0), n)
    α ~ filldist(Normal(Am, As), n)

    ensemble_prob = EnsembleProblem(prob, 
                                    prob_func=make_prob_func(initial_conditions, ρ, α, max_suvr, times), 
                                    output_func=output_func)

    ensemble_sol = solve(ensemble_prob, 
                         Tsit5(), 
                         abstol = 1e-9, 
                         reltol = 1e-9, 
                         trajectories=n)

    vecsol = reduce(vcat, [vec(sol) for sol in ensemble_sol])

    # for sub in eachindex(data)
    #     data[sub] .~ Normal.(Array(ensemble_sol[sub]), σ)
    # end
    data ~ MvNormal(vecsol, σ^2 * I)
end

vecsubdata = reduce(vcat, reduce(hcat, subdata))
prob = ODEProblem(NetworkGlobalFKPP, 
                  initial_conditions[1], 
                  (0.,5.0), 
                  [1.0,1.0, max_suvr])

m = globalfkpp(vecsubdata, prob, initial_conditions, max_suvr, times, n_pos);
m()

chains_params = Turing.MCMCChains.get_sections(global_pst, :parameters)
loglikelihoods = pointwise_loglikelihoods(m, chains_params)
lls = loglikelihoods["data"]

k = length(global_pst.name_map.parameters)
global_aic = 2 * (log(k) - log(maximum(lls)))

global_loo = psis_loo(m, global_pst)
findall(x -> x > 1, global_loo.psis_object.pareto_k)