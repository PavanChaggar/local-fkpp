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
local_initial_conditions = [sd[:,1] for sd in norm_subdata];
times =  [get_times(data, i) for i in tau_pos];

local_pst = deserialize(projectdir("adni/chains/local-fkpp/pst-taupos-4x2000-vc.jls"));

local_meanpst = mean(local_pst);
local_params = [[local_meanpst[Symbol("ρ[$i]"), :mean], local_meanpst[Symbol("α[$i]"), :mean]] for i in 1:27];
local_sols = [solve(ODEProblem(NetworkLocalFKPP, init, (0.0,5.0), p), Tsit5(), saveat=t) for (init, t, p) in zip(local_initial_conditions, times, local_params)];

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
# Predictions
#-------------------------------------------------------------------------------

function getdiff(d, n)
    d[:,n] .- d[:,1]
end

sols = local_sols;
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
            if size(subdata[i], 2) >= scan
                scatter!(subdata[i][:,scan], sols[i][:,scan], marker='o', markersize=15)
            end
        end
    end

    gl = [f[2, i] = GridLayout() for i in 1:3]
    for i in 1:3
        scan = i + 1

        if scan < 4
            diffs = getdiff.(subdata, scan)
            soldiff = getdiff.(sols, scan)
        else
            idx = findall(x -> size(x,2) == scan, subdata)
            diffs = getdiff.(subdata[idx], scan)
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
            scatter!(diffs[i], soldiff[i], marker='o', markersize=15)
        end
    end

    f
end

#-------------------------------------------------------------------------------
# Model selection
#-------------------------------------------------------------------------------
function make_idata(m, pst, data, args...)
    chains_params = Turing.MCMCChains.get_sections(pst, :parameters)
    loglikelihoods = pointwise_loglikelihoods(m, chains_params)
    #nms = string.(keys(pst_pred))
    nms = keys(loglikelihoods)
    loglikelihoods_vals = getindex.(Ref(loglikelihoods), nms)
    n_samples, n_chains = size(pst[:n_steps])
    loglikelihoods_arr = Array{Float64}(undef, n_chains, n_samples, length(data))
    for j in 1:n_chains
        for i in 1:length(data)
            loglikelihoods_arr[j,:,i] .= loglikelihoods_vals[i]
        end
    end
    from_mcmcchains(pst;
                    # posterior_predictive=pst_pred,
                    log_likelihood=Dict("ll" => loglikelihoods_arr),
                    library="Turing",
                    observed_data=Dict("data" => data))
end

function make_prob_func(initial_conditions, p, a, p_max, times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[i], p=[p[i], a[i], p_max], saveat=times[i])
    end
end

function output_func(sol,i)
    (sol,false)
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
    for i in eachindex(data)
        data[i] ~ Normal(vecsol[i], σ)
    end
end

vecsubdata = reduce(vcat, reduce(hcat, subdata))
prob = ODEProblem(NetworkGlobalFKPP, 
                  initial_conditions[1], 
                  (0.,5.0), 
                  [1.0,1.0, max_suvr])

m = globalfkpp(vecsubdata, prob, initial_conditions, max_suvr, times, n_pos);
m()

global_loo = psis_loo(m, global_pst)

_log_likelihood = Turing.pointwise_loglikelihoods(
    m, MCMCChains.get_sections(global_pst, :parameters)
);

ynames = string.(keys(_log_likelihood))
log_likelihood_y = getindex.(Ref(_log_likelihood), ynames)
n_samples, n_chains = size(global_pst[:n_steps])
log_likelihood = Array{Float64}(undef, n_chains, n_samples, length(vecsubdata))
for j in 1:n_chains
    for i in 1:length(data)
        log_likelihood[j,:,i] .= log_likelihood_y[i][:,j]
    end
end

idata_turing = from_mcmcchains(
    global_pst;
    log_likelihood,
    observed_data=(; vecsubdata),
    library=Turing,
)
