using Pkg
cd("/home/chaggar/Projects/local-fkpp")
Pkg.activate(".")

using Connectomes
using ADNIDatasets
using CSV, DataFrames
using DrWatson: projectdir
using DifferentialEquations
using SciMLSensitivity
using Zygote
using Turing
using Distributions
using Serialization
using DelimitedFiles, LinearAlgebra
using Random
using LinearAlgebra
using SparseArrays
include(projectdir("functions.jl"))
#-------------------------------------------------------------------------------
# Load connectome, regional parameters and sort data
#-------------------------------------------------------------------------------
include(projectdir("adni/inference/inference-preamble.jl"))

#-------------------------------------------------------------------------------
# Connectome + ODEE
#-------------------------------------------------------------------------------
L = laplacian_matrix(c)

vols = get_vol.(pos_data)
init_vols = [v[:,1] for v in vols]
max_norm_vols = reduce(hcat, [v ./ maximum(v) for v in init_vols])
mean_norm_vols = vec(mean(max_norm_vols, dims=2))
Lv = sparse(inv(diagm(mean_norm_vols)) * L)


function NetworkGlobalFKPP(du, u, p, t; L = Lv)
    du .= -p[1] * L * (u .- p[3]) .+ p[2] .* (u .- p[3]) .* ((p[4] .- p[3]) .- (u .- p[3]))
end

function make_prob_func(initial_conditions, p, a, p_min, p_max, times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[i], p=[p[i], a[i], p_min, p_max], saveat=times[i])
    end
end

function output_func(sol,i)
    (sol,false)
end

_subdata = calc_suvr.(pos_data)
subdata = [normalise(sd, u0, cc) for sd in _subdata]

vecsubdata = reduce(vcat, reduce(hcat, subdata))

initial_conditions = [sd[:,1] for sd in subdata]
times =  get_times.(pos_data)
max_t = maximum(reduce(vcat, times))

min_suvr = minimum(u0)
max_suvr = maximum(cc)

prob = ODEProblem(NetworkGlobalFKPP, 
                  initial_conditions[1], 
                  (0.,max_t), 
                  [1.0,1.0, min_suvr, max_suvr])
                  
sol = solve(prob, Tsit5())

ensemble_prob = EnsembleProblem(prob, prob_func=make_prob_func(initial_conditions, ones(n_pos), ones(n_pos), min_suvr, max_suvr, times), output_func=output_func)
ensemble_sol = solve(ensemble_prob, Tsit5(), EnsembleSerial(), trajectories=n_pos)

function get_retcodes(es)
    [sol.retcode for sol in es]
end

function vec_sol(es)
    reduce(vcat, [vec(sol) for sol in es])
end

#-------------------------------------------------------------------------------
# Inference 
#-------------------------------------------------------------------------------
@model function globalfkpp(data, prob, initial_conditions, min_suvr, max_suvr, times, n)
    σ ~ InverseGamma(2, 3)
    
    Pm ~ LogNormal(0.0, 1.0)
    Ps ~ truncated(Normal(), lower=0)

    Am ~ Normal(0.0, 1.0)
    As ~ truncated(Normal(), lower=0)

    ρ ~ filldist(truncated(Normal(Pm, Ps), lower=0), n)
    α ~ filldist(Normal(Am, As), n)

    ensemble_prob = EnsembleProblem(prob, 
                                    prob_func=make_prob_func(initial_conditions, ρ, α, min_suvr, max_suvr, times), 
                                    output_func=output_func)

    ensemble_sol = solve(ensemble_prob, 
                         Tsit5(),
                         EnsembleSerial(),
                         abstol = 1e-9, 
                         reltol = 1e-9, 
                         trajectories=n, 
                         sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

    if !allequal(get_retcodes(ensemble_sol)) 
        Turing.@addlogprob! -Inf
        println("failed")
        return nothing
    end
    vecsol = vec_sol(ensemble_sol)

    data ~ MvNormal(vecsol, σ^2 * I)
end

Random.seed!(8953);

m = globalfkpp(vecsubdata, prob, initial_conditions, min_suvr, max_suvr, times, n_pos);
m();

println("starting inference")
n_chains = 1
n_samples = 2000
pst = sample(m,
             Turing.NUTS(0.8),
             MCMCSerial(),
             n_samples, 
             n_chains,
             progress=true)
serialize(projectdir("adni/new-chains/global-fkpp/scaled/pst-taupos-$(n_chains)x$(n_samples)-normal-6.jls"), pst)

# calc log likelihood 
log_likelihood = pointwise_loglikelihoods(m, MCMCChains.get_sections(pst, :parameters));
serialize(projectdir("adni/new-chains/global-fkpp/scaled/ll-taupos-$(n_chains)x$(n_samples)-normal-6.jls"), log_likelihood)