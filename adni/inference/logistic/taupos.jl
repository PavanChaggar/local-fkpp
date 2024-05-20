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
function NetworkLogistic(du, u, p, t; u0 = u0, cc = cc)
    du .= p[1] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

function make_prob_func(initial_conditions, p, times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[i], p=[p[i]], saveat=times[i])
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

prob = ODEProblem(NetworkLogistic, 
                  initial_conditions[1], 
                  (0.,maximum(reduce(vcat, times))), 
                  1.0)
                  
sol = solve(prob, Tsit5())

ensemble_prob = EnsembleProblem(prob, prob_func=make_prob_func(initial_conditions, ones(n_pos), times), output_func=output_func)
ensemble_sol = solve(ensemble_prob, Tsit5(), trajectories=n_pos)

function get_retcodes(es)
    [sol.retcode for sol in es]
end

function vec_sol(es)
    reduce(vcat, [vec(sol) for sol in es])
end
#-------------------------------------------------------------------------------
# Inference 
#-------------------------------------------------------------------------------
@model function logistic(data, prob, initial_conditions, times, n)
    σ ~ LogNormal(0.0, 1.0)

    Am ~ Normal(0.0, 1.0)
    As ~ LogNormal(0.0, 1.0)

    α ~ filldist(Normal(Am, As), n)

    ensemble_prob = EnsembleProblem(prob, 
                                    prob_func=make_prob_func(initial_conditions, α, times), 
                                    output_func=output_func)

    ensemble_sol = solve(ensemble_prob, 
                         Tsit5(), 
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

# setadbackend(:zygote)
Random.seed!(1234)

m = logistic(vecsubdata, prob, initial_conditions, times, n_pos);
m();

n_chains = 4
n_samples = 2000
pst = sample(m, 
             Turing.NUTS(0.8),
             MCMCSerial(), 
             n_samples, 
             n_chains,
             progress=true)
serialize(projectdir("adni/new-chains/logistic/pst-taupos-$(n_chains)x$(n_samples).jls"), pst)

# calc log likelihood 
pst = deserialize(projectdir("adni/new-chains/logistic/pst-taupos-$(n_chains)x$(n_samples).jls"));
log_likelihood = pointwise_loglikelihoods(m, MCMCChains.get_sections(pst, :parameters));
serialize(projectdir("adni/new-chains/logistic/ll-taupos-$(n_chains)x$(n_samples).jls"), log_likelihood)