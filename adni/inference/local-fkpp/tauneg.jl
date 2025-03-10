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
# Connectome, ROIs
#-------------------------------------------------------------------------------
include(projectdir("adni/inference/inference-preamble.jl"))

#-------------------------------------------------------------------------------
# Data
#-------------------------------------------------------------------------------
subsuvr = calc_suvr.(neg_data)
_subdata = [normalise(sd, u0, cc) for sd in subsuvr]

blsd = [sd .- u0 for sd in _subdata]
nonzerosubs = findall(x -> sum(x) < 2, [sum(sd, dims=1) .== 0 for sd in blsd])

subdata = _subdata[nonzerosubs]
vecsubdata = reduce(vcat, reduce(hcat, subdata))

initial_conditions = [sd[:,1] for sd in subdata]
times =  get_times.(neg_data[nonzerosubs])

n_neg = length(neg_data[nonzerosubs])

#-------------------------------------------------------------------------------
# Connectome + ODE
#-------------------------------------------------------------------------------
L = laplacian_matrix(c)

vols = [get_vol(neg_data, i) for i in nonzerosubs]
init_vols = [v[:,1] for v in vols]
max_norm_vols = reduce(hcat, [v ./ maximum(v) for v in init_vols])
mean_norm_vols = vec(mean(max_norm_vols, dims=2))
Lv = sparse(inv(diagm(mean_norm_vols)) * L)

function NetworkLocalFKPP(du, u, p, t; L = Lv, u0 = u0, cc = cc)
    du .= -p[1] * L * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

function make_prob_func(initial_conditions, p, a, times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[i], p=[p[i], a[i]], saveat=times[i])
    end
end

function output_func(sol,i)
    (sol,false)
end


prob = ODEProblem(NetworkLocalFKPP, 
                  initial_conditions[1], 
                  (0.,maximum(reduce(vcat, times))), 
                  [1.0,1.0])
                  
sol = solve(prob, Tsit5())

ensemble_prob = EnsembleProblem(prob, prob_func=make_prob_func(initial_conditions, ones(n_neg), ones(n_neg), times), output_func=output_func)
ensemble_sol = solve(ensemble_prob, Tsit5(), trajectories=n_neg)

function get_retcodes(es)
    [SciMLBase.successful_retcode(sol) for sol in es]
end

function vec_sol(es)
    reduce(vcat, [vec(sol) for sol in es])
end

#-------------------------------------------------------------------------------
# Inference 
#-------------------------------------------------------------------------------
@model function localfkpp(data, prob, initial_conditions, times, n)
    σ ~ InverseGamma(2,3)
    
    Pm ~ LogNormal(0.0, 1.0) #LogNormal(0.0,1.0)
    Ps ~ LogNormal(0.0, 1.0) 

    Am ~ Normal(0.0,1.0)
    As ~ LogNormal(0.0, 1.0) 

    ρ ~ filldist(truncated(Normal(Pm, Ps), lower=0), n)
    α ~ filldist(truncated(Normal(Am, As), lower = 0), n)

    ensemble_prob = EnsembleProblem(prob, 
                                    prob_func=make_prob_func(initial_conditions, ρ, α, times), 
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

# Turing.setadbackend(:zygote)
Random.seed!(1234); 

m = localfkpp(vecsubdata, prob, initial_conditions, times, n_neg)
m();

println("starting inference")
n_chains = 1
n_samples = 2000
pst = sample(m, 
             Turing.NUTS(0.8), #, metricT=AdvancedHMC.DenseEuclideanMetric), 
             MCMCSerial(), 
             n_samples, 
             n_chains)
serialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-tauneg-$(n_chains)x$(n_samples)-lognormal.jls"), pst)

#calc log likelihood 
log_likelihood = pointwise_loglikelihoods(m, MCMCChains.get_sections(pst, :parameters));
serialize(projectdir("adni/new-chains/local-fkpp/length-free/ll-tauneg-$(n_chains)x$(n_samples)-lognormal.jls"), log_likelihood)