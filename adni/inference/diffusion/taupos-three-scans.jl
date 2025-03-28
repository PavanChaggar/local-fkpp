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

function NetworkDiffusion(du, u, p, t; Lv = Lv, u0=u0)
    du .= -p[1] * L * (u .- u0)
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
[normalise!(_subdata[i], u0, cc) for i in 1:n_pos]
subdata = [sd[:, 1:3] for sd in _subdata]

vecsubdata = reduce(vcat, reduce(hcat, subdata))

initial_conditions = [sd[:,1] for sd in subdata]

_times =  get_times.(pos_data)
times = [t[1:3] for t in _times]
max_t = maximum(reduce(vcat, times))

prob = ODEProblem(NetworkDiffusion, 
                  initial_conditions[1], 
                  (0.,max_t), 
                  1.0)
                  
sol = solve(prob, Tsit5())

ensemble_prob = EnsembleProblem(prob, prob_func=make_prob_func(initial_conditions, collect(1:n_pos), times), output_func=output_func)
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
@model function diffusion(data, prob, initial_conditions, times, n)
    σ ~ InverseGamma(2, 3)
    
    Pm ~ LogNormal(0, 1.0)
    Ps ~ truncated(Normal(), lower=0)

    ρ ~ filldist(truncated(Normal(Pm, Ps), lower=0), n)

    ensemble_prob = EnsembleProblem(prob, 
                                    prob_func=make_prob_func(initial_conditions, ρ, times), 
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
Random.seed!(1234);

m = diffusion(vecsubdata, prob, initial_conditions, times, n_pos);
m();

println("starting inference")
n_chains = 1
n_samples = 2_000
pst = sample(m, 
             Turing.NUTS(0.8),
             n_samples, 
             progress=true)
serialize(projectdir("adni/new-chains/diffusion/length-free/pst-taupos-$(n_chains)x$(n_samples)-three-normal.jls"), pst)

# # calc log likelihood 
# pst = deserialize(projectdir("adni/chains/diffusion/pst-taupos-4x2000.jls"));
# log_likelihood = pointwise_loglikelihoods(m, MCMCChains.get_sections(pst, :parameters));
# serialize(projectdir("adni/chains/diffusion/ll-taupos-$(n_chains)x$(n_samples).jls"), log_likelihood)