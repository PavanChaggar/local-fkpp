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
# Connectome and ROIs
#-------------------------------------------------------------------------------
include(projectdir("adni/inference/inference-preamble.jl"))

#-------------------------------------------------------------------------------
# Data 
#-------------------------------------------------------------------------------
data = ADNIDataset(negdf, dktnames; min_scans=3, qc=true)
n_data = length(data)

mtl_cutoff = 1.375
neo_cutoff = 1.395

mtl_pos = filter(x -> regional_mean(data, mtl, x) >= mtl_cutoff, 1:n_data)
neo_pos = filter(x -> regional_mean(data, neo, x) >= neo_cutoff, 1:n_data)

tau_pos = findall(x -> x ∈ unique([mtl_pos; neo_pos]), 1:n_data)
tau_neg = findall(x -> x ∉ tau_pos, 1:n_data)

neg_data = data[tau_neg]

subsuvr = calc_suvr.(neg_data)
_subdata = [normalise(sd, u0, cc) for sd in subsuvr]

blsd = [sd .- u0 for sd in _subdata]
nonzerosubs = findall(x -> sum(x) < 2, [sum(sd, dims=1) .== 0 for sd in blsd])
goodsubs = setdiff(nonzerosubs, [15])
subdata = _subdata[goodsubs]
vecsubdata = reduce(vcat, reduce(hcat, subdata))

initial_conditions = [sd[:,1] for sd in subdata]
times =  [get_times(neg_data, i) for i in goodsubs]

n_subjects = length(subdata)
#-------------------------------------------------------------------------------
# Connectome + ODEE
#-------------------------------------------------------------------------------
L = laplacian_matrix(c)

vols = [get_vol(neg_data, i) for i in goodsubs]
init_vols = [v[:,1] for v in vols]
max_norm_vols = reduce(hcat, [v ./ maximum(v) for v in init_vols])
mean_norm_vols = vec(mean(max_norm_vols, dims=2))
Lv = sparse(inv(diagm(mean_norm_vols)) * L)

function NetworkLocalFKPP(du, u, p, t; Lv = Lv, u0 = u0, cc = cc)
    du .= -p[1] * Lv * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
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

ensemble_prob = EnsembleProblem(prob, prob_func=make_prob_func(initial_conditions, ones(n_subjects), ones(n_subjects), times), output_func=output_func)
ensemble_sol = solve(ensemble_prob, Tsit5(), trajectories=n_subjects)

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
    
    Pm ~ LogNormal(0.0, 1.0)
    Ps ~ truncated(Normal(), lower=0), # LogNormal(0.0, 1.0)

    Am ~ Normal(0.0, 1.0)
    As ~ truncated(Normal(), lower=0), # LogNormal(0.0, 1.0)

    ρ ~ filldist(truncated(Normal(Pm, Ps), lower=0), n)
    α ~ filldist(Normal(Am, As), n) 

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
    # for i in 1:n
    #     prob_n = remake(prob, u0 = initial_conditions[i], p = [ρ[i], α[i]])
    #     # solve ode at time points specific to each subject
    #     predicted = solve(
    #         prob_n,
    #         Tsit5(),
    #         abstol=1e-6, 
    #         reltol=1e-6,
    #         saveat=times[i],
    #         sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))
    #     )
    #     if !SciMLBase.successful_retcode(sol) 
    #         Turing.@addlogprob! -Inf
    #         println("failed")
    #         return nothing
    #     end
    #     Turing.@addlogprob! loglikelihood(MvNormal(vec(predicted), σ), data[i])
    # end
end

# Turing.setadbackend(:zygote)
Random.seed!(1234); 

m = localfkpp(vecsubdata, prob, initial_conditions, times, n_subjects)
m();

n_chains = 1
n_samples = 2_000
println("starting inference")
pst = sample(m, 
             Turing.NUTS(0.8), #, adtype=AutoZygote()),
             MCMCSerial(), 
             n_samples, 
             n_chains)
serialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-abneg-$(n_chains)x$(n_samples).jls"), pst)
