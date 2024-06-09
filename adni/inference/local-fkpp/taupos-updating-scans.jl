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

vols = [get_vol(data, i) for i in tau_pos]
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

_subdata = calc_suvr.(pos_data)
updating_idx = findall(x -> size(x, 2) > 3, _subdata)
fixed_idx = findall(x -> size(x, 2) < 4, _subdata)

fixed_data = pos_data[fixed_idx]
updating_data = pos_data[updating_idx]

fixed_subdata = [normalise(sd, u0, cc) for sd in calc_suvr.(fixed_data)]
_updating_subdata = [normalise(sd, u0, cc) for sd in calc_suvr.(updating_data)]

function get_retcodes(es)
    [sol.retcode for sol in es]
end

function vec_sol(es)
    reduce(vcat, [vec(sol) for sol in es])
end
#-------------------------------------------------------------------------------
# Inference 
#-------------------------------------------------------------------------------
@model function localfkpp(data, prob, initial_conditions, times, n)
    σ ~ LogNormal(0.0, 1.0)

    Pm ~ LogNormal(0.0, 1.0) # LogNormal(0.0,1.0)
    Ps ~ LogNormal(0.0, 1.0)

    Am ~ Normal(0.0, 1.0) # Normal(0.0,1.0)
    As ~ LogNormal(0.0, 1.0)

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
end

for updates in [3]
    if updates == 1
        vecsubdata = reduce(vcat, reduce(hcat, fixed_subdata))

        initial_conditions = [sd[:,1] for sd in fixed_subdata]

        times = get_times.(fixed_data)

        maxt = maximum(reduce(vcat, times))

        prob = ODEProblem(NetworkLocalFKPP, 
                        initial_conditions[1], 
                        (0.,maxt), 
                        [1.0,1.0])
        n_pos = length(times)
    else
        updating_subdata = [sd[:, 1:updates] for sd in _updating_subdata]

        vecsubdata = [reduce(vcat, reduce(hcat, fixed_subdata)); reduce(vcat, reduce(hcat, updating_subdata))]

        initial_conditions = [[sd[:,1] for sd in fixed_subdata]; [sd[:,1] for sd in updating_subdata]]

        fixed_times = get_times.(fixed_data)
        updating_times = [t[1:updates] for t in get_times.(updating_data)]

        times = [fixed_times; updating_times]

        maxt = maximum(reduce(vcat, times))

        prob = ODEProblem(NetworkLocalFKPP, 
                        initial_conditions[1], 
                        (0.,maxt), 
                        [1.0,1.0])
        n_pos = length(times)
    end
    println(updates)
    println(length(times))
    m = localfkpp(vecsubdata, prob, initial_conditions, times, n_pos);
    m();

    Random.seed!(1234)

    n_chains = 1
    n_samples = 2_000
    pst = sample(m, 
                Turing.NUTS(0.8),
                n_samples, 
                progress=true)
    serialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-taupos-$(n_chains)x$(n_samples)-updated-$(updates).jls"), pst)
end