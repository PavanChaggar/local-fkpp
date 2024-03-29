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
using Distributed

addprocs(2; exeflags="--project=$(Base.active_project())")

@everywhere using Pkg
@everywhere Pkg.status()

@everywhere begin    
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
    using Distributed
end

@everywhere include(projectdir("functions.jl"))
  
#-------------------------------------------------------------------------------
# Load connectome, regional parameters and sort data
#-------------------------------------------------------------------------------
@everywhere include(projectdir("adni/inference/inference-preamble.jl"))

#-------------------------------------------------------------------------------
# Connectome + ODEE
#-------------------------------------------------------------------------------
@everywhere begin
    subsuvr = [calc_suvr(data, i) for i in tau_neg]
    _subdata = [normalise(sd, u0, cc) for sd in subsuvr]

    blsd = [sd .- u0 for sd in _subdata]
    nonzerosubs = findall(x -> sum(x) < 2, [sum(sd, dims=1) .== 0 for sd in blsd])

    L = laplacian_matrix(c)

    vols = [get_vol(data, i) for i in tau_neg]
    init_vols = [v[:,1] for v in vols]
    max_norm_vols = reduce(hcat, [v ./ maximum(v) for v in init_vols])
    mean_norm_vols = vec(mean(max_norm_vols, dims=2))
    Lv = sparse(inv(diagm(mean_norm_vols)) * L)

    subdata = _subdata[nonzerosubs]

    _times =  [get_times(data, i) for i in tau_neg]
    times = _times[nonzerosubs]
    maxt = maximum(reduce(vcat, times))
    n_neg = length(nonzerosubs)
end

@everywhere function make_prob_func(initial_conditions, ρ, α, times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[i], p=[ρ[i], α[i]], saveat=times[i])
    end
end

@everywhere function output_func(sol,i)
    (sol,false)
end

@everywhere function get_retcodes(es)
    [sol.retcode for sol in es]
end

@everywhere function vec_sol(es)
    reduce(vcat, [vec(sol) for sol in es])
end

#-------------------------------------------------------------------------------
# Inference 
#-------------------------------------------------------------------------------
@everywhere @model function localfkpp(data, prob, inits, times, n)
    σ ~ LogNormal(0.0, 1.0)
    
    Pm ~ LogNormal(0.0, 1.0)
    Ps ~ LogNormal(0.0, 1.0)

    Am ~ Normal(0.0, 1.0)
    As ~ LogNormal(0.0, 1.0)

    ρ ~ filldist(truncated(Normal(Pm, Ps), lower=0), n)
    α ~ filldist(Normal(Am, As), n)

    ensemble_prob = EnsembleProblem(prob, 
                                    prob_func=make_prob_func(inits, 
                                                             ρ, α, times), 
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

@everywhere function shuffle_samples(rng)
    Random.seed!(rng)
    shuffle_idx = shuffle(collect(1:72))

    shuffled_data = [sd[shuffle_idx,:] for sd in subdata]

    shuffled_vecsubdata = reduce(vcat, reduce(hcat, shuffled_data))

    shuffled_initial_conditions = [sd[:,1] for sd in shuffled_data]

    _u0 = u0[shuffle_idx]
    _cc = cc[shuffle_idx]

    function NetworkLocalFKPP(du, u, p, t; L = Lv, u0 = _u0, cc = _cc)
        du .= -p[1] * L * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
    end

    prob = ODEProblem(NetworkLocalFKPP, shuffled_initial_conditions[1], (0, maxt), [1.0,1.0]);
    solve(prob, Tsit5())

    m = localfkpp(shuffled_vecsubdata, prob, shuffled_initial_conditions, times, n_neg);
    m();

    n_samples = 1_000
    pst = sample(m, NUTS(0.8), n_samples, progress=true)
    return pst
end

pst = pmap(shuffle_samples, [1234, 4321])
serialize(projectdir("adni/new-chains/local-fkpp/shuffled/neg/length-free/pst-tauneg-dstr.jls"), chainscat(pst...))