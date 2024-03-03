using ADNIDatasets
using Connectomes
using DifferentialEquations
using Turing 
using CSV, DataFrames
using DrWatson: projectdir
using DelimitedFiles, LinearAlgebra
using Plots
using SciMLSensitivity
include(projectdir("functions.jl"))
include(projectdir("adni/inference/inference-preamble.jl"))

L = laplacian_matrix(c)[1:36, 1:36]
vi = cc[1:36] .- u0[1:36]

function ScaledNetworkLocalFKPP(du, u, p, t; L = L, vi=vi)   
    du .= -p[1] * L * u .+ p[2:end] .* vi .* u .* ( 1 .- u )
end

inits = zeros(36)
inits[[27]] .= 0.1

prob = ODEProblem(ScaledNetworkLocalFKPP, 
                  inits, 
                  (0.,15.), 
                  [0.1, 1.0])
                  
sol = solve(prob, Tsit5(), saveat=[3.0, 4.0, 5.0])

Plots.plot(sol, labels=false)

d = vec(sol) .+ (randn(length(vec(sol))) .* 0.025)

@model function seeding(prob, ts)
    σ ~ InverseGamma(2, 3)

    ρ ~ truncated(Normal(), lower=0)
    α ~ truncated(Normal(), lower=0)
    t ~ Uniform(0, 5)
    u ~ Dirichlet(36, 0.1)

    _ts = t .+ ts
    tspan = convert.(eltype(t),(0.0,maximum(_ts)))

    _prob = remake(prob, u0= u .* 0.1, tspan=tspan, p=[ρ, α])

    sol = solve(_prob, Tsit5(), abstol=1e-9, reltol=1e-6, saveat=_ts)

    if SciMLBase.successful_retcode(sol) !== true
        Turing.@addlogprob! -Inf
        println("failed")
    end

    data ~ MvNormal(vec(sol), σ^2 * I)
end

m = seeding(prob, [0.0, 1.0, 2.0])
m()

pst = m | (data = d,)
pst()

using TuringBenchmarking, ADTypes

results = TuringBenchmarking.benchmark_model(pst; adbackends=[ADTypes.AutoForwardDiff(chunksize=40)])

pst_samples = sample(pst, NUTS(0.9), 1_000)

meanpst = mean(pst_samples)
scatter([meanpst["u[$i]", :mean] for i in 1:36] .* 0.2)
# #------------------------------------------------------------------------
# # Data Application
# #------------------------------------------------------------------------
# data = ADNIDataset(posdf, dktnames; min_scans=4)
# n_data = length(data)
# # Ask Jake where we got these cutoffs from? 
# mtl_cutoff = 1.375
# neo_cutoff = 1.395

# mtl_pos = filter(x -> regional_mean(data, mtl, x) >= mtl_cutoff, 1:n_data)
# neo_pos = filter(x -> regional_mean(data, neo, x) >= neo_cutoff, 1:n_data)

# tau_pos = findall(x -> x ∈ mtl_pos && x ∈ neo_pos, 1:n_data)

# taudata = data[tau_pos]

# _subdata = [calc_suvr(data, i) for i in tau_pos]
# subdata = [normalise(sd, u0, cc) for sd in _subdata]

# t = get_times.(subdata)

# c = [(s[1:36,:] .- u0[1:36]) ./ (cc[1:36] .- u0[1:36]) for s in subdata]

# m = seeding(prob, t[1])
# m()

# pst = m | (d = vec(c[1]),)
# pst()