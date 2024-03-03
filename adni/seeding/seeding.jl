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
                  (0.,10.), 
                  [0.1, 1.0])
                  
sol = solve(prob, Tsit5(), saveat=[3.0, 4.0, 5.0])

Plots.plot(sol, labels=false)

d = clamp.(vec(sol) .+ (randn(length(vec(sol))) .* 0.025), 0.0, 1.0)

@model function seeding(prob, ts)
    σ ~ InverseGamma(2, 3)

    ρ ~ truncated(Normal(), lower=0)
    α ~ truncated(Normal(), lower=0)

    u ~ Dirichlet(36, 0.01)

    prob = remake(prob, u0 = u .* 0.1, p = [ρ, α])
                    
    sol = solve(prob, Tsit5(), abstol=1e-9, reltol=1e-6, saveat=ts)

    if SciMLBase.successful_retcode(sol) !== true
        Turing.@addlogprob! -Inf
        println("failed")
    end

    d ~ MvNormal(vec(sol), σ^2 * I)
end

m = seeding(prob, [3.0, 4.0, 5.0])
m()

pst = m | (d = d,)
pst()

pst_samples = sample(pst, NUTS(adtype=ADTypes.AutoZygote()), 1_000)

meanpst = mean(pst_samples)
scatter([meanpst["u[$i]", :mean] for i in 1:36] .* 0.2)