using ADNIDatasets
using Connectomes
using DifferentialEquations
using Turing 
using CSV, DataFrames
using DrWatson: projectdir
using DelimitedFiles, LinearAlgebra
using Plots
using SciMLSensitivity
using AdvancedHMC
using GLMakie, Serialization
include(projectdir("functions.jl"))
include(projectdir("adni/inference/inference-preamble.jl"))

L = laplacian_matrix(c)[1:36, 1:36]
vi = cc[1:36] .- u0[1:36]

function ScaledNetworkLocalFKPP(du, u, p, t; L = L, vi=vi)   
    du .= -p[1] * L * u .+ p[2:end] .* vi .* u .* ( 1 .- u )
end

inits = zeros(36)
inits[[27]] .= 0.02

prob = ODEProblem(ScaledNetworkLocalFKPP, 
                  inits, 
                  (0.,15.), 
                  [0.1, 1.0])
                  
sol = solve(prob, Tsit5(), saveat=1.0)

# Plots.plot(sol, labels=false)

d = vec(sol([3, 4, 5])) .+ (randn(length(vec(sol([3, 4, 5])))) .* 0.025)

@model function seeding(prob, ts, m)
    σ ~ InverseGamma(2, 3)

    ρ ~ truncated(Normal(), lower=0)
    α ~ truncated(Normal(), lower=0)
    t ~ Uniform(0, 20)
    u ~ Dirichlet(36, 0.5)

    _ts = t .+ ts
    tspan = convert.(eltype(t),(0.0,maximum(_ts)))

    _prob = remake(prob, u0= u .* m, tspan=tspan, p=[ρ, α])

    sol = solve(_prob, Tsit5(), abstol=1e-9, reltol=1e-6, saveat=_ts)

    if SciMLBase.successful_retcode(sol) !== true
        Turing.@addlogprob! -Inf
        println("failed")
        return nothing
    end

    d ~ MvNormal(vec(sol), σ^2 * I)
end

m = seeding(prob, [0.0, 1.0, 2.0], 0.5)
m()

# pst = m | (data = d,)
# pst()

# # using TuringBenchmarking, ADTypes

# # results = TuringBenchmarking.benchmark_model(pst; adbackends=[ADTypes.AutoForwardDiff(chunksize=40)])

# pst_samples = sample(pst, Turing.NUTS(0.8, metricT=AdvancedHMC.DenseEuclideanMetric), 1_000)

div_idx = findall( x -> x == 1, vec(pst_samples[:numerical_error]))
meanpst = mean(pst_samples)

sum(sol(3 - meanpst["t", :mean]))

scatter([meanpst["u[$i]", :mean] for i in 1:36] .* 0.2)
f = scatter(vec(pst_samples["u[27]"]))
scatter!(div_idx, vec(pst_samples["u[27]"])[div_idx])
f

#------------------------------------------------------------------------
# Data Application
#------------------------------------------------------------------------
data = ADNIDataset(posdf, dktnames; min_scans=4)
n_data = length(data)
# Ask Jake where we got these cutoffs from? 
mtl_cutoff = 1.375
neo_cutoff = 1.395

mtl_pos = filter(x -> regional_mean(data, mtl, x) >= mtl_cutoff, 1:n_data)
neo_pos = filter(x -> regional_mean(data, neo, x) >= neo_cutoff, 1:n_data)

tau_pos = findall(x -> x ∈ mtl_pos && x ∈ neo_pos, 1:n_data)

tau_data = data[tau_pos]

_subdata = calc_suvr.(tau_data)
subdata = [normalise(sd, u0, cc) for sd in _subdata]

t = get_times.(tau_data)

c = [(s[1:36,:] .- u0[1:36]) ./ (cc[1:36] .- u0[1:36]) for s in subdata]

f = scatter(t[1], c[1]', labels=false, color=:grey)
scatter!(t[1], c[1][27,:], labels=false, color=:red)
f

m = seeding(prob, t[2], 0.5)
m()

pst = m | (d = vec(c[2]),)
pst()

pst_samples = sample(pst, Turing.NUTS(0.8, metricT=AdvancedHMC.DenseEuclideanMetric), 1_000)

pst_samples = deserialize(projectdir("adni/chains/seed_samples.jls"))

meanpst = mean(pst_samples)

pst_inits = [meanpst["u[$i]", :mean] for i in 1:36]

right_cortex = filter(x -> get_hemisphere(x) == "right", cortex)
right_nodes = get_node_id.(right_cortex)

plot_roi(right_nodes, pst_inits ./ maximum(pst_inits), ColorSchemes.viridis)

scatter(pst_inits)

prob = ODEProblem(ScaledNetworkLocalFKPP, 
                  pst_inits, 
                  (0.,15.), 
                  [meanpst[:ρ, :mean], meanpst[:α, :mean]])
                  
sol = solve(prob, Tsit5(), saveat=meanpst[:t, :mean] .+ t[2])

scatter(sum(Array(sol), dims=1), sum(c[2], dims=1))

scatter(c[2][:, end], sol[end], xlims=(0.0,1.0), ylims=(0.0,1.0))
