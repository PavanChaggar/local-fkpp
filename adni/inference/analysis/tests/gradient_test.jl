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
using AdvancedHMC
using Distributions
using Serialization
using DelimitedFiles, LinearAlgebra
using Random
using LinearAlgebra
include(projectdir("functions.jl"))
#-------------------------------------------------------------------------------
# Connectome and ROIs
#-------------------------------------------------------------------------------
connectome_path = Connectomes.connectome_path()
all_c = filter(Connectome(connectome_path; norm=true), 1e-2);

subcortex = filter(x -> x.Lobe == "subcortex", all_c.parc);
cortex = filter(x -> x.Lobe != "subcortex", all_c.parc);

c = slice(all_c, cortex) |> filter

mtl_regions = ["entorhinal", "Left-Amygdala", "Right-Amygdala"]
mtl = findall(x -> x ∈ mtl_regions, cortex.Label)
neo_regions = ["inferiortemporal", "middletemporal"]
neo = findall(x -> x ∈ neo_regions, cortex.Label)
#-------------------------------------------------------------------------------
# Data 
#-----------------------------------------------------------------------------
sub_data_path = projectdir("adni/data/AV1451_Diagnosis-STATUS-STIME-braak-regions.csv")
alldf = CSV.read(sub_data_path, DataFrame)

posdf = filter(x -> x.STATUS == "POS", alldf)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in cortex.ID]

data = ADNIDataset(posdf, dktnames; min_scans=3)

# Ask Jake where we got these cutoffs from? 
mtl_cutoff = 1.375
neo_cutoff = 1.395

mtl_pos = filter(x -> regional_mean(data, mtl, x) >= mtl_cutoff, 1:50)
neo_pos = filter(x -> regional_mean(data, neo, x) >= neo_cutoff, 1:50)

tau_pos = findall(x -> x ∈ unique([mtl_pos; neo_pos]), 1:50)
tau_neg = findall(x -> x ∉ tau_pos, 1:50)

n_pos = length(tau_pos)
n_neg = length(tau_neg)

gmm_moments = CSV.read(projectdir("adni/data/component_moments.csv"), DataFrame)
#gmm_moments2 = CSV.read(projectdir("data/adni-data/component_moments-bothcomps.csv"), DataFrame)
ubase, upath = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(ubase)
cc = quantile.(upath, .99)
#-------------------------------------------------------------------------------
# Connectome + ODEE
#-------------------------------------------------------------------------------
L = laplacian_matrix(c)

function NetworkLocalFKPP(du, u, p, t; L = L, u0 = u0, cc = cc)
    du .= -p[1] * L * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

function make_prob_func(initial_conditions, p, a, times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[i], p=[p[i], a[i]], saveat=times[i])
    end
end

function output_func(sol,i)
    ([vec(sol), sol.retcode],false)
end

function output_func2(sol,i)
    (sol,false)
end


subdata = [calc_suvr(data, i) for i in tau_pos]
for i in 1:n_pos
    normalise!(subdata[i], u0, cc)
end

vecsubdata = reduce(vcat, reduce(hcat, subdata))

initial_conditions = [sd[:,1] for sd in subdata]
times =  [get_times(data, i) for i in tau_pos]

prob = ODEProblem(NetworkLocalFKPP, 
                  initial_conditions[1], 
                  (0.,15.), 
                  [1.0,1.0])
                  
sol = solve(prob, AutoVern7(Rodas4()))

ensemble_prob = EnsembleProblem(prob, prob_func=make_prob_func(initial_conditions, ones(n_pos), ones(n_pos), times), output_func=output_func)
ensemble_sol = solve(ensemble_prob, Tsit5(), trajectories=n_pos)

@inline function allequal(x)
    length(x) < 2 && return true
    e1 = x[1]
    i = 2
    @inbounds for i=2:length(x)
        x[i] == e1 || return false
    end
    return true
end
function get_retcodes(es)
    [sol[2] for sol in es]
end

function vec_sol(es)
    reduce(vcat, [sol[1] for sol in es])
end

#-------------------------------------------------------------------------------
# Inference 
#-------------------------------------------------------------------------------
@model function localfkpp(data, prob, initial_conditions, times, n)
    σ ~ LogNormal(0, 1)
    
    Pm ~ LogNormal(0, 0.5)
    Ps ~ LogNormal(0, 0.5)

    Am ~ Normal(0, 1)
    As ~ LogNormal(0, 0.5)

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
        return nothing
    end
    vecsol = vec_sol(ensemble_sol)

    data ~ MvNormal(vecsol, σ^2 * I)
end

@model function localfkpp2(data, prob, initial_conditions, times, n)
    σ ~ LogNormal(0, 1)
    
    Pm ~ LogNormal(0, 0.5)
    Ps ~ LogNormal(0, 0.5)

    Am ~ Normal(0, 1)
    As ~ LogNormal(0, 0.5)

    ρ ~ filldist(truncated(Normal(Pm, Ps), lower=0), n)
    α ~ filldist(Normal(Am, As), n)

    ensemble_prob = EnsembleProblem(prob, 
                                    prob_func=make_prob_func(initial_conditions, ρ, α, times), 
                                    output_func=output_func2)

    ensemble_sol = solve(ensemble_prob, 
                         Tsit5(), 
                         abstol = 1e-9, 
                         reltol = 1e-9, 
                         trajectories=n, 
                         sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
    if !allequal([sol.retcode for sol in ensemble_sol]) 
        Turing.@addlogprob! -Inf
        println("failed")
        return nothing
    end
    vecsol = reduce(vcat, ensemble_sol)

    data ~ MvNormal(vecsol, σ^2 * I)
end

@model function localfkpp3(data, prob, initial_conditions, times, n)
    σ ~ LogNormal(0, 1)
    
    Pm ~ LogNormal(0, 0.5)
    Ps ~ LogNormal(0, 0.5)

    Am ~ Normal(0, 1)
    As ~ LogNormal(0, 0.5)

    ρ ~ filldist(truncated(Normal(Pm, Ps), lower=0), n)
    α ~ filldist(Normal(Am, As), n)

    for i in 1:n
        _prob = remake(prob, u0=initial_conditions[i], p = [ρ[i], α[i]], saveat=times[i])
        _sol = solve(_prob, Tsit5(), abstol=1e-9, reltol=1e-9, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

        data[i] ~ MvNormal(vec(_sol),  σ^2 * I)
    end
end
#setadbackend(:zygote)
m = localfkpp(vecsubdata, prob, initial_conditions, times, n_pos);
m();

@code_warntype m.f(
    m,
    Turing.VarInfo(m),
    Turing.SamplingContext(
        Random.GLOBAL_RNG, Turing.SampleFromPrior(), Turing.DefaultContext(),
    ),
    m.args...,
)

using LogDensityProblems
using LogDensityProblemsAD
using BenchmarkTools

function test_gradient(model, adbackend)
    var_info = Turing.VarInfo(model)
    ctx = Turing.DefaultContext()

    spl = Turing.DynamicPPL.Sampler(Turing.NUTS(.8))

    vi = Turing.DynamicPPL.VarInfo(var_info, spl, var_info[spl])
    f = LogDensityProblemsAD.ADgradient(adbackend, 
                                            Turing.LogDensityFunction(vi, model, spl, ctx))
    θ = vi[spl]
    return f, θ
end

Random.seed!(1234);

m = localfkpp3(vsubdata, prob, initial_conditions, times, n_pos);
m();

f, θ = test_gradient(m, Turing.Essential.ForwardDiffAD{40}());
LogDensityProblems.logdensity_and_gradient(f, θ)

@benchmark LogDensityProblems.logdensity_and_gradient(f, θ)

Random.seed!(1234);

m = localfkpp(vecsubdata, prob, initial_conditions, times, n_pos);
m();

f, θ = test_gradient(m, Turing.Essential.ZygoteAD());
LogDensityProblems.logdensity_and_gradient(f, θ)

@benchmark LogDensityProblems.logdensity_and_gradient(f, θ)

ensemble_prob = EnsembleProblem(prob, prob_func=make_prob_func(initial_conditions, ones(n_pos), ones(n_pos), times), output_func=output_func)
ensemble_sol = solve(ensemble_prob, Tsit5(), trajectories=n_pos)

ensemble_prob = EnsembleProblem(prob, prob_func=make_prob_func(initial_conditions, ones(n_pos), ones(n_pos), times), output_func=output_func)
ensemble_sol = solve(ensemble_prob, Tsit5(), trajectories=2)

function esumf(p)
    ensemble_prob = EnsembleProblem(prob, 
                                    prob_func=make_prob_func(initial_conditions, ones(2).*p, ones(2).*p, times), 
                                    output_func=output_func)

    ensemble_sol = solve(ensemble_prob, Tsit5(), trajectories=2)
    vec_sol(ensemble_sol) |> sum
end
esumf(1.0)
using Zygote, ForwardDiff
ForwardDiff.gradient(essum, ensemble_sol)

using DifferentialEquations
using SciMLSensitivity
using Zygote, ForwardDiff
using ComponentArrays

function decay(du, u, p, t)
    du .= -p[1] .* u
end

prob = ODEProblem(decay, [1.0], (0.0,5.0), [1.0])
sol = solve(prob, Tsit5(), saveat=0.1)

function sumf(p)
    sum(solve(remake(prob, p = p), Tsit5(), saveat=0.5, ))
end
sumf(1.0)
ForwardDiff.gradient(sumf, [1.0])
Zygote.gradient(sumf, [1.0])

function make_prob_func(u, p)
    function prob_func(prob,i,repeat)
        remake(prob, u0=ComponentVector(;u), p=ComponentVector(;p))
    end
end

function esum(p)
    ensemble_prob = EnsembleProblem(prob, prob_func=make_prob_func(ones(2), ones(2) .* p))
    ensemble_sol = solve(ensemble_prob, Tsit5(), EnsembleSerial(), trajectories=2)

    reduce(vcat, ensemble_sol) |> sum
end

esum(2.0)
ForwardDiff.gradient(esum, [2])
Zygote.gradient(esum, [2.0])