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
using SparseArrays
include(projectdir("functions.jl"))

#-------------------------------------------------------------------------------
# Connectome and ROIs
#-------------------------------------------------------------------------------
connectome_path = Connectomes.connectome_path()
all_c = filter(Connectome(connectome_path; norm=true), 1e-2);

subcortex = filter(x -> x.Lobe == "subcortex", all_c.parc)
cortex = filter(x -> x.Lobe != "subcortex", all_c.parc)

c = slice(all_c, cortex) |> filter

mtl_regions = ["entorhinal", "Left-Amygdala", "Right-Amygdala"]
mtl = findall(x -> x ∈ mtl_regions, cortex.Label)
neo_regions = ["inferiortemporal", "middletemporal"]
neo = findall(x -> x ∈ neo_regions, cortex.Label)
#-------------------------------------------------------------------------------
# Data 
#-------------------------------------------------------------------------------
sub_data_path = projectdir("adni/data/AV1451_Diagnosis-STATUS-STIME-braak-regions.csv")
alldf = CSV.read(sub_data_path, DataFrame)

posdf = filter(x -> x.STATUS == "NEG", alldf)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in cortex.ID]

data = ADNIDataset(posdf, dktnames; min_scans=3)
n_subjects = length(data)


gmm_moments = CSV.read(projectdir("adni/data/component_moments.csv"), DataFrame)
ubase, upath = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(ubase)
cc = quantile.(upath, .99)
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

subsuvr = [calc_suvr(data, i) for i in 1:n_subjects]
_subdata = [normalise(sd, u0, cc) for sd in subsuvr]

blsd = [sd .- u0 for sd in _subdata]
nonzerosubs = findall(x -> sum(x) < 2, [sum(sd, dims=1) .== 0 for sd in blsd])

subdata = _subdata[nonzerosubs]
vecsubdata = reduce(vcat, reduce(hcat, subdata))

initial_conditions = [sd[:,1] for sd in subdata]
_times =  [get_times(data, i) for i in 1:n_subjects]
times = _times[nonzerosubs]

n_nz = length(subdata)

prob = ODEProblem(NetworkLogistic, 
                  initial_conditions[1], 
                  (0.,maximum(reduce(vcat, times))), 
                  [1.0,1.0])
                  
sol = solve(prob, Tsit5())

ensemble_prob = EnsembleProblem(prob, prob_func=make_prob_func(initial_conditions, ones(n_nz), times), output_func=output_func)
ensemble_sol = solve(ensemble_prob, Tsit5(), trajectories=n_nz)

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

Turing.setadbackend(:zygote)
Random.seed!(1234); 

m = logistic(vecsubdata, prob, initial_conditions, times, n_nz)
m();

n_chains = 4
pst = sample(m, 
             Turing.NUTS(0.8),
             MCMCSerial(), 
             2_000, 
             n_chains,
             progress=false)
serialize(projectdir("adni/chains/logistic/pst-abneg-$(n_chains)x2000.jls"), pst)