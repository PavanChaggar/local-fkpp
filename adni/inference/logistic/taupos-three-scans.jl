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
using LinearAlgebra, SparseArrays
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
sub_data_path = projectdir("adni/data/new_data/UCBERKELEYAV1451_8mm_02_17_23_AB_Status.csv")
alldf = CSV.read(sub_data_path, DataFrame)

posdf = filter(x -> x.AB_Status == 1, alldf)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in cortex.ID]

data = ADNIDataset(posdf, dktnames; min_scans=3)
n_data = length(data)
# Ask Jake where we got these cutoffs from? 
mtl_cutoff = 1.375
neo_cutoff = 1.395

mtl_pos = filter(x -> regional_mean(data, mtl, x) >= mtl_cutoff, 1:n_data)
neo_pos = filter(x -> regional_mean(data, neo, x) >= neo_cutoff, 1:n_data)

tau_pos = findall(x -> x ∈ unique([mtl_pos; neo_pos]), 1:n_data)
tau_neg = findall(x -> x ∉ tau_pos, 1:n_data)

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
function NetworkLogistic(du, u, p, t; u0 = u0, cc = cc)
    du .= p[1] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

function make_prob_func(initial_conditions, p, times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[:,i], p=[p[i]], saveat=times[i])
    end
end

function output_func(sol,i)
    (sol,false)
end

_subdata = [calc_suvr(data, i) for i in tau_pos]
[normalise!(_subdata[i], u0, cc) for i in 1:n_pos]

subdata = [sd[:, 1:3] for sd in _subdata]
vecsubdata = reduce(vcat, reduce(hcat, subdata))

function percent_signal(inits, u0, cc)
    (inits .- u0) ./ (cc .- u0)
end

initial_conditions_vec = [sd[:,1] for sd in subdata]
initial_conditions_var_prior = [percent_signal(inits, u0, cc) .* 0.5 for inits in initial_conditions_vec]
initial_conditions_arr = reduce(hcat, initial_conditions_vec)

_times =  [get_times(data, i) for i in tau_pos]
times = [t[1:3] for t in _times]
maxt = maximum(reduce(vcat, times))

prob = ODEProblem(NetworkLogistic, 
                  initial_conditions_arr[:,1], 
                  (0., maxt), 
                  [1.0])
                  
sol = solve(prob, Tsit5())

ensemble_prob = EnsembleProblem(prob, prob_func=make_prob_func(initial_conditions_arr, ones(n_pos), times), output_func=output_func)
ensemble_sol = solve(ensemble_prob, Tsit5(), trajectories=n_pos)
           
function get_retcodes(es)
    [sol.retcode for sol in es]
end

function vec_sol(es)
    reduce(vcat, [vec(sol) for sol in es])
end
#-------------------------------------------------------------------------------
# Inference 
#-------------------------------------------------------------------------------
@model function localfkpp(data, prob, times, inits, vars, u0, cc, n)
    σ ~ LogNormal(0.0, 1)

    Am ~ Normal(0.0, 1.0)
    As ~ LogNormal(0.0, 1.0)

    α ~ filldist(Normal(Am, As), n)

    u ~ arraydist(truncated.(Normal.(inits, vars .+ 1e-5), u0, cc))
    _inits = reshape(u, 72, n)

    ensemble_prob = EnsembleProblem(prob, 
                                    prob_func=make_prob_func(_inits, α, times), 
                                    output_func=output_func)

    ensemble_sol = solve(ensemble_prob, 
                         Tsit5(), 
                         abstol = 1e-6, 
                         reltol = 1e-6, 
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

setadbackend(:zygote)
Random.seed!(1234);

inits_vec = reduce(vcat, initial_conditions_vec)
inits_vars = reduce(vcat, initial_conditions_var_prior)
u0_vec = reduce(vcat, fill(u0, n_pos))
cc_vec = reduce(vcat, fill(cc, n_pos))

m = localfkpp(vecsubdata, prob, times,
              inits_vec, inits_vars,
              u0_vec, cc_vec, n_pos);
              
m();

n_chains = 1
pst = sample(m, 
             Turing.NUTS(0.8), #, metricT=AdvancedHMC.DenseEuclideanMetric), 
             MCMCSerial(), 
             2_000, 
             n_chains,
             progress=true)

serialize(projectdir("adni/chains/logistic/pst-taupos-$(n_chains)x2000-three-indp0.jls"), pst)