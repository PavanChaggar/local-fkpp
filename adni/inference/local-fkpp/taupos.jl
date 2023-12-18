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
connectome_path = Connectomes.connectome_path()
all_c = filter(Connectome(connectome_path; norm=true, weight_function = (n, l) -> n ), 1e-2);

subcortex = filter(x -> get_lobe(x) == "subcortex", all_c.parc);
cortex = filter(x -> get_lobe(x) != "subcortex", all_c.parc);

c = slice(all_c, cortex) |> filter

mtl_regions = ["entorhinal", "Left-Amygdala", "Right-Amygdala"]
mtl = findall(x -> x ∈ mtl_regions, get_label.(cortex))
neo_regions = ["inferiortemporal", "middletemporal"]
neo = findall(x -> x ∈ neo_regions, get_label.(cortex))
#-------------------------------------------------------------------------------
# Data 
#-----------------------------------------------------------------------------
sub_data_path = projectdir("adni/data/new_new_data/UCBERKELEY_TAU_6MM_18Dec2023_AB_STATUS.csv")

alldf = CSV.read(sub_data_path, DataFrame)

#posdf = filter(x -> x.STATUS == "POS", alldf)
posdf = filter(x -> x.AB_Status == 1, alldf)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in get_node_id.(cortex)]

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
ubase, upath = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(ubase)
cc = quantile.(upath, .99)
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

_subdata = [calc_suvr(data, i) for i in tau_pos]
subdata = [normalise(sd, u0, cc) for sd in _subdata]

vecsubdata = reduce(vcat, reduce(hcat, subdata))

initial_conditions = [sd[:,1] for sd in subdata]
times =  [get_times(data, i) for i in tau_pos]

prob = ODEProblem(NetworkLocalFKPP, 
                  initial_conditions[1], 
                  (0.,maximum(reduce(vcat, times))), 
                  [1.0,1.0])
                  
sol = solve(prob, Tsit5())

ensemble_prob = EnsembleProblem(prob, prob_func=make_prob_func(initial_conditions, ones(n_pos), ones(n_pos), times), output_func=output_func)
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

setadbackend(:zygote)
Random.seed!(1234)  

m = localfkpp(vecsubdata, prob, initial_conditions, times, n_pos);
m();

using BenchmarkTools
using TuringBenchmarking
using ReverseDiff, Zygote
suite = TuringBenchmarking.make_turing_suite(
    m,
    adbackends = [
        TuringBenchmarking.ZygoteAD()
    ]
);
@info "Turing.jl" run(suite)

println("starting inference")
n_chains = 4
n_samples = 2000
pst = sample(m, 
             Turing.NUTS(0.8), #, metricT=AdvancedHMC.DenseEuclideanMetric), 
             MCMCSerial(), 
             n_samples, 
             n_chains)
serialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-taupos-$(n_chains)x$(n_samples).jls"), pst)

#calc log likelihood 
log_likelihood = pointwise_loglikelihoods(m, MCMCChains.get_sections(pst, :parameters));
serialize(projectdir("adni/new-chains/local-fkpp/length-free/ll-taupos-$(n_chains)x$(n_samples).jls"), log_likelihood)