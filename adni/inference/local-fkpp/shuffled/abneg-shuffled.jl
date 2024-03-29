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
all_c = filter(Connectome(connectome_path; norm=true, weight_function = (n, l) -> n ./ l), 1e-2);

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
sub_data_path = projectdir("adni/data/new_data/UCBERKELEYAV1451_8mm_02_17_23_AB_Status.csv")
alldf = CSV.read(sub_data_path, DataFrame)

negdf = filter(x -> x.AB_Status == 0, alldf)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in cortex.ID]

data = ADNIDataset(negdf, dktnames; min_scans=3)
n_subjects = length(data)

gmm_moments = CSV.read(projectdir("adni/data/component_moments.csv"), DataFrame)
ubase, upath = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(ubase)
cc = quantile.(upath, .99)

#-------------------------------------------------------------------------------
# Connectome + ODEE
#-------------------------------------------------------------------------------
L = laplacian_matrix(c)

vols = [get_vol(data, i) for i in 1:n_subjects]
init_vols = [v[:,1] for v in vols]
max_norm_vols = reduce(hcat, [v ./ maximum(v) for v in init_vols])
mean_norm_vols = vec(mean(max_norm_vols, dims=2))
Lv = sparse(inv(diagm(mean_norm_vols)) * L)

function NetworkLocalFKPP(du, u, p, t; Lv = Lv, u0 = u0, cc = cc)
    du .= -p[1] * Lv * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

function make_prob_func(initial_conditions, p, a, u0, cc, idx, times)
    function prob_func(prob,i,repeat)
        _idx = idx[i]
        remake(prob, u0=initial_conditions[i], p=[p[i], a[i], u0[_idx], cc[_idx]], saveat=times[i])
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

function shuffle_cols(arr)
    idx = shuffle(collect(1:72))
    # reduce(hcat, [shuffle(view(arr, :, i)) for i in 1:size(arr, 2)])
    idx, arr[idx, :]
end

_times =  [get_times(data, i) for i in 1:n_subjects]
times = _times[nonzerosubs]

n_subjects = length(subdata)

# n_subjects = length(subdata)
# prob = ODEProblem(NetworkLocalFKPP, 
#                   initial_conditions[1], 
#                   (0.,maximum(reduce(vcat, times))), 
#                   [1.0,1.0])
                  

# sol = solve(prob, Tsit5())

# ensemble_prob = EnsembleProblem(prob, prob_func=make_prob_func(initial_conditions, ones(n_subjects), ones(n_subjects), times), output_func=output_func)
# ensemble_sol = solve(ensemble_prob, Tsit5(), trajectories=n_subjects)

function get_retcodes(es)
    [sol.retcode for sol in es]
end

function vec_sol(es)
    reduce(vcat, [vec(sol) for sol in es])
end
#-------------------------------------------------------------------------------
# Inference 
#-------------------------------------------------------------------------------
@model function localfkpp(data, prob, initial_conditions, times, u0, cc, idx, n)
    σ ~ LogNormal(0.0, 1.0)
    
    Pm ~ LogNormal(0.0, 1.0)
    Ps ~ LogNormal(0.0, 1.0)

    Am ~ Normal(0.0, 1.0)
    As ~ LogNormal(0.0, 1.0)

    ρ ~ filldist(truncated(Normal(Pm, Ps), lower=0), n)
    α ~ filldist(Normal(Am, As), n)

    ensemble_prob = EnsembleProblem(prob, 
                                    prob_func=make_prob_func(initial_conditions, 
                                                             ρ, α, 
                                                             u0, cc, idx, 
                                                             times), 
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

Turing.setadbackend(:forwarddiff)
Random.seed!(1234); 

for i in 1:10
    println("Starting chain $i")

    shuffles = shuffle_cols.(subdata)
    idx = [sh[1] for sh in shuffles]
    shuffled_data = [sh[2] for sh in shuffles]

    shuffled_vecsubdata = reduce(vcat, reduce(hcat, shuffled_data))

    shuffled_initial_conditions = [sd[:,1] for sd in shuffled_data]

    prob = ODEProblem(NetworkLocalFKPP, 
                    shuffled_initial_conditions[1], 
                    (0.,maximum(reduce(vcat, times))), 
                    [1.0,1.0, u0[idx[1]], cc[idx[1]]])

    m = localfkpp(shuffled_vecsubdata, prob, shuffled_initial_conditions, 
                  times, u0, cc, idx, n_subjects)
    m();

    n_samples = 1_000
    pst = sample(m,
                Turing.NUTS(5_00, 0.8),
                n_samples, 
                progress=true)
    serialize(projectdir("adni/chains/local-fkpp/shuffled/ab/pst-abneg-$(n_samples)-shuffled-$(i).jls"), pst)
end