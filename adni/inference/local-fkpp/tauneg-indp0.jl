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

posdf = filter(x -> x.STATUS == "POS", alldf)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in cortex.ID]

data = ADNIDataset(posdf, dktnames; min_scans=3)

function regional_mean(data, rois, sub)
    subsuvr = calc_suvr(data, sub)
    mean(subsuvr[rois,end])
end

mtl_cutoff = 1.375
neo_cutoff = 1.395

mtl_pos = filter(x -> regional_mean(data, mtl, x) >= mtl_cutoff, 1:50)
neo_pos = filter(x -> regional_mean(data, neo, x) >= neo_cutoff, 1:50)

tau_pos = findall(x -> x ∈ unique([mtl_pos; neo_pos]), 1:50)
tau_neg = findall(x -> x ∉ tau_pos, 1:50)

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

vols = [get_vol(data, i) for i in tau_neg]
init_vols = [v[:,1] for v in vols]
max_norm_vols = reduce(hcat, [v ./ maximum(v) for v in init_vols])
mean_norm_vols = vec(mean(max_norm_vols, dims=2))
Lv = sparse(inv(diagm(mean_norm_vols)) * L)

function NetworkLocalFKPP(du, u, p, t; Lv = Lv, u0 = u0, cc = cc)
    du .= -p[1] * Lv * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

function make_prob_func(initial_conditions, p, a, times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[:,i], p=[p[i], a[i]], saveat=times[i])
    end
end

function output_func(sol,i)
    (sol,false)
end

subsuvr = [calc_suvr(data, i) for i in tau_neg]
_subdata = [normalise(sd, u0, cc) for sd in subsuvr]

blsd = [sd .- u0 for sd in _subdata]
nonzerosubs = findall(x -> sum(x) < 2, [sum(sd, dims=1) .== 0 for sd in blsd])
nonzerosubs = setdiff(nonzerosubs, [2, 7])
subdata = _subdata[nonzerosubs]
vecsubdata = reduce(vcat, reduce(hcat, subdata))

initial_conditions = reduce(hcat, [sd[:,1] for sd in subdata])
_times =  [get_times(data, i) for i in tau_neg]
times = _times[nonzerosubs]

n_neg = length(nonzerosubs)

prob = ODEProblem(NetworkLocalFKPP, 
                  initial_conditions[:,1], 
                  (0.,maximum(reduce(vcat, times))), 
                  [1.0,1.0])
                  
sol = solve(prob, Tsit5())

ensemble_prob = EnsembleProblem(prob, prob_func=make_prob_func(initial_conditions, ones(n_neg), ones(n_neg), times), output_func=output_func)
ensemble_sol = solve(ensemble_prob, Tsit5(), trajectories=n_neg)

function get_retcodes(es)
    [sol.retcode for sol in es]
end

function vec_sol(es)
    reduce(vcat, [vec(sol) for sol in es])
end
#-------------------------------------------------------------------------------
# Inference 
#-------------------------------------------------------------------------------
@model function localfkpp(data, prob, times, u0, cc, n)
    σ ~ LogNormal(0.0, 1.0)
    
    Pm ~ LogNormal(0.0, 1.0)
    Ps ~ LogNormal(0.0, 1.0)

    Am ~ Normal(0.0, 1.0)
    As ~ LogNormal(0.0, 1.0)

    ρ ~ filldist(truncated(Normal(Pm, Ps), lower=0), n)
    α ~ filldist(Normal(Am, As), n)

    u ~ arraydist(reduce(hcat, [Uniform.(u0, cc) for _ in 1:n]))

    ensemble_prob = EnsembleProblem(prob, 
                                    prob_func=make_prob_func(u, ρ, α, times), 
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

m = localfkpp(vecsubdata, prob, times, u0, cc, n_neg);
m();

pst = sample(m, 
             Turing.NUTS(0.8),
             1_000, 
             progress=true)

# serialize(projectdir("adni/chains/local-fkpp/pst-tauneg-1000-indp0.jls"), pst)

# function plot_u0!(chain, u0, n)
#     scatter!(u0, color=(:red, 0.5), label="data")
#     for i in 1:83
#         hist!(ax, vec(chain[Symbol("u[$i, $n]")]), bins=20, scale_to=0.6, offset=i, direction=:x, color=(:grey, 0.5), label="posterior")
#     end
# end

# for j in 1:21
#     f, ax = scatter(initial_conditions[:,j], color=(:red, 0.5), label="data")
#     for i in 1:72
#         hist!(ax, vec(pst2[Symbol("u[$i,$j]")]), bins=20, scale_to=0.6, offset=i, direction=:x, color=(:grey, 0.5), label="posterior")
#     end
#     ylims!(ax, 0.9, 2.0)
#     display(f)
# end

# meanpst = mean(pst2)
# for j in 1:21
#     inits = [meanpst["u[$i,$j]",:mean] for i in 1:72]
#     f = Figure(resolution=(500, 500))
#     ax = Axis(f[1,1])
#     scatter!(initial_conditions[:,j], inits)
#     lines!(0.9:0.1:2.0,0.9:0.1:2.0)
#     xlims!(ax, 0.9, 2.0)
#     ylims!(ax, 0.9, 2.0)
#     display(f)
# end