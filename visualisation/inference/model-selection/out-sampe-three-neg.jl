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
all_c = filter(Connectome(connectome_path; norm=true, weight_function = (n, l) -> n ./ l), 1e-2);

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
# Pos data 
#-------------------------------------------------------------------------------
_subdata = [calc_suvr(data, i) for i in tau_neg]
[normalise!(_subdata[i], u0, cc) for i in 1:n_neg]

blsd = [sd .- u0 for sd in _subdata]
nonzerosubs = findall(x -> sum(x) < 2, [sum(sd, dims=1) .== 0 for sd in blsd])

subdata = _subdata[nonzerosubs]

outsample_idx = findall(x -> size(x, 2) > 3, subdata)

four_subdata = subdata[outsample_idx]

insample_subdata = [sd[:, 1:3] for sd in subdata]
insample_four_subdata = insample_subdata[outsample_idx]
insample_inits = [d[:,1] for d in insample_four_subdata]

outsample_subdata = [sd[:, 4:end] for sd in subdata[outsample_idx]]

max_suvr = maximum(reduce(vcat, reduce(hcat, insample_subdata)))

_times =  [get_times(data, i) for i in tau_neg][nonzerosubs]
times = _times[outsample_idx]
insample_times = [t[1:3] for t in _times]

outsample_times = [t[4:end] for t in _times[outsample_idx]]

#-------------------------------------------------------------------------------
# Models
#-------------------------------------------------------------------------------
L = laplacian_matrix(c)

vols = [get_vol(data, i) for i in tau_neg[nonzerosubs]]
init_vols = [v[:,1] for v in vols]
max_norm_vols = reduce(hcat, [v ./ maximum(v) for v in init_vols])
mean_norm_vols = vec(mean(max_norm_vols, dims=2))
Lv = sparse(inv(diagm(mean_norm_vols)) * L)

function NetworkLocalFKPP(du, u, p, t; Lv = Lv, u0 = u0, cc = cc)
    du .= -p[1] * Lv * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

function NetworkLogistic(du, u, p, t; Lv = Lv)
    du .= p[1] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

#-------------------------------------------------------------------------------
# Posteriors
#-------------------------------------------------------------------------------
local_pst = deserialize(projectdir("adni/chains/local-fkpp/pst-tauneg-1x2000-three-indp0.jls"));
logistic_pst = deserialize(projectdir("adni/chains/logistic/pst-tauneg-1x2000-three-indp0.jls"));

[sum(p[:numerical_error]) for p in [local_pst, logistic_pst]]
#-------------------------------------------------------------------------------
# Local model
#-------------------------------------------------------------------------------
local_meanpst = mean(local_pst);

foursubs_vec_idx = reshape(1:1944, 72, 27)[:, outsample_idx];

local_all_inits_vec = [local_pst["u[$i]"] for i in vec(foursubs_vec_idx)]
local_all_inits = reshape(local_all_inits_vec, 72, 11)
local_inits = [transpose(reduce(hcat, local_all_inits[:, i])) for i in 1:11]

local_mean_inits = vec.(mean.(local_inits, dims=2))

local_ps = [Array(local_pst[Symbol("ρ[$i]")]) for i in outsample_idx];
local_as = [Array(local_pst[Symbol("α[$i]")]) for i in outsample_idx];
local_params = [[local_meanpst[Symbol("ρ[$i]"), :mean], local_meanpst[Symbol("α[$i]"), :mean]] for i in outsample_idx];

function simulate(f, initial_conditions, params, times)
    max_t = maximum(reduce(vcat, times))
    [solve(
        ODEProblem(
            f, inits, (0, max_t), p
        ), 
        Tsit5(), saveat=t
    )
    for (inits, p, t) in zip(initial_conditions, params, times)
    ]
end

local_preds = simulate(NetworkLocalFKPP, local_mean_inits, local_params, times);
#-------------------------------------------------------------------------------
# Logistic model
#-------------------------------------------------------------------------------
logistic_meanpst = mean(logistic_pst);

logistic_all_inits_vec = [logistic_pst["u[$i]"] for i in vec(foursubs_vec_idx)]
logistic_all_inits = reshape(logistic_all_inits_vec, 72, 11)
logistic_inits = [transpose(reduce(hcat, logistic_all_inits[:, i])) for i in 1:11]

logistic_mean_inits = vec.(mean.(logistic_inits, dims=2))

logistic_params = [[logistic_meanpst[Symbol("α[$i]"), :mean]] for i in outsample_idx];

logistic_preds = simulate(NetworkLogistic, logistic_mean_inits, logistic_params, times);
#-------------------------------------------------------------------------------
# Tau Positive Prediction Plot
#-------------------------------------------------------------------------------
using CairoMakie, ColorSchemes, Colors

function getdiff(d, n)
    d[:,n] .- d[:,1]
end

function getdiff(d, n, n2)
    d[:,n] .- d[:,n2]
end

function getdiff(d)
    d[:,end] .- d[:,1]
end

begin
    scan = 4
    f = Figure(resolution = (1000, 1000))
    for (i, preds) in enumerate([local_preds, logistic_preds])
    ax = Axis(f[1,i])
        for j in 1:11
            scatter!(four_subdata[j][:, scan] |> vec, Array(preds[j])[:, scan]);
            xlims!(ax, 0.9,2.)
            ylims!(ax, 0.9,2.)
        end
    lines!(0.9:0.1:2.7, 0.9:0.1:2.7, color=:grey)
    ax = Axis(f[2,i])
        for j in 1:11
            _data = getdiff(four_subdata[j], scan)
            _preds = getdiff(preds[j], scan)
            scatter!(_data, _preds);
            xlims!(ax, -0.3,0.75)
            ylims!(ax, -0.3,0.75)
        end
    lines!(-0.1:0.01:1.0, -0.1:0.01:1.0, color=:grey)
    end
    f
end

out_sample_fourth_scans = [sd[:,4] for sd in four_subdata]
mean_fourth_scan = mean(reduce(hcat, out_sample_fourth_scans), dims=2) |> vec
mean_insample_inits = mean(reduce(hcat, [sd[:,1] for sd in four_subdata]), dims=2)

begin
    f = Figure(resolution = (2000, 1000))
    for (i, _preds) in enumerate([local_preds, logistic_preds])
        ax = Axis(f[1,i])

        _out_sample_fourth_preds = [p[:,4] for p in _preds]
        _mean_fourth_preds = mean(reduce(hcat, _out_sample_fourth_preds), dims=2) |> vec

        scatter!(mean_fourth_scan, _mean_fourth_preds)    
        xlims!(ax, 1.0,2.5)
        ylims!(ax, 1.0,2.5)
        lines!(1.0:0.1:2.5, 1.0:0.1:2.5, color=:grey)

        ax = Axis(f[2,i])
        
        _out_sample_inits = [p[:,1] for p in _preds]
        _mean_init_preds = mean(reduce(hcat, _out_sample_inits), dims=2) |> vec

        scatter!(vec(mean_fourth_scan .- mean_insample_inits), 
        vec(_mean_fourth_preds .- _mean_init_preds))   
        xlims!(ax, -0.1,0.5)
        ylims!(ax, -0.1,0.5)
        lines!(-0.1:0.01:0.5, -0.1:0.01:0.5, color=:grey)
    end
    f
end