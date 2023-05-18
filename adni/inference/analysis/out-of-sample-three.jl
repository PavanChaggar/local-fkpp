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
using CairoMakie
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
_subdata = [calc_suvr(data, i) for i in tau_pos]
[normalise!(_subdata[i], u0, cc) for i in 1:n_pos]

outsample_idx = findall(x -> size(x, 2) > 3, _subdata)

four_subdata = _subdata[outsample_idx]

insample_subdata = [sd[:, 1:3] for sd in _subdata]
insample_four_subdata = insample_subdata[outsample_idx]
insample_inits = [d[:,1] for d in insample_four_subdata]

outsample_subdata = [sd[:, 4:end] for sd in _subdata[outsample_idx]]

max_suvr = maximum(reduce(vcat, reduce(hcat, insample_subdata)))

_times =  [get_times(data, i) for i in tau_pos]
times = _times[outsample_idx]
insample_times = [t[1:3] for t in _times]

outsample_times = [t[4:end] for t in _times[outsample_idx]]
#-------------------------------------------------------------------------------
# Models
#-------------------------------------------------------------------------------
L = laplacian_matrix(c)

vols = [get_vol(data, i) for i in tau_pos]
init_vols = [v[:,1] for v in vols]
max_norm_vols = reduce(hcat, [v ./ maximum(v) for v in init_vols])
mean_norm_vols = vec(mean(max_norm_vols, dims=2))
Lv = sparse(inv(diagm(mean_norm_vols)) * L)

function NetworkLocalFKPP(du, u, p, t; Lv = Lv, u0 = u0, cc = cc)
    du .= -p[1] * Lv * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

function NetworkGlobalFKPP(du, u, p, t; Lv = Lv)
    du .= -p[1] * Lv * u .+ p[2] .* u .* (1 .- ( u ./ p[3]))
end

function NetworkDiffusion(du, u, p, t; Lv = Lv)
    du .= -p[1] * Lv * u
end

function NetworkLogistic(du, u, p, t; Lv = Lv)
    du .= p[1] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

#----------------------------`---------------------------------------------------
# Posteriors
#-------------------------------------------------------------------------------
local_pst = deserialize(projectdir("adni/chains/local-fkpp/pst-taupos-1x2000-three.jls"));
# global_pst = deserialize(projectdir("adni/chains/global-fkpp/pst-taupos-1x2000-three-indp0.jls"));
# diffusion_pst = deserialize(projectdir("adni/chains/diffusion/pst-taupos-1x2000-three-indp0.jls"));
logistic_pst = deserialize(projectdir("adni/chains/logistic/pst-taupos-1x2000-three.jls"));

[sum(p[:numerical_error]) for p in [local_pst, logistic_pst]]
#-------------------------------------------------------------------------------
# Local model
#-------------------------------------------------------------------------------
local_meanpst = mean(local_pst);

foursubs_vec_idx = reshape(1:2232, 72, 31)[:, outsample_idx];

local_all_inits_vec = [local_pst["u[$i]"] for i in vec(foursubs_vec_idx)]
local_all_inits = reshape(local_all_inits_vec, 72, 10)
local_inits = [transpose(reduce(hcat, local_all_inits[:, i])) for i in 1:10]

local_mean_inits = vec.(mean.(local_inits, dims=2))

local_ps = [Array(local_pst[Symbol("ρ[$i]")]) for i in outsample_idx];
local_as = [Array(local_pst[Symbol("α[$i]")]) for i in outsample_idx];
local_ss = Array(local_pst[Symbol("σ")]);
local_params = [[local_meanpst[Symbol("ρ[$i]"), :mean], local_meanpst[Symbol("α[$i]"), :mean]] for i in outsample_idx];

function simulate(f, initial_conditions, params, times)
    max_t = maximum(reduce(vcat, times))
    [solve(
        ODEProblem(
            f, inits, (0, max_t), p
        ), 
        Tsit5(), abstol=1e-9, reltol=1e-9, saveat=t
    )
    for (inits, p, t) in zip(initial_conditions, params, times)
    ]
end

local_preds = simulate(NetworkLocalFKPP, insample_inits, local_params, times);

local_out_preds = [pred[:, 4:end] for pred in local_preds]
out_data = [sd[:, 4:end] for sd in four_subdata]

begin
    f = Figure()
    ax = Axis(f[1,1])
    for i in 1:10
        scatter!(out_data[i] |> vec, out_preds[i] |> vec, color=(:blue, 0.5));
        xlims!(ax, 1.0,2.7)
        ylims!(ax, 1.0,2.7)
    end
    lines!(1.0:0.1:2.5, 1.0:0.1:2.5, color=:grey)
    f
end

sum([loglikelihood(
                MvNormal(vec(local_out_preds[i]), local_meanpst[:σ, :mean]^2 * I), 
                vec(out_data[i])) for i in 1:10])
              
sum((vec(reduce(hcat, local_out_preds)) .- vec(reduce(hcat, out_data))).^2)

function elppd_local_local(pst, ps, as, initial_conditions, subdata, out_times)
    σ = vec(pst[:σ])

    slls = Vector{Float64}()
    for (_p, _a, inits, y, t) in zip(ps, as, initial_conditions, subdata, out_times)
        lls = Vector{Float64}()
        for (i, (p, a, s)) in enumerate(zip(_p, _a, σ))
            _prob = ODEProblem(NetworkLocalFKPP, inits, (0.,7.), [p , a])
            _sol = solve(_prob, Tsit5(), saveat=t[4:end])
            push!(lls, exp(loglikelihood(MvNormal(vec(_sol), s^2 * I), vec(y))))
        end
        push!(slls, sum(lls) / 2000)
    end
    sum(log.(slls))
end
elppd_local_local(local_pst, local_ps, local_as, insample_inits, out_data, times)

#-------------------------------------------------------------------------------
# Posteriors and predictions logistic
#-------------------------------------------------------------------------------
logistic_meanpst = mean(logistic_pst);

logistic_all_inits_vec = [logistic_pst["u[$i]"] for i in vec(foursubs_vec_idx)]
logistic_all_inits = reshape(logistic_all_inits_vec, 72, 10)
logistic_inits = [transpose(reduce(hcat, logistic_all_inits[:, i])) for i in 1:10]

logistic_mean_inits = vec.(mean.(logistic_inits, dims=2))

logistic_as = [Array(logistic_pst[Symbol("α[$i]")]) for i in outsample_idx];

logistic_params = [[logistic_meanpst[Symbol("α[$i]"), :mean]] for i in outsample_idx];

logistic_preds = simulate(NetworkLogistic, insample_inits, logistic_params, times);

logistic_out_preds = [pred[:, 4:end] for pred in logistic_preds]

sum([loglikelihood(
                MvNormal(vec(logistic_out_preds[i]), logistic_meanpst[:σ, :mean]^2 * I), 
                vec(out_data[i])) for i in 1:10])
              
sum((vec(reduce(hcat, logistic_out_preds)) .- vec(reduce(hcat, out_data))).^2)

begin
    f = Figure()
    ax = Axis(f[1,1])
    for i in 1:10
        scatter!(out_data[i] |> vec, logistic_out_preds[i] |> vec, color=(:blue, 0.5));
        xlims!(ax, 1.0,2.7)
        ylims!(ax, 1.0,2.7)
    end
    lines!(1.0:0.1:2.5, 1.0:0.1:2.5, color=:grey)
    f
end

function elppd_logistic(pst, as, initial_conditions, subdata, out_times)
    σ = vec(pst[:σ])

    slls = Vector{Float64}()
    for (_a, inits, y, t) in zip(as, initial_conditions, subdata, out_times)
        lls = Vector{Float64}()
        for (i, (a, s)) in enumerate(zip(_a, σ))
            _prob = ODEProblem(NetworkLogistic, inits, (0.,7.), [a])
            _sol = solve(_prob, Tsit5(), saveat=t[4:end])
            push!(lls, exp(loglikelihood(MvNormal(vec(_sol), s^2 * I), vec(y))))
        end
        push!(slls, sum(lls) / 2000)
    end
    sum(log.(slls))
end

elppd_logistic(logistic_pst, logistic_as, insample_inits, out_data, times)