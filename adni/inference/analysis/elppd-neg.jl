using Connectomes
using ADNIDatasets
using CSV, DataFrames
using DrWatson: projectdir
using DifferentialEquations
using Distributions
using Serialization, MCMCChains
using DelimitedFiles, LinearAlgebra
using Random
using LinearAlgebra, SparseArrays
include(projectdir("functions.jl"))
#-------------------------------------------------------------------------------
# Connectome and ROIs
#-------------------------------------------------------------------------------
connectome_path = Connectomes.connectome_path()
all_c = filter(Connectome(connectome_path; norm=true, weight_function = (n, l) -> n ./ l), 1e-2);

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
sub_data_path = projectdir("adni/data/new_data/UCBERKELEYAV1451_8mm_02_17_23_AB_Status.csv")
alldf = CSV.read(sub_data_path, DataFrame)

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
#gmm_moments2 = CSV.read(projectdir("data/adni-data/component_moments-bothcomps.csv"), DataFrame)
ubase, upath = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(ubase)
cc = quantile.(upath, .99)

#-------------------------------------------------------------------------------
# neg data 
#-------------------------------------------------------------------------------
subsuvr = [calc_suvr(data, i) for i in tau_neg]
_subdata = [normalise(sd, u0, cc) for sd in subsuvr]

blsd = [sd .- u0 for sd in _subdata]
nonzerosubs = findall(x -> sum(x) < 2, [sum(sd, dims=1) .== 0 for sd in blsd])

subdata = _subdata[nonzerosubs]

outsample_idx = findall(x -> size(x, 2) > 3, subdata)

four_subdata = subdata[outsample_idx]

insample_subdata = [sd[:, 1:3] for sd in subdata]
insample_four_subdata = insample_subdata[outsample_idx]
insample_inits = [d[:,1] for d in insample_four_subdata]

outsample_subdata = [sd[:, 4:end] for sd in four_subdata]

max_suvr = maximum(reduce(vcat, reduce(hcat, insample_subdata)))

_times =  [get_times(data, i) for i in tau_neg]
nonzero_times = _times[nonzerosubs]
times = nonzero_times[outsample_idx]
insample_times = [t[1:3] for t in _times]

outsample_times = [t[4:end] for t in times]
#-------------------------------------------------------------------------------
# Connectome + ODEE
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

function NetworkGlobalFKPP(du, u, p, t; Lv = Lv)
    du .= -p[1] * Lv * u .+ p[2] .* u .* (1 .- ( u ./ p[3]))
end

function NetworkDiffusion(du, u, p, t; Lv = Lv)
    du .= -p[1] * Lv * u
end

function NetworkLogistic(du, u, p, t; Lv = Lv)
    du .= p[1] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

#-------------------------------------------------------------------------------
# Local FKPP
#-------------------------------------------------------------------------------
local_pst = deserialize(projectdir("adni/chains/local-fkpp/ballistic/pst-tauneg-1x2000-three.jls"));
local_ps = [Array(local_pst[Symbol("ρ[$i]")]) for i in outsample_idx];
local_as = [Array(local_pst[Symbol("α[$i]")]) for i in outsample_idx];

function elppd_local(pst, ps, as, initial_conditions, subdata, out_times)
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

local_elppd = elppd_local(local_pst, local_ps, local_as, insample_inits, outsample_subdata, times)

#-------------------------------------------------------------------------------
# Global FKPP
#-------------------------------------------------------------------------------
global_pst = deserialize(projectdir("adni/chains/global-fkpp/ballistic/pst-tauneg-1x2000-three.jls"));
global_ps = [Array(global_pst[Symbol("ρ[$i]")]) for i in outsample_idx];
global_as = [Array(global_pst[Symbol("α[$i]")]) for i in outsample_idx];

function elppd_global(pst, ps, as, max_suvr, initial_conditions, subdata, out_times)
    σ = vec(pst[:σ])

    slls = Vector{Float64}()
    for (_p, _a, inits, y, t) in zip(ps, as, initial_conditions, subdata, out_times)
        lls = Vector{Float64}()
        for (i, (p, a, s)) in enumerate(zip(_p, _a, σ))
            _prob = ODEProblem(NetworkGlobalFKPP, inits, (0.,7.), [p , a, max_suvr])
            _sol = solve(_prob, Tsit5(), saveat=t[4:end])
            push!(lls, exp(loglikelihood(MvNormal(vec(_sol), s^2 * I), vec(y))))
        end
        push!(slls, sum(lls) / 2000)
    end
    sum(log.(slls))
end

global_elppd = elppd_global(global_pst, global_ps, global_as, max_suvr, insample_inits, outsample_subdata, times)

#-------------------------------------------------------------------------------
# Diffusion
#-------------------------------------------------------------------------------
diffusion_pst = deserialize(projectdir("adni/chains/diffusion/ballistic/pst-tauneg-1x2000-three.jls"));
diffusion_ps = [Array(diffusion_pst[Symbol("ρ[$i]")]) for i in outsample_idx];

function elppd_diffusion(pst, ps, initial_conditions, subdata, out_times)
    σ = vec(pst[:σ])

    slls = Vector{Float64}()
    for (_p, inits, y, t) in zip(ps, initial_conditions, subdata, out_times)
        lls = Vector{Float64}()
        for (i, (p, s)) in enumerate(zip(_p, σ))
            _prob = ODEProblem(NetworkDiffusion, inits, (0.,7.), [p])
            _sol = solve(_prob, Tsit5(), saveat=t[4:end])
            push!(lls, exp(loglikelihood(MvNormal(vec(_sol), s^2 * I), vec(y))))
        end
        push!(slls, sum(lls) / 2000)
    end
    sum(log.(slls))
end

diffusion_elppd = elppd_diffusion(diffusion_pst, diffusion_ps, insample_inits, outsample_subdata, times)
#-------------------------------------------------------------------------------
# Logistic
#-------------------------------------------------------------------------------
logistic_pst = deserialize(projectdir("adni/chains/logistic/pst-taupos-1x2000-three.jls"));
logistic_as = [Array(logistic_pst[Symbol("α[$i]")]) for i in outsample_idx];

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

logistic_elppd = elppd_logistic(logistic_pst, logistic_as, insample_inits, outsample_subdata, times)

max_elppd = maximum([local_elppd, global_elppd, logistic_elppd, diffusion_elppd])

elppd_df = DataFrame("Local" => local_elppd - max_elppd, 
                     "Global" => global_elppd - max_elppd, 
                     "Diffusion" => diffusion_elppd - max_elppd,
                     "Logistic" => logistic_elppd - max_elppd)

local_ll = deserialize(projectdir("adni/chains/local-fkpp/length-free/ll-tauneg-4x2000.jls"));
global_ll = deserialize(projectdir("adni/chains/global-fkpp/length-free/ll-tauneg-4x2000.jls"));
diffusion_ll = deserialize(projectdir("adni/chains/diffusion/length-free/ll-tauneg-4x2000.jls"));
logistic_ll = deserialize(projectdir("adni/chains/logistic/ll-tauneg-4x2000.jls"));

max_lls= [maximum(dict["data"]) for dict in [local_ll, global_ll, diffusion_ll, logistic_ll]]

ll_df = DataFrame(zip(["Local", "Global", "Diffusion", "Logistic"], max_lls .- maximum(max_lls)))