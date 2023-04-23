using Connectomes
using ADNIDatasets
using CSV, DataFrames
using DrWatson: projectdir
using DifferentialEquations
using Distributions
using Serialization
using LinearAlgebra
using Random
using LinearAlgebra
using Turing
using SparseArrays
include(projectdir("functions.jl"))
include(projectdir("braak-regions.jl"))
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
sub_data_path = projectdir("adni/data/new_data/UCBERKELEYAV1451_8mm_02_17_23_AB_Status.csv")
alldf = CSV.read(sub_data_path, DataFrame)

posdf = filter(x -> x.AB_Status == 1, alldf)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in cortex.ID]

data = ADNIDataset(posdf, dktnames; min_scans=2, max_scans=2)
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

#-------------------------------------------------------------------------------
# Local FKPP
#-------------------------------------------------------------------------------
function NetworkLocalFKPP(du, u, p, t; Lv = Lv, u0 = u0, cc = cc)
    du .= -p[1] * Lv * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

_subdata = [calc_suvr(data, i) for i in tau_pos];
subdata = [normalise(sd, u0, cc) for sd in _subdata];
initial_conditions = [sd[:,1] for sd in subdata];
times =  [get_times(data, i) for i in tau_pos];
second_times = [t[2] for t in times]

local_pst = deserialize(projectdir("adni/chains/local-fkpp/pst-taupos-4x2000.jls"));
#-------------------------------------------------------------------------------
# Global FKPP
#-------------------------------------------------------------------------------
function NetworkGlobalFKPP(du, u, p, t; Lv = Lv)
    du .= -p[1] * Lv * u .+ p[2] .* u .* (1 .- ( u ./ p[3]))
end

max_suvr = maximum(reduce(vcat, reduce(hcat, subdata)))

global_pst = deserialize(projectdir("adni/chains/global-fkpp/pst-taupos-4x2000.jls"));
#-------------------------------------------------------------------------------
# Diffusion
#-------------------------------------------------------------------------------
function NetworkDiffusion(du, u, p, t; Lv = Lv)
    du .= -p[1] * Lv * u
end

diffusion_pst = deserialize(projectdir("adni/chains/diffusion/pst-taupos-4x2000.jls"));
#-------------------------------------------------------------------------------
# Logistic
#-------------------------------------------------------------------------------
function NetworkLogistic(du, u, p, t; Lv = Lv)
    du .= p[1] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

logistic_pst = deserialize(projectdir("adni/chains/logistic/pst-taupos-4x2000.jls"));
#-------------------------------------------------------------------------------
# ELPPD approximation
#-------------------------------------------------------------------------------
function elppd_local_fkpp(pst, initial_conditions, subdata, second_times)
    ps, as, σ = vec(pst[:Pm]), vec(pst[:Am]), vec(pst[:σ])

    slls = Vector{Float64}()
    for (inits, y) in zip(initial_conditions, subdata)
        lls = Vector{Float64}()
        for (p, a, s) in zip(ps, as, σ)
            _prob = ODEProblem(NetworkLocalFKPP, inits, (0.,5.), [p , a])
            _sol = solve(_prob, Tsit5(), saveat=[second_times[1]])
            push!(lls, exp(loglikelihood(MvNormal(vec(_sol), s^2 * I), y[:,2])))
        end
        push!(slls, sum(lls) / 8000)
    end
    sum(log.(slls))
end

local_elppd = elppd_local_fkpp(local_pst, initial_conditions, subdata, second_times)

function elppd_logistic(pst, initial_conditions, subdata, second_times)
    as, σ = vec(pst[:Am]), vec(pst[:σ])

    slls = Vector{Float64}()
    for (inits, y) in zip(initial_conditions, subdata)
        lls = Vector{Float64}()
        for (a, s) in zip(as, σ)
            _prob = ODEProblem(NetworkLogistic, inits, (0.,5.), a)
            _sol = solve(_prob, Tsit5(), saveat=[second_times[1]])
            push!(lls, exp(loglikelihood(MvNormal(vec(_sol), s^2 * I), y[:,2])))
        end
        push!(slls, sum(lls) / 8000)
    end
    sum(log.(slls))
end

logistic_elppd = elppd_logistic(logistic_pst, initial_conditions, subdata, second_times)

function elppd_global_fkpp(pst, initial_conditions, subdata, second_times, max_s)
    ps, as, σ = vec(pst[:Pm]), vec(pst[:Am]), vec(pst[:σ])

    slls = Vector{Float64}()
    for (inits, y) in zip(initial_conditions, subdata)
        lls = Vector{Float64}()
        for (p, a, s) in zip(ps, as, σ)
            _prob = ODEProblem(NetworkGlobalFKPP, inits, (0.,5.), [p , a, max_s])
            _sol = solve(_prob, Tsit5(), saveat=[second_times[1]])
            push!(lls, exp(loglikelihood(MvNormal(vec(_sol), s^2 * I), y[:,2])))
        end
        push!(slls, sum(lls) / 8000)
    end
    sum(log.(slls))
end

global_elppd = elppd_global_fkpp(global_pst, initial_conditions, subdata, second_times, max_suvr)

function elppd_diffusion(pst, initial_conditions, subdata, second_times)
    ps, σ = vec(pst[:Pm]), vec(pst[:σ])

    slls = Vector{Float64}()
    for (inits, y) in zip(initial_conditions, subdata)
        lls = Vector{Float64}()
        for (p, s) in zip(ps, σ)
            _prob = ODEProblem(NetworkDiffusion, inits, (0.,5.), p)
            _sol = solve(_prob, Tsit5(), saveat=[second_times[1]])
            push!(lls, exp(loglikelihood(MvNormal(vec(_sol), s^2 * I), y[:,2])))
        end
        push!(slls, sum(lls) / 8000)
    end
    sum(log.(slls))
end

diffusion_elppd = elppd_diffusion(diffusion_pst, initial_conditions, subdata, second_times)

max_elppd = maximum([local_elppd, global_elppd, logistic_elppd, diffusion_elppd])

elppd_df = DataFrame("Local" => local_elppd - max_elppd, 
                     "Global" => global_elppd - max_elppd, 
                     "Diffusion" => diffusion_elppd - max_elppd,
                     "Logistic" => logistic_elppd - max_elppd)

local_ll = deserialize(projectdir("adni/chains/local-fkpp/ll-taupos-4x2000.jls"));
global_ll = deserialize(projectdir("adni/chains/global-fkpp/ll-taupos-4x2000.jls"));
diffusion_ll = deserialize(projectdir("adni/chains/diffusion/ll-taupos-4x2000.jls"));
logistic_ll = deserialize(projectdir("adni/chains/logistic/ll-taupos-4x2000.jls"));

max_lls= [maximum(dict["data"]) for dict in [local_ll, global_ll, diffusion_ll, logistic_ll]]

ll_df = DataFrame(zip(["Local", "Global", "Diffusion", "Logistic"], max_lls .- maximum(max_lls)))