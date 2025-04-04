using Connectomes
using ADNIDatasets
using CSV, DataFrames
using DrWatson: projectdir
using DifferentialEquations
using Distributions
using Serialization, MCMCChains
using DelimitedFiles, LinearAlgebra
using Random
using Turing
using LinearAlgebra, SparseArrays
include(projectdir("functions.jl"))
#-------------------------------------------------------------------------------
# Load connectome, regional parameters and sort data
#-------------------------------------------------------------------------------
include(projectdir("adni/inference/inference-preamble.jl"))
#-------------------------------------------------------------------------------
# Pos data 
#-------------------------------------------------------------------------------
_subdata = calc_suvr.(pos_data)
[normalise!(_subdata[i], u0, cc) for i in 1:n_pos]

outsample_idx = findall(x -> size(x, 2) > 3, _subdata)

four_subdata = _subdata[outsample_idx]

insample_subdata = [sd[:, 1:3] for sd in _subdata]
insample_four_subdata = insample_subdata[outsample_idx]
insample_inits = [d[:,1] for d in insample_four_subdata]

outsample_subdata = [sd[:, 4:end] for sd in four_subdata]

max_suvr = maximum(reduce(vcat, reduce(hcat, insample_subdata)))

_times =  get_times.(pos_data)
times = _times[outsample_idx]
insample_times = [t[1:3] for t in _times]

outsample_times = [t[4:end] for t in _times[outsample_idx]]
#-------------------------------------------------------------------------------
# Models
#-------------------------------------------------------------------------------
L = laplacian_matrix(c)

vols = get_vol.(pos_data)
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
# Connectome + ODEE
#-------------------------------------------------------------------------------
local_pst = deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-taupos-1x2000-three.jls"));
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
            push!(lls, pdf(MvNormal(vec(_sol), s^2 * I), vec(y)))
        end
        push!(slls, sum(lls) / 2000)
    end
    sum(log.(slls))
end

local_elppd = elppd_local(local_pst, local_ps, local_as, insample_inits, outsample_subdata, times)

#-------------------------------------------------------------------------------
# Global FKPP
#-------------------------------------------------------------------------------
global_pst = deserialize(projectdir("adni/new-chains/old/global-fkpp/length-free/pst-taupos-1x2000-three.jls"));
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
            push!(lls, pdf(MvNormal(vec(_sol), s^2 * I), vec(y)))
        end
        push!(slls, sum(lls) / 2000)
    end
    sum(log.(slls))
end

global_elppd = elppd_global(global_pst, global_ps, global_as, max_suvr, insample_inits, outsample_subdata, times)

#-------------------------------------------------------------------------------
# Diffusion
#-------------------------------------------------------------------------------
diffusion_pst = deserialize(projectdir("adni/new-chains/old/diffusion/length-free/pst-taupos-1x2000-three.jls"));
diffusion_ps = [Array(diffusion_pst[Symbol("ρ[$i]")]) for i in outsample_idx];

function elppd_diffusion(pst, ps, initial_conditions, subdata, out_times)
    σ = vec(pst[:σ])

    slls = Vector{Float64}()
    for (_p, inits, y, t) in zip(ps, initial_conditions, subdata, out_times)
        lls = Vector{Float64}()
        for (i, (p, s)) in enumerate(zip(_p, σ))
            _prob = ODEProblem(NetworkDiffusion, inits, (0.,7.), [p])
            _sol = solve(_prob, Tsit5(), saveat=t[4:end])
            push!(lls, pdf(MvNormal(vec(_sol), s^2 * I), vec(y)))
        end
        push!(slls, sum(lls) / 2000)
    end
    sum(log.(slls))
end

diffusion_elppd = elppd_diffusion(diffusion_pst, diffusion_ps, insample_inits, outsample_subdata, times)
#-------------------------------------------------------------------------------
# Logistic
#-------------------------------------------------------------------------------
logistic_pst = deserialize(projectdir("adni/new-chains/logistic/pst-taupos-1x2000-three.jls"));
logistic_as = [Array(logistic_pst[Symbol("α[$i]")]) for i in outsample_idx];

function elppd_logistic(pst, as, initial_conditions, subdata, out_times)
    σ = vec(pst[:σ])

    slls = Vector{Float64}()
    for (_a, inits, y, t) in zip(as, initial_conditions, subdata, out_times)
        lls = Vector{Float64}()
        for (i, (a, s)) in enumerate(zip(_a, σ))
            _prob = ODEProblem(NetworkLogistic, inits, (0.,7.), [a])
            _sol = solve(_prob, Tsit5(), saveat=t[4:end])
            push!(lls, pdf(MvNormal(vec(_sol), s^2 * I), vec(y)))
        end
        push!(slls, sum(lls) / 2000)
    end
    sum(log.(slls))
end

logistic_elppd = elppd_logistic(logistic_pst, logistic_as, insample_inits, outsample_subdata, times)
#-------------------------------------------------------------------------------
# ELPPD approximation
#-------------------------------------------------------------------------------
elppd_df = DataFrame("Local" => local_elppd,
                     "Global" => global_elppd,
                     "Diffusion" => diffusion_elppd,
                     "Logistic" => logistic_elppd)

local_ll = deserialize(projectdir("adni/new-chains/local-fkpp/length-free/ll-taupos-4x2000.jls"));
global_ll = deserialize(projectdir("adni/new-chains/global-fkpp/length-free/ll-taupos-4x2000.jls"));
diffusion_ll = deserialize(projectdir("adni/new-chains/diffusion/length-free/ll-taupos-4x2000.jls"));
logistic_ll = deserialize(projectdir("adni/new-chains/logistic/ll-taupos-4x2000.jls"));

nparams = [length(names(MCMCChains.get_sections(pst, :parameters))) for pst in [local_pst, global_pst, diffusion_pst, logistic_pst]]

max_lls= [maximum(dict["data"]) for dict in [local_ll, global_ll, diffusion_ll, logistic_ll]]
bic = [(n * log(13536) - (2 * maximum(dict["data"]))) for (dict, n) in zip([local_ll, global_ll, diffusion_ll, logistic_ll], nparams)]

bic_df = DataFrame(zip(["Local", "Global", "Diffusion", "Logistic"], bic))