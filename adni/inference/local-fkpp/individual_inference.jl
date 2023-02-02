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
sub_data_path = projectdir("adni/data/AV1451_Diagnosis-STATUS-STIME-braak-regions.csv")
alldf = CSV.read(sub_data_path, DataFrame)

posdf = filter(x -> x.STATUS == "POS", alldf)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in cortex.ID]

data = ADNIDataset(posdf, dktnames; min_scans=3)

# Ask Jake where we got these cutoffs from? 
mtl_cutoff = 1.375
neo_cutoff = 1.395

mtl_pos = filter(x -> regional_mean(data, mtl, x) >= mtl_cutoff, 1:50)
neo_pos = filter(x -> regional_mean(data, neo, x) >= neo_cutoff, 1:50)

tau_pos = findall(x -> x ∈ unique([mtl_pos; neo_pos]), 1:50)
tau_neg = findall(x -> x ∉ tau_pos, 1:50)

n_pos = length(tau_pos)
n_neg = length(tau_neg)
n_subs = collect(1:length(data))

gmm_moments = CSV.read(projectdir("adni/data/component_moments.csv"), DataFrame)
#gmm_moments2 = CSV.read(projectdir("data/adni-data/component_moments-bothcomps.csv"), DataFrame)
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
# Pos data
#-------------------------------------------------------------------------------
_posdata = [calc_suvr(data, i) for i in tau_pos]
posdata = [normalise(sd, u0, cc) for sd in _posdata]

pos_inits = [sd[:,1] for sd in posdata]
pos_times =  [get_times(data, i) for i in tau_pos]

#-------------------------------------------------------------------------------
# Neg data
#-------------------------------------------------------------------------------
negsuvr = [calc_suvr(data, i) for i in tau_neg]
_negdata = [normalise(sd, u0, cc) for sd in negsuvr]

blsd = [sd .- u0 for sd in _negdata]
nonzerosubs = findall(x -> sum(x) < 2, [sum(sd, dims=1) .== 0 for sd in blsd])

negdata = _negdata[nonzerosubs]

neg_inits = [sd[:,1] for sd in negdata]
_times =  [get_times(data, i) for i in tau_neg]
neg_times = _times[nonzerosubs]

#-------------------------------------------------------------------------------
# data cat
#-------------------------------------------------------------------------------
subdata = [posdata ; negdata]
initial_conditions = [pos_inits ; neg_inits]
times = [pos_times ; neg_times]

#-------------------------------------------------------------------------------
# Model
#-------------------------------------------------------------------------------
function NetworkLocalFKPP(du, u, p, t; Lv = Lv, u0 = u0, cc = cc)
    du .= -p[1] * Lv * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

tvec = reduce(vcat, times)

prob = ODEProblem(NetworkLocalFKPP, 
                  initial_conditions[1], 
                  (0.,maximum(tvec)), 
                  [1.0,1.0])
                  
sol = solve(prob, Tsit5(), saveat=times[1])
#-------------------------------------------------------------------------------
# Inference 
#-------------------------------------------------------------------------------
@model function localfkpp(data, prob, initial_conditions, times)
    σ ~ LogNormal(0.0, 1.0)
    
    ρ ~ LogNormal(0.0,1.0)
    α ~ Normal(0.0, 1.0)

    u ~ arraydist(truncated.(Normal.(initial_conditions, 0.5), u0, cc))

    _prob = remake(prob, u0 = u, p = [ρ, α], saveat=times)

    sol = solve(_prob, 
                Tsit5(), 
                abstol = 1e-9, 
                reltol = 1e-9, 
                sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
                
    if sol.retcode != :Success
        Turing.@addlogprob! -Inf
        return nothing
    end

    data ~ MvNormal(vec(sol), σ^2 * I)
end

setadbackend(:zygote)
Random.seed!(1234)

m = localfkpp(vec(subdata[1]), prob, initial_conditions[1], times[1]);
m();

psts = Vector{Chains}()
for (sd, inits, t) in zip(subdata, initial_conditions, times)
    println("starting inference")
    m = localfkpp(vec(sd), prob, inits, t);
    pst = sample(m, 
                Turing.NUTS(0.8), 
                2_000, 
                progress=false)
    push!(psts, pst)
end
serialize(projectdir("adni/chains/local-fkpp/ind-pst-taupos-2000.jls"), psts)