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
using MCMCChains
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
sub_data_path = projectdir("adni/data/AV1451_Diagnosis-STATUS-STIME-braak-regions.csv");
alldf = CSV.read(sub_data_path, DataFrame)

posdf = filter(x -> x.STATUS == "POS", alldf)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in cortex.ID]

data = ADNIDataset(posdf, dktnames; min_scans=2, max_scans=2)

# Ask Jake where we got these cutoffs from? 
mtl_cutoff = 1.375
neo_cutoff = 1.395

function regional_mean(data, rois, sub)
    subsuvr = calc_suvr(data, sub)
    mean(subsuvr[rois,1])
end

mtl_pos = filter(x -> regional_mean(data, mtl, x) >= mtl_cutoff, 1:50)
neo_pos = filter(x -> regional_mean(data, neo, x) >= neo_cutoff, 1:50)

tau_pos = findall(x -> x ∈ unique([mtl_pos; neo_pos]), 1:50)
tau_neg = findall(x -> x ∉ tau_pos, 1:50)

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
L = laplacian_matrix(c)

function NetworkLocalFKPP(du, u, p, t; L = L, u0 = u0, cc = cc)
    du .= -p[1] * L * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

subdata = [calc_suvr(data, i) for i in tau_pos]
for i in 1:n_pos
    normalise!(subdata[i], u0, cc)
end

initial_conditions = [sd[:,1] for sd in subdata]
times =  [get_times(data, i) for i in tau_pos]

prob = ODEProblem(NetworkLocalFKPP, 
                  initial_conditions[1], 
                  (0.,10.), 
                  [1.0,1.0])
                  
sol = solve(prob, Tsit5())
#-------------------------------------------------------------------------------
# Out of sample mean prediction
#-------------------------------------------------------------------------------
pst = deserialize(projectdir("adni/chains/local-fkpp/pst-taupos-4x2000.jls"));

meanpst = mean(pst)

Pm, Am = meanpst[:Pm, :mean], meanpst[:Am, :mean]

probs = [ODEProblem(NetworkLocalFKPP, initial_conditions[i], (0.,5.), [Pm, Am]) for i in 1:n_pos];
sols = [solve(probs[i], Tsit5(), saveat=times[i]) for i in 1:n_pos];

using CairoMakie

f = Figure(resolution=(600, 500))
ax = Axis(f[1,1], 
          xlabel="SUVR", 
          ylabel="Prediction", 
          titlesize=26, xlabelsize=20, ylabelsize=20)
xlims!(ax, 0.9, 2.5)
ylims!(ax, 0.9, 2.5)
for i in 1:n_pos
    scatter!(subdata[i][:,2], sols[i][2])
end
f

function get_diff(d)
    d[:,end] .- d[:,1]
end

f = Figure(resolution=(600, 500))
ax = Axis(f[1,1], 
          xlabel="SUVR", 
          ylabel="Prediction", 
          titlesize=26, xlabelsize=20, ylabelsize=20)
xlims!(ax, -0.25, 0.5)
ylims!(ax, -0.25, 0.5)
for i in 1:n_pos
    scatter!(get_diff(subdata[i]), get_diff(sols[i]))
end
f

function get_diff(d, roi)
    d[roi,end] .- d[roi, 1]
end

residuals = reduce(hcat, [get_diff(subdata[i]) .- get_diff(sols[i]) for i in 1:n_pos])
sqrs = residuals .^ 2

sum(sqrs, dims=1)
sum(sqrs, dims=2)

node = 65
f = Figure(resolution=(600, 500))
ax = Axis(f[1,1], 
          xlabel="SUVR", 
          ylabel="Prediction", 
          titlesize=26, xlabelsize=20, ylabelsize=20)
xlims!(ax, -0.3, 0.3)
ylims!(ax, -0.3, 0.3)
for i in 1:n_pos
    scatter!(get_diff(subdata[i], node), get_diff(sols[i], node))
end
f

#-------------------------------------------------------------------------------
# Out of sample trajectories
#-------------------------------------------------------------------------------
function make_prob_func(initial_conditions, p, a, times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions, p=[p[i], a[i]], saveat=0.1)
    end
end

function output_func(sol,i)
    (sol,false)
end

vecP, vecA, vecσ = vec(pst[:Pm]), vec(pst[:Am]), vec(pst[:σ])

node=29
for sub in 1:1
ensemble_prob = EnsembleProblem(prob, 
                                prob_func=make_prob_func(initial_conditions[sub], 
                                                         vecP, vecA, 
                                                         times[sub]), 
                                output_func=output_func)

ensemble_sol = solve(ensemble_prob, Tsit5(), trajectories=8000)

es = ensemble_sol[node,:,:]

f = Figure(resolution=(750, 500))
ax = Axis(f[1,1])
for i in rand(1:8000, 200)
    noise = rand(Normal(0, vecσ[i]), length(ensemble_sol[i].t))
    lines!(ensemble_sol[i].t, ensemble_sol[i][29,:] .+ noise, color=(:grey, 0.1))
    scatter!(times[sub], subdata[sub][29,:])
end
display(f)
end