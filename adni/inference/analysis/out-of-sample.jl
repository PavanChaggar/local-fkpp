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

vols = [get_vol(data, i) for i in tau_pos]
init_vols = [v[:,1] for v in vols]
max_norm_vols = reduce(hcat, [v ./ maximum(v) for v in init_vols])
mean_norm_vols = vec(mean(max_norm_vols, dims=2))
Lv = sparse(inv(diagm(mean_norm_vols)) * L)

function NetworkLocalFKPP(du, u, p, t; Lv = Lv, u0 = u0, cc = cc)
    du .= -p[1] * Lv * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
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
# Model
#-------------------------------------------------------------------------------
@model function localfkpp(prob, initial_conditions, times, n)
    σ ~ LogNormal(0, 1)
    
    Pm ~ LogNormal(0, 1)
    Am ~ Normal(0, 1)
    prob = remake(prob, u0 = initial_conditions, p = [Pm, Am])
    
    sol = solve(prob, Tsit5(), saveat=times)

    data ~ arraydist(Normal.(sol, σ))
    return (; σ, Pm, Am, data)
end

m = localfkpp(prob, initial_conditions[1], times[1], n_pos);
m()

#-------------------------------------------------------------------------------
# Out of sample mean prediction
#-------------------------------------------------------------------------------
pst = deserialize(projectdir("adni/chains/local-fkpp/pst-taupos-4x2000-vc.jls"));

meanpst = mean(pst)

Pm, Am = meanpst[:Pm, :mean], meanpst[:Am, :mean]

mean_probs = [ODEProblem(NetworkLocalFKPP, initial_conditions[i], (0.,5.), [Pm, Am]) for i in 1:n_pos];
mean_sols = [solve(mean_probs[i], Tsit5(), saveat=times[i]) for i in 1:n_pos];

function get_quantiles(q)
    lower = [q[Symbol("data[$i,$j]"), Symbol("2.5%")] for i in 1:72, j in 1:2]
    upper = [q[Symbol("data[$i,$j]"), Symbol("97.5%")] for i in 1:72, j in 1:2]
    mean = [q[Symbol("data[$i,$j]"), Symbol("50.0%")] for i in 1:72, j in 1:2]
    (;lower, upper, mean)
end

function make_predictions(pst, initial_conditions, times)
    m = localfkpp(prob, initial_conditions, times, n_pos);
    _predictions = predict(m, pst[[:σ, :Pm, :Am]])
    qs = quantile(_predictions; q=[0.025, 0.5, 0.975])
    get_quantiles(qs)
end

predictions = [make_predictions(pst, initial_conditions[i], times[i]) for i in 1:n_pos];
mean_predictions = [predictions[i].mean[:,:] for i in 1:n_pos]

#-------------------------------------------------------------------------------
# Out of sample mean predictions -- figures
#-------------------------------------------------------------------------------
using CairoMakie

begin
    f = Figure(resolution=(600, 500))
    ax = Axis(f[1,1], 
            xlabel="SUVR", 
            ylabel="Prediction", 
            titlesize=26, xlabelsize=20, ylabelsize=20)
    xlims!(ax, 0.9, 4.0)
    ylims!(ax, 0.9, 4.0)
    lines!(0.9:0.1:4.0,0.9:0.1:4.0, 
           linestyle=:dash, 
           linewidth=5, 
           color=(:grey, 0.5))
    for i in 1:n_pos
        scatter!(subdata[i][:,2], 
                predictions[i].mean[:,2],
                color=(:grey, 0.2))
    end
    f
end

begin
    f = Figure(resolution=(600, 500))
    ax = Axis(f[1,1], 
            xlabel="SUVR", 
            ylabel="Prediction", 
            titlesize=26, xlabelsize=20, ylabelsize=20)
    xlims!(ax, 0.9, 4.0)
    ylims!(ax, 0.9, 4.0)
    lines!(0.9:0.1:4.0,0.9:0.1:4.0, 
           linestyle=:dash, 
           linewidth=5, 
           color=(:grey, 0.5))
    for i in 1:n_pos
        errorbars!(subdata[i][:,2], 
                predictions[i].mean[:,2], 
                abs.(predictions[i].lower[:,2] .- predictions[i].mean[:,2]),
                predictions[i].upper[:,2] .- predictions[i].mean[:,2],
                direction=:x,
                whiskerwidth = 10,
                color=(:grey, 0.2))
    end
    f
end

second_scan_pred = reduce(hcat, [mean_predictions[i][:,2] for i in 1:n_pos])
mean_second_scan_pred = mean(second_scan_pred, dims=2) |> vec

second_scan_lower = reduce(hcat, [predictions[i].lower[:,2] for i in 1:n_pos])
second_scan_lower_mean = mean(second_scan_lower, dims=2) |> vec

second_scan_upper = reduce(hcat, [predictions[i].upper[:,2] for i in 1:n_pos])
second_scan_upper_mean = mean(second_scan_upper, dims=2) |> vec

second_scan = reduce(hcat, [subdata[i][:,2] for i in 1:n_pos])
mean_second_scan = mean(second_scan, dims=2) |> vec

begin
    f = Figure(resolution=(600, 500))
    ax = Axis(f[1,1], 
            xlabel="SUVR", 
            ylabel="Prediction", 
            titlesize=26, xlabelsize=20, ylabelsize=20)
    xlims!(ax, 0.9, 2.0)
    ylims!(ax, 0.9, 2.0)
    lines!(0.9:0.1:4.0,0.9:0.1:4.0, 
           linestyle=:dash, 
           linewidth=2, 
           color=(:grey, 0.5))
    scatter!(mean_second_scan,mean_second_scan_pred, color=(:blue, 0.5))
    f
end

function get_diff(d)
    d[:,end] .- d[:,1]
end

lobes = unique(c.parc.Lobe)
region_dict = Dict(zip(lobes,collect(1:5)))
region_idx = [region_dict[region] for region in c.parc.Lobe]
cols = [:blue, :green, :black, :red, :brown, :magenta]
col_dict = Dict(zip(collect(1:5), cols))
regions = [findall(x -> x == i, region_idx) for i in 1:5]

_braak_regions = map(getbraak, braak)
braak_regions = [reduce(vcat, [findall(x -> x == roi, c.parc.ID) for roi in braak_region]) for braak_region in _braak_regions]

begin
    f = Figure(resolution=(600, 500))
    ax = Axis(f[1,1], 
            xlabel="SUVR", 
            ylabel="Prediction", 
            titlesize=26, xlabelsize=20, ylabelsize=20)
    xlims!(ax, -0.25, 0.5)
    ylims!(ax, -0.25, 0.5)
    for i in 1:n_pos
        scatter!(get_diff(subdata[i]), 
                 get_diff(mean_predictions[i]), color=((:grey, 0.5), 0.1))
    end
    f
end

begin
    f = Figure(resolution=(2000, 500))
    for k in 1:5
        ax = Axis(f[1,k], 
                xlabel="SUVR", 
                ylabel="Prediction",
                title="$(lobes[k])",
                titlesize=26, xlabelsize=20, ylabelsize=20)
        xlims!(ax, -0.25, 0.5)
        ylims!(ax, -0.25, 0.5)
        lines!(-0.25:0.1:0.5, -0.25:0.1:0.5, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
        for i in 1:n_pos
            for j in regions[k]
            scatter!(get_diff(subdata[i])[j], 
                    get_diff(mean_predictions[i])[j], color=(cols[k], 0.2))
            end
        end
    end 
    f
end

mean_diff = mean(reduce(hcat, [get_diff(subdata[i]) for i in 1:n_pos]), dims=2) |> vec
mean_pred_diff = mean(reduce(hcat, [get_diff(mean_predictions[i]) for i in 1:n_pos]), dims=2) |> vec

begin
    f = Figure(resolution=(600, 500))
    ax = Axis(f[1,1], 
            xlabel="δ SUVR", 
            ylabel="δ Prediction", 
            titlesize=26, xlabelsize=20, ylabelsize=20)
    xlims!(ax, -0.05, 0.15)
    ylims!(ax, -0.05, 0.15)
    lines!(-0.05:0.05:0.15, -0.05:0.05:0.15, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
    for i in 1:72
        scatter!(mean_diff[i], mean_pred_diff[i], color=(:blue, 0.5))
    end
    f
end

begin
    f = Figure(resolution=(2500, 500))
    for (j, roi) in enumerate(braak_regions)
        ax = Axis(f[1,j],
                xlabel="δ SUVR", 
                ylabel="δ Prediction",
                title="Braak $(j)",
                titlesize=26, xlabelsize=20, ylabelsize=20)
        xlims!(ax, -0.05, 0.15)
        ylims!(ax, -0.05, 0.15)
        lines!(-0.05:0.05:0.15, -0.05:0.05:0.15, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
        for i in roi
            scatter!(mean_diff[i], 
                    mean_pred_diff[i], color=(cols[j], 1.0))
        end
    end
    f
end

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