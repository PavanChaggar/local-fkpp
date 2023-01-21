using Connectomes
using ADNIDatasets
using CSV, DataFrames
using DrWatson: projectdir
using DifferentialEquations
using DiffEqSensitivity
using Turing
using Distributions
using Serialization
using DelimitedFiles, LinearAlgebra
using Random
using LinearAlgebra
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

neo_only = findall(x -> x ∈ setdiff(neo_pos, mtl_pos), tau_pos)
mtl_only = findall(x -> x ∈ setdiff(tau_pos, setdiff(neo_pos, mtl_pos)), tau_pos)

gmm_moments = CSV.read(projectdir("adni/data/component_moments.csv"), DataFrame)
ubase, upath = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(ubase)
cc = quantile.(upath, .99);

#-------------------------------------------------------------------------------
# Connectome + ODEE
#-------------------------------------------------------------------------------
L = laplacian_matrix(c);

function NetworkLocalFKPP(du, u, p, t; L = L, u0 = u0, cc = cc)
    du .= -p[1] * L * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

function getdiff(d, n)
    d[:,n] .- d[:,1]
end

function getdiff(d)
    d[:,end] .- d[:,1]
end

function getdiffroi(d, roi)
    d[roi,end] .- d[roi, 1]
end

_subdata = [calc_suvr(data, i) for i in tau_pos];
subdata = [normalise(sd, u0, cc) for sd in _subdata];
initial_conditions = [sd[:,1] for sd in subdata];
times =  [get_times(data, i) for i in tau_pos];

pst = deserialize(projectdir("adni/chains/local-fkpp/pst-taupos-4x2000.jls"));

meanpst = mean(pst);
params = [[meanpst[Symbol("ρ[$i]"), :mean], meanpst[Symbol("α[$i]"), :mean]] for i in 1:27];
sols = [solve(ODEProblem(NetworkLocalFKPP, init, (0.0,5.0), p), Tsit5(), saveat=t) for (init, t, p) in zip(initial_conditions, times, params)];

using CairoMakie; CairoMakie.activate!()

begin
    f = Figure(resolution=(1500, 1000))
    gl = [f[1, i] = GridLayout() for i in 1:3]
    for i in 1:3
        scan = i + 1
        ax = Axis(gl[i][1,1], 
                xlabel="SUVR", 
                ylabel="Prediction", 
                title="Scan: $scan", 
                titlesize=26, xlabelsize=20, ylabelsize=20)
        if i > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        xlims!(ax, 0.8, 4.0)
        ylims!(ax, 0.8, 4.0)
        lines!(0.8:0.1:4.0, 0.8:0.1:4.0, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
        for i in 1:27
            if size(subdata[i], 2) >= scan
                scatter!(subdata[i][:,scan], sols[i][:,scan], marker='o', markersize=15)
            end
        end
    end

    gl = [f[2, i] = GridLayout() for i in 1:3]
    for i in 1:3
        scan = i + 1

        if scan < 4
            diffs = getdiff.(subdata, scan)
            soldiff = getdiff.(sols, scan)
        else
            idx = findall(x -> size(x,2) == scan, subdata)
            diffs = getdiff.(subdata[idx], scan)
            soldiff = getdiff.(sols[idx], scan)
        end

        ax = Axis(gl[i][1,1], 
                xlabel="Δ SUVR",
                ylabel="Δ Prediction",
                titlesize=26, xlabelsize=20, ylabelsize=20, 
                xticks=collect(-1:0.5:1))
        if i > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        start = -1.0
        stop = 1.0
        xlims!(ax, start, stop)
        ylims!(ax, start, stop)
        lines!(start:0.1:stop, start:0.1:stop, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
        for i in eachindex(diffs)
            scatter!(diffs[i], soldiff[i], marker='o', markersize=15)
        end
    end

    f
end
save(projectdir("adni/visualisation/hier-inf/pred-taupos.pdf"), f)
save(projectdir("adni/visualisation/hier-inf/png/pred-taupos.png"), f)

begin
    f = Figure(resolution=(1500, 500))
    gl = [f[1, i] = GridLayout() for i in 1:3]
    for i in 1:3
        scan = i + 1

        if scan < 4
            diffs = getdiff.(subdata, scan)
            soldiff = getdiff.(sols, scan)
        else
            idx = findall(x -> size(x,2) == scan, subdata)
            diffs = getdiff.(subdata[idx], scan)
            soldiff = getdiff.(sols[idx], scan)
        end

        ax = Axis(gl[i][1,1], 
                xlabel="SUVR", 
                ylabel="Prediction",  
                title="Scan: $scan", 
                titlesize=26, xlabelsize=20, ylabelsize=20, 
                xticks=collect(-1:0.5:1))
        if i > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        start = -1.0
        stop = 1.0
        xlims!(ax, start, stop)
        ylims!(ax, start, stop)
        lines!(start:0.1:stop, start:0.1:stop, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
        for i in eachindex(diffs)
            scatter!(diffs[i], soldiff[i], marker='o', markersize=15)
        end
    end
    f
end
save(projectdir("adni/visualisation/hier-inf/pred-delta-taupos.pdf"), f)

# begin
#     f = Figure()
#     diffs = getdiff.(totaltau)
#     soldiff = getdiff.(totalmeansols)
#     ax = Axis(f[1,1], 
#             xlabel="Δ Total SUVR", 
#             ylabel="Δ Total Prediction", 
#             title="Difference between last scan and first scan")

#     ylims!(ax, -1.5, 10)
#     xlims!(ax, -1.5, 10)
#     lines!(-1.5:1:10, -1.5:1:10, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
#     for i in 1:27
#         scatter!(diffs[i], soldiff[i], markersize=15)
#     end
#     f
# end
# save(projectdir("adni/visualisation/hier-inf/pred-delta-totaltaupos.png"), f)


subsuvr = [calc_suvr(data, i) for i in tau_neg]
_subdata = [normalise(sd, u0) for sd in subsuvr]

blsd = [sd .- u0 for sd in _subdata]
nonzerosubs = findall(x -> sum(x) < 2, [sum(sd, dims=1) .== 0 for sd in blsd])

subdata = _subdata[nonzerosubs]

totaltau = sum.(subdata,  dims=1)

initial_conditions = [sd[:,1] for sd in subdata]
_times =  [get_times(data, i) for i in tau_neg]
times = _times[nonzerosubs]

pst2 = deserialize(projectdir("adni/hierarchical-inference/local-fkpp/chains/hier-local-pst-tauneg-uniform-4x2000.jls"));

meanpst = mean(pst2)
params = [[meanpst[Symbol("ρ[$i]"), :mean], meanpst[Symbol("α[$i]"), :mean]] for i in 1:21]
sols = [solve(ODEProblem(NetworkExFKPP, init, (0.0,10.0), p), Tsit5(), saveat=t) for (init, t, p) in zip(initial_conditions, times, params)];
totalsols = sum.(sols, dims=1)

begin
    f = Figure(resolution=(1500, 1000))
    gl = [f[1, i] = GridLayout() for i in 1:3]
    for i in 1:3
        scan = i + 1
        ax = Axis(gl[i][1,1], 
                xlabel="SUVR", 
                ylabel="Prediction", 
                title="Scan: $scan", 
                titlesize=26, xlabelsize=20, ylabelsize=20)
                if i > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        xlims!(ax, 0.8, 4.0)
        ylims!(ax, 0.8, 4.0)
        lines!(0.8:0.1:4.0, 0.8:0.1:4.0, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
        for i in 1:21
            if size(subdata[i], 2) >= scan
                scatter!(subdata[i][:,scan], sols[i][:,scan], marker='o', markersize=15)
            end
        end
    end
    
    gl = [f[2, i] = GridLayout() for i in 1:3]
    for i in 1:3
        scan = i + 1

        if scan < 4
            diffs = getdiff.(subdata, scan)
            soldiff = getdiff.(sols, scan)
        else
            idx = findall(x -> size(x,2) == scan, subdata)
            diffs = getdiff.(subdata[idx], scan)
            soldiff = getdiff.(sols[idx], scan)
        end

        ax = Axis(gl[i][1,1], 
                xlabel="Δ SUVR", 
                ylabel="Δ Prediction",
                titlesize=26, xlabelsize=20, ylabelsize=20)
        if i > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        start = -1.0
        stop = 1.0
        xlims!(ax, start, stop)
        ylims!(ax, start, stop)
        lines!(start:0.1:stop, start:0.1:stop, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
        for i in eachindex(diffs)
            scatter!(diffs[i], soldiff[i], marker='o', markersize=15)
        end
    end

    f
end
save(projectdir("adni/visualisation/hier-inf/pred-tauneg.pdf"), f)
save(projectdir("adni/visualisation/hier-inf/png/pred-tauneg.png"), f)

begin
    f = Figure(resolution=(1500, 500))
    gl = [f[1, i] = GridLayout() for i in 1:3]
    for i in 1:3
        scan = i + 1

        if scan < 4
            diffs = getdiff.(subdata, scan)
            soldiff = getdiff.(sols, scan)
        else
            idx = findall(x -> size(x,2) == scan, subdata)
            diffs = getdiff.(subdata[idx], scan)
            soldiff = getdiff.(sols[idx], scan)
        end

        ax = Axis(gl[i][1,1], 
                xlabel="Δ SUVR", 
                ylabel="Δ Prediction",  
                title="Scan: $scan", 
                titlesize=26, xlabelsize=20, ylabelsize=20)
        if i > 1
            hideydecorations!(ax, ticks=false)
        end
        start = -1.0
        stop = 1.0
        xlims!(ax, start, stop)
        ylims!(ax, start, stop)
        lines!(start:0.1:stop, start:0.1:stop, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
        for i in eachindex(diffs)
            scatter!(diffs[i], soldiff[i], marker='o', markersize=15)
        end
    end
    f
end
save(projectdir("adni/visualisation/hier-inf/pred-delta-tauneg.pdf"), f)

begin
    f = Figure(resolution=(2000, 500))
    gl = [f[1, i] = GridLayout() for i in 1:4]
    for i in 1:4
        scan = i
        ax = Axis(gl[i][1,1], 
                xlabel="SUVR", 
                ylabel="Prediction", 
                title="Scan: $scan", 
                titlesize=26, xlabelsize=20, ylabelsize=20)

        ylims!(ax, 75, 150)
        xlims!(ax, 75, 150)
        lines!(75:5:150, 75:5:150, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
        for i in 1:21
            if size(subdata[i], 2) >= scan
                scatter!(vec(totaltau[i]), vec(totalsols[i]), markersize=15)
            end
        end
    end
    for (label, layout) in zip(["A", "B", "C", "D"], [gl...])
        Label(layout[1, 1, TopLeft()], label,
            textsize = 20,
            font = "TeX Gyre Heros Bold",
            padding = (0, 5, 5, 0),
            halign = :right)
    end
    f
end
save(projectdir("adni/visualisation/hier-inf/pred-totaltauneg.png"), f)

begin
    f = Figure()
    diffs = getdiff.(subdata)
    soldiff = getdiff.(sols)
    ax = Axis(f[1,1], 
            xlabel="Δ SUVR", 
            ylabel="Δ Prediction", 
            title="Difference between last scan and first scan")
    for i in 1:21
        scatter!(diffs[i], soldiff[i], marker='o', markersize=15)
    end
    f
end
save(projectdir("adni/visualisation/hier-inf/pred-delta-tauneg.png"), f)


begin
    f = Figure()
    diffs = getdiff.(totaltau)
    soldiff = getdiff.(totalsols)
    ax = Axis(f[1,1], 
            xlabel="Δ Total SUVR", 
            ylabel="Δ Total Prediction", 
            title="Difference between last scan and first scan")

    ylims!(ax, -10, 5)
    xlims!(ax, -10, 5)
    lines!(-10:1:5, -10:1:5, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
    for i in 1:21
        scatter!(diffs[i], soldiff[i], markersize=15)
    end
    f
end
save(projectdir("adni/visualisation/hier-inf/pred-delta-totaltauneg.png"), f)


begin
    f = Figure(resolution=(2000, 500))
    gl = [f[1, i] = GridLayout() for i in 1:4]
    for i in 1:4
    scan = i
    ax = Axis(gl[i][1,1], 
            xlabel="Total SUVR", 
            ylabel="Total Prediction", 
            title="Scan: $scan", 
            titlesize=26, xlabelsize=20, ylabelsize=20)

    ylims!(ax, 75, 150)
    xlims!(ax, 75, 150)
    lines!(75:5:150, 75:5:150, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
    for i in 1:27
        if size(subdata[i], 2) >= scan
            scatter!(vec(totaltau[i]), vec(totalmeansols[i]), markersize=10)
        end
    end
    end
    for (label, layout) in zip(["A", "B", "C", "D"], [gl...])
        Label(layout[1, 1, TopLeft()], label,
            textsize = 20,
            font = "TeX Gyre Heros Bold",
            padding = (0, 5, 5, 0),
            halign = :right)
    end
    f
end
save(projectdir("adni/visualisation/hier-inf/pred-totaltaupos.png"), f)

begin
    f = Figure()
    diffs = getdiff.(subdata)
    soldiff = getdiff.(sols)
    ax = Axis(f[1,1], 
            xlabel="Δ SUVR", 
            ylabel="Δ Prediction", 
            title="Difference between last scan and first scan")
    for i in 1:27
        scatter!(diffs[i], soldiff[i], marker='o', markersize=15)
    end
    f
end
save(projectdir("adni/visualisation/hier-inf/pred-delta-taupos.png"), f)
