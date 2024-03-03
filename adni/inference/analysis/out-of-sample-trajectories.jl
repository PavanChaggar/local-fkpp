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
include(projectdir("adni/inference/inference-preamble.jl"))
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

pst = deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-taupos-4x2000.jls"));
pst
meanpst = mean(pst);
params = [[meanpst[:Pm, :mean], meanpst[:Am, :mean]] for i in 1:57]
meansols = [solve(ODEProblem(NetworkLocalFKPP, init, (0.0,7.0), p), Tsit5(), saveat=0.05) for (init, t, p) in zip(initial_conditions, times, params)];

sols = Vector{Vector{Array{Float64}}}();

for (i, j) in enumerate(tau_pos)
    isols = Vector{Array{Float64}}()
    for s in 1:40:8000
        params = [pst[:Pm][s], pst[:Am][s]]
        σ = pst[:σ][s]
        sol = solve(ODEProblem(NetworkLocalFKPP, initial_conditions[i], (0.0,7.0), params), Tsit5(), saveat=0.1)
        noise = (randn(size(Array(sol))) .* σ)
        push!(isols, Array(sol) .+ noise)
    end
    push!(sols, isols)
end

function get_quantiles(mean_sols)
    [vec(mapslices(x -> quantile(x, q), mean_sols, dims = 2)) for q in [0.975, 0.025, 0.5]]
end

using CairoMakie; CairoMakie.activate!()

node=29
begin
    f = Figure(resolution=(1500,1800))
    g = f[1:4,:] = GridLayout()
    g2 = f[5:6,:] = GridLayout()
    ylabelsize = 30
    xlabelsize = 30
    for j in 1:5
        sub = 0 + j
        q1, q2, q3 = get_quantiles(reduce(hcat, [sols[sub][i][node, :] for i in 1:200]))

        ax = Axis(g[1, sub], ylabel="SUVR", ylabelsize=ylabelsize)
        ylims!(ax, 1.0,3.5)
        hidexdecorations!(ax, ticks = false, grid=false)
        if j > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        band!(0.0:0.1:7.0, q1, q2, color=(:grey, 0.5))
        lines!(meansols[sub].t, meansols[sub][node,:], color=(:red, 0.8), linewidth=3)
        scatter!(times[sub], subdata[sub][node,:], color=:navy)
    end
    f
    for j in 1:5
        sub = 5 + j
        q1, q2, q3 = get_quantiles(reduce(hcat, [sols[sub][i][node, :] for i in 1:200]))

        ax = Axis(g[2, j], ylabel="SUVR", ylabelsize=ylabelsize)
        hidexdecorations!(ax, ticks = false, grid=false)
        ylims!(ax, 1.0,3.5)
        if j > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        band!(0.0:0.1:7.0, q1, q2, color=(:grey, 0.5))
        lines!(meansols[sub].t, meansols[sub][node,:], color=(:red, 0.8), linewidth=3)
        scatter!(times[sub], subdata[sub][node,:], color=:navy)
    end
    f
    for j in 1:5
        sub = 10 + j
        q1, q2, q3 = get_quantiles(reduce(hcat, [sols[sub][i][node, :] for i in 1:200]))

        ax = Axis(g[3, j], ylabel="SUVR", ylabelsize=ylabelsize)
        hidexdecorations!(ax, ticks = false, grid=false)
        ylims!(ax, 1.0,3.5)
        if j > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        band!(0.0:0.1:7.0, q1, q2, color=(:grey, 0.5))
        lines!(meansols[sub].t, meansols[sub][node,:], color=(:red, 0.8), linewidth=3)
        scatter!(times[sub], subdata[sub][node,:], color=:navy)
    end
    for j in 1:5
        sub = 15 + j
        q1, q2, q3 = get_quantiles(reduce(hcat, [sols[sub][i][node, :] for i in 1:200]))

        ax = Axis(g[4, j], ylabel="SUVR", ylabelsize=ylabelsize)
        hidexdecorations!(ax, ticks = false, grid=false)
        if j > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        ylims!(ax, 1.0,3.5)
        band!(0.0:0.1:7.0, q1, q2, color=(:grey, 0.5))
        lines!(meansols[sub].t, meansols[sub][node,:], color=(:red, 0.8), linewidth=3)
        scatter!(times[sub], subdata[sub][node,:], color=:navy)
    end
    for j in 1:4
        sub = 20 + j
        q1, q2, q3 = get_quantiles(reduce(hcat, [sols[sub][i][node, :] for i in 1:200]))

        ax = Axis(g2[1, j], ylabel="SUVR", xlabel="Time / Years", 
                  ylabelsize=ylabelsize, xlabelsize=xlabelsize)
        if j > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end
        ylims!(ax, 1.0,3.5)
        band!(0.0:0.1:7.0, q1, q2, color=(:grey, 0.5))
        lines!(meansols[sub].t, meansols[sub][node,:], color=(:red, 0.8), linewidth=3)
        scatter!(times[sub], subdata[sub][node,:], color=:navy)
    end

    elem_1 = LineElement(color = (:red, 0.6), linewidth=5)
    elem_2 = MarkerElement(color = (:navy, 0.6), marker=:circ, markersize=20)
    elem_3 = PolyElement(color = (:lightgrey, 1.0))
    legend = Legend(g2[2,4:5],
           [elem_1, elem_3, elem_2],
           [" Mean Predictions", " 95% Quantile", " Observations"],
           patchsize = (100, 50), rowgap = 10, labelsize=40)
    rowgap!(g, 15)
    rowgap!(f.layout,15)
    rowgap!(g2,-20)
    f
end