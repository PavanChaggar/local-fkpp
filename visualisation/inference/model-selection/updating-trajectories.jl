using Connectomes
using ADNIDatasets
using CSV, DataFrames
using DrWatson: projectdir
using OrdinaryDiffEq
using Turing
using Distributions
using Serialization
using DelimitedFiles, LinearAlgebra
using Random, SparseArrays
using CairoMakie, ColorSchemes
using Colors
include(projectdir("functions.jl"))
#-------------------------------------------------------------------------------
# Connectome and ROIs
#-------------------------------------------------------------------------------
include(projectdir("adni/inference/inference-preamble.jl"))
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

#-------------------------------------------------------------------------------
# Posteriors
#-------------------------------------------------------------------------------
prr = deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-taupos-1x2000-updated-0.jls"));
pst1 = deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-taupos-1x2000-updated-1.jls"));
pst2 = deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-taupos-1x2000-updated-2.jls"));
pst3 = deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-taupos-1x2000-updated-3.jls"));

[sum(p[:numerical_error]) for p in [pst1, pst2, pst3]]
#-------------------------------------------------------------------------------
# Local model
#-------------------------------------------------------------------------------
function simulate(f, inits, p, tspan, t)
    solve(
        ODEProblem(
            f, inits, (0, tspan), p
        ), 
        Tsit5(), abstol=1e-9, reltol=1e-9, saveat=t
    )

end

function NetworkLocalFKPP(du, u, p, t; L = Lv, u0 = u0, cc = cc)
    du .= -p[1] * L * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

function make_prob_func(initial_conditions, p, a)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions, p=[p[i], a[i]], saveat=0.5)
    end
end

function output_func(sol,i)
    (sol,false)
end

prob = ODEProblem(NetworkLocalFKPP, insample_inits[1], (0.,10.), [1.0,1.0])

function ensemble_simulate(prob, insample_inits, ps, as, output_func)
    ensemble_prob = EnsembleProblem(prob, prob_func=make_prob_func(insample_inits, ps, as), output_func=output_func)
    solve(ensemble_prob, Tsit5(), abstol = 1e-6, reltol = 1e-6, trajectories=2000)
end

function get_sols(psts, prob, insample_inits, outsample_idx)
    sols = Vector{Vector{EnsembleSolution}}()
    for (i, pst) in enumerate(psts)
        _sols = Vector{EnsembleSolution}()
        for j in 1:16
                ps, as = vec(pst["ρ[$(41 + j)]"]), vec(pst["α[$(41 + j)]"])
                preds = ensemble_simulate(prob, insample_inits[j], ps, as, output_func)
                push!(_sols, preds)
            # elseif i == 4
            #     ps, as = vec(pst["ρ[$(outsample_idx[j])]"]), vec(pst["α[$(outsample_idx[j])]"])
            #     preds = ensemble_simulate(prob, insample_inits[j], ps, as, output_func)
            #     push!(_sols, preds)
            # end 
        end
        push!(sols, _sols)
    end
    sols
end

sols = get_sols([prr, pst1, pst2, pst3], prob, insample_inits, outsample_idx);

function add_noise(sol::ODESolution, noise::Float64)
    sol .+ (randn(size(sol)) .* noise)
end

function add_noise(sols, psts)
    noise_sols = Vector{Vector{Array}}()
    for (sol, pst) in zip(sols, psts)
        _noise_sols = Vector{Array}()
        noise = vec(pst[:σ])
        for j in 1:16
            push!(_noise_sols, [add_noise(_sol, n) for (_sol, n) in zip(sol[j], noise)])
        end
        push!(noise_sols, _noise_sols)
    end
    noise_sols
end

noise_sols = add_noise(sols, [prr, pst1, pst2, pst3]);

function get_quantiles(mean_sols)
    [vec(mapslices(x -> quantile(x, q), mean_sols, dims = 2)) for q in [0.975, 0.025, 0.5]]
end

node = 65
cols = ColorSchemes.seaborn_bright;

begin
    cols = Makie.wong_colors()
    f = Figure(size=(600, 500))
    idx = 2
    for (i, node) in enumerate([61, 63, 65])
        ax = Axis(f[i, 1:3], yticks=1:0.5:3.5, xticks=0:2.5:10, xlabel="Time / years", 
                  ylabel="SUVR")
        if i < 3 
            hidexdecorations!(ax, grid=false, ticks=false)
            hideydecorations!(ax, ticks=false, ticklabels=false, grid=false)
        end
        hidespines!(ax, :t, :r)
        xlims!(ax, -1.0,10.)
        ylims!(ax, 1.0,4.0)
        for (col, sol, alpha) in zip(cols[1:3], noise_sols[2:4], [0.25, 0.5, 0.75])
            q1, q2, q3 = get_quantiles(reduce(hcat, [sol[idx][j][node, :] for j in 1:2000]))
            band!(0.0:0.5:10.0, q1, q2, color=alphacolor(col, alpha))
            # scatter!(times[idx][1], four_subdata[idx][node,1], color=:black, markersize=25)
            # scatter!(times[idx][2:end], four_subdata[idx][node,2:end], color=:white, markersize=25, strokewidth = 3, strokcolor=:black)
        end
        for _t in 1:3
            scatter!(times[idx][_t], four_subdata[idx][node,_t], markersize=20, color=(cols[_t], 0.75), strokewidth=1, strokecolor=:black)
        end
        scatter!(times[idx][4:end], four_subdata[idx][node,4:end], markersize=20, color=(cols[4], 0.75), strokewidth=1, strokecolor=:black)
        Label(f[i, 1:3],
            "Left " * titlecase(get_label.(cortex)[node]),
            fontsize=15,
            justification = :right, halign=:right, valign=:top, 
            padding = (5.0f0, 0.0f0, 0.0f0, 5.0f0),
            color = :black, tellheight=false, tellwidth=false)
    end

    markersizes = [10, 10, 10, 10]
    colors = cols[1:4]
    polynames = ["1 Training Scan", "2 Training Scans", "3 Training Scans", "Out-of-Sample"]
    
    group_size = [MarkerElement(color = col, marker= '●', markersize=20) for col in cols[[1,2,3,4]]]
    
    Legend(f[:,4],
        [group_size],
        [string.(polynames)],
        [""], 
        patchsize=(10, 10), tellheight = true, tellwidth=false, orientation=:horizontal, 
        titleposition=:top, nbanks=4, framevisible=false, colgap=25, 
        titlegap=10, patchlabelgap=10)

    rowgap!(f.layout, 10)
    colgap!(f.layout, 20)
    f
end
save(projectdir("visualisation/inference/model-selection/output/out-of-sample-updates-five-subs.pdf"), f)

begin
    cols = Makie.wong_colors()
    alphas = [0.25, 0.5, 0.75]
    for k in 1:3
        f = Figure(size=(600, 500))
        idx = 2
        for (i, node) in enumerate([61, 63, 65])
            ax = Axis(f[i, 1:3], yticks=1:0.5:3.5, xticks=0:2.5:10, xlabel="Time / years", 
                    ylabel="SUVR")
            if i < 3 
                hidexdecorations!(ax, grid=false, ticks=false)
                hideydecorations!(ax, ticks=false, ticklabels=false, grid=false)
            end
            hidespines!(ax, :t, :r)
            xlims!(ax, -1.0,10.)
            ylims!(ax, 1.0,4.0)

            if k == 1
                q1, q2, q3 = get_quantiles(reduce(hcat, [noise_sols[k+1][idx][j][node, :] for j in 1:2000]))
                band!(0.0:0.5:10.0, q1, q2, color=alphacolor(cols[1], 0.25))

                scatter!(times[idx][1], four_subdata[idx][node,1], markersize=20, color=(cols[1], 0.75), strokewidth=1, strokecolor=:black)
                scatter!(times[idx][2:end], four_subdata[idx][node,2:end], markersize=20, color=(cols[4], 0.75), strokewidth=1, strokecolor=:black)
            else
                for _t in 1:k
                    q1, q2, q3 = get_quantiles(reduce(hcat, [noise_sols[_t+1][idx][j][node, :] for j in 1:2000]))
                    band!(0.0:0.5:10.0, q1, q2, color=alphacolor(cols[_t], alphas[_t]))

                    scatter!(times[idx][_t], four_subdata[idx][node,_t], markersize=20, color=(cols[_t], 0.75), strokewidth=1, strokecolor=:black)
                end
                scatter!(times[idx][k+1:end], four_subdata[idx][node,k+1:end], markersize=20, color=(cols[4], 0.75), strokewidth=1, strokecolor=:black)
            end
            Label(f[i, 1:3],
                "Left " * titlecase(get_label.(cortex)[node]),
                fontsize=15,
                justification = :right, halign=:right, valign=:top, 
                padding = (5.0f0, 0.0f0, 0.0f0, 5.0f0),
                color = :black, tellheight=false, tellwidth=false)
        end

        markersizes = [10,10]
        colors = cols[[1,4]]
        insample_colors = cols[1:k]
        outsample_colors = cols[4]
        colors = [insample_colors; outsample_colors]

        insample_names = reduce(vcat, ["$(scan) Training Scans" for scan in 1:k])
        polynames = [insample_names; "Out-of-Sample"]
        
        group_size = [MarkerElement(color = col, marker= '●', markersize=20) for col in colors]
        
        Legend(f[:,4],
            [group_size],
            [string.(polynames)],
            [""], 
            patchsize=(10, 10), tellheight = true, tellwidth=false, orientation=:horizontal, 
            titleposition=:top, nbanks=4, framevisible=false, colgap=25, 
            titlegap=10, patchlabelgap=10)

        rowgap!(f.layout, 10)
        colgap!(f.layout, 20)
        display(f)
        save(projectdir("visualisation/inference/model-selection/output/out-of-sample-updates-five-subs-$(k).pdf"), f)
    end
end

for node in [61, 63, 65]
    dktnames[node]
    begin
        _sols = sols
        sol_idx = [2, 3, 4]
        cols = Makie.wong_colors()
        f = Figure(size=(1600, 1600), fontsize=35)
        for i in 1:4
            ax = Axis(f[1,i], xticksize=10, yticks=1:0.5:3.5, xticks=0:2.5:10, 
                            ygridcolor=RGBAf(0, 0, 0, 0.35), xgridcolor=RGBAf(0, 0, 0, 0.35))
            if i > 1 
                hideydecorations!(ax, grid=false, ticks=false)
            end
            hidexdecorations!(ax, grid=false, ticks=false, label=false)
            xlims!(ax, -1.0,11.)
            ylims!(ax, 1.0,3.25)
            hidespines!(ax, :t, :r)
            for (sol, idx, alpha) in zip(_sols[sol_idx], [1, 2, 3], [0.25, 0.5, 0.75])
                q1, q2, q3 = get_quantiles(reduce(hcat, [sol[i][j][node, :] for j in 1:2000]))
                band!(0.0:0.5:10.0, q1, q2, color=alphacolor(cols[idx], alpha))
            end
            for idx in 1:3
                scatter!(times[i][idx], four_subdata[i][node,idx], markersize=35, color=(cols[idx], 0.75), strokewidth=3, strokecolor=:black)
            end
            scatter!(times[i][4:end], four_subdata[i][node,4:end], markersize=35, color=(cols[4], 0.75), strokewidth=3, strokecolor=:black)
        end
        for (k, i) in enumerate(5:8)
            ax = Axis(f[2,k], xticksize=10, yticks=1:0.5:3.5, xticks=0:2.5:10, 
                            ygridcolor=RGBAf(0, 0, 0, 0.35), xgridcolor=RGBAf(0, 0, 0, 0.35))
            if k > 1 
                hideydecorations!(ax, grid=false, ticks=false)
            end
            hidexdecorations!(ax, grid=false, ticks=false)
            xlims!(ax, -1.0,11.)
            ylims!(ax, 1.0,3.25)
            hidespines!(ax, :t, :r)
            for (sol, idx, alpha) in zip(_sols[sol_idx], [1, 2, 3], [0.25, 0.5, 0.75])
                q1, q2, q3 = get_quantiles(reduce(hcat, [sol[i][j][node, :] for j in 1:2000]))
                band!(0.0:0.5:10.0, q1, q2, color=alphacolor(cols[idx], alpha))
            end
            for idx in 1:3
                scatter!(times[i][idx], four_subdata[i][node,idx], markersize=35, color=(cols[idx], 0.75), strokewidth=3, strokecolor=:black)
            end
            scatter!(times[i][4:end], four_subdata[i][node,4:end], markersize=35, color=(cols[4], 0.75), strokewidth=3, strokecolor=:black)

        end
        for (k, i) in enumerate(9:12)
            ax = Axis(f[3,k], xticksize=10, yticks=1:0.5:3.5, xticks=0:2.5:10, 
                            ygridcolor=RGBAf(0, 0, 0, 0.35), xgridcolor=RGBAf(0, 0, 0, 0.35))
            if k > 1 
                hideydecorations!(ax, grid=false, ticks=false)
            end
            hidexdecorations!(ax, grid=false, ticks=false)
            xlims!(ax, -1.0,11.)
            ylims!(ax, 1.0,3.25)
            hidespines!(ax, :t, :r)
            for (sol, idx, alpha) in zip(_sols[sol_idx], [1, 2, 3], [0.25, 0.5, 0.75])
                q1, q2, q3 = get_quantiles(reduce(hcat, [sol[i][j][node, :] for j in 1:2000]))
                band!(0.0:0.5:10.0, q1, q2, color=alphacolor(cols[idx], alpha))
            end
            for idx in 1:3
                scatter!(times[i][idx], four_subdata[i][node,idx], markersize=35, color=(cols[idx], 0.75), strokewidth=3, strokecolor=:black)
            end
            scatter!(times[i][4:end], four_subdata[i][node,4:end], markersize=35, color=(cols[4], 0.75), strokewidth=3, strokecolor=:black)

        end
        for (k, i) in enumerate(13:16)
            ax = Axis(f[4,k], xticksize=10, yticks=1:0.5:3.5, xticks=0:2.5:10, 
                            ygridcolor=RGBAf(0, 0, 0, 0.35), xgridcolor=RGBAf(0, 0, 0, 0.35),
                            xlabel="Time / Years", ylabel="SUVR")
            if k > 1 
                hideydecorations!(ax, grid=false, ticks=false)
                hidexdecorations!(ax, grid=false, ticks=false, ticklabels=false)
            end
            xlims!(ax, -1.0,11.)
            ylims!(ax, 1.0,3.25)
            hidespines!(ax, :t, :r)
            for (sol, idx, alpha) in zip(_sols[sol_idx], [1, 2, 3], [0.25, 0.5, 0.75])
                q1, q2, q3 = get_quantiles(reduce(hcat, [sol[i][j][node, :] for j in 1:2000]))
                band!(0.0:0.5:10.0, q1, q2, color=alphacolor(cols[idx], alpha))
            end
            for idx in 1:3
                scatter!(times[i][idx], four_subdata[i][node,idx], markersize=35, color=(cols[idx], 0.75), strokewidth=3, strokecolor=:black)
            end
            scatter!(times[i][4:end], four_subdata[i][node,4:end], markersize=35, color=(cols[4], 0.75), strokewidth=3, strokecolor=:black)

        end
        rowgap!(f.layout, 10)
        colgap!(f.layout, 10)


        markersizes = [10, 10, 10, 10]
        colors = cols[1:4]
        scatternames = ["1 In-Sample Scan", "2 In-Sample Scans", "3 In-Sample Scans", "Out-of-sample Scans"]
        polynames = ["1 Training Scan", "2 Training Scans", "3 Training Scans"]
        
        group_size = [MarkerElement(color = col, marker= '●', markersize=30) for col in cols[1:4]]
        
        group_color = [PolyElement(color = alphacolor(col, 0.75)) for col in cols[1:3]]
        
        Legend(f[5,:],
            [group_size, group_color],
            [string.(scatternames), string.(polynames)],
            ["Data", "Predictions"], 
            patchsize=(50, 50), tellheight = true, orientation=:horizontal, titleposition=:top, nbanks=2,
            framevisible=false, colgap=25, titlegap=10, groupgap=50, patchlabelgap=10)
        display(f)
    end
    save(projectdir("visualisation/inference/model-selection/output/out-of-sample-updates-$(dktnames[node]).pdf"), f)
end
