using Connectomes
using DifferentialEquations
using DrWatson
using Distributions
using ADNIDatasets
using LinearAlgebra, SparseArrays
using Serialization, MCMCChains
include(projectdir("functions.jl"))
#-------------------------------------------------------------------------------
# Connectome and ROIs
#-------------------------------------------------------------------------------
connectome_path = Connectomes.connectome_path()
all_c = filter(Connectome(connectome_path; norm=true, weight_function = (n, l) -> n ./ l.^2), 1e-2);

subcortex = filter(x -> x.Lobe == "subcortex", all_c.parc)
cortex = filter(x -> x.Lobe != "subcortex", all_c.parc)

c = slice(all_c, cortex) |> filter

right_cortical_nodes = filter(x -> x.Hemisphere == "right", c.parc)

mtl_regions = ["entorhinal", "Left-Amygdala", "Right-Amygdala"]
mtl = findall(x -> x ∈ mtl_regions, cortex.Label)
neo_regions = ["inferiortemporal", "middletemporal"]
neo = findall(x -> x ∈ neo_regions, cortex.Label)
#-------------------------------------------------------------------------------
# Data 
#-------------------------------------------------------------------------------
sub_data_path = projectdir("adni/data/new_data/UCBERKELEYAV1451_8mm_02_17_23_AB_Status.csv")
alldf = CSV.read(sub_data_path, DataFrame)

#posdf = filter(x -> x.STATUS == "POS", alldf)
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
ubase, upath = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(ubase)
cc = quantile.(upath, .99)
#-------------------------------------------------------------------------------
# Connectome + ODE
#-------------------------------------------------------------------------------
L = laplacian_matrix(c)

vols = [get_vol(data, i) for i in tau_neg]
init_vols = [v[:,1] for v in vols]
max_norm_vols = reduce(hcat, [v ./ maximum(v) for v in init_vols])
mean_norm_vols = vec(mean(max_norm_vols, dims=2))
Lv = sparse(inv(diagm(mean_norm_vols)) * L)

function NetworkLocalFKPP(du, u, p, t; Lv = Lv, u0 = u0, cc = cc)
    du .= -p[1] * Lv * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

subsuvr = [calc_suvr(data, i) for i in tau_neg]
_subdata = [normalise(sd, u0, cc) for sd in subsuvr]

blsd = [sd .- u0 for sd in _subdata]
nonzerosubs = findall(x -> sum(x) < 2, [sum(sd, dims=1) .== 0 for sd in blsd])
subdata = _subdata[nonzerosubs]
#-------------------------------------------------------------------------------
# Subject params
#-------------------------------------------------------------------------------
pst = deserialize(projectdir("adni/chains/local-fkpp/pst-tauneg-4x2000.jls"));
meanpst = mean(pst);
sub = 2
ρ = meanpst["ρ[$sub]", :mean]
α = meanpst["α[$sub]", :mean]
prob = ODEProblem(NetworkLocalFKPP, subdata[sub][:,1], (0.0,20.0), [ρ,α])

ts = range(0.0, 20.0, 5)
n = length(ts)
sol = solve(prob, Rodas4(), reltol=1e-12, saveat=ts);
allsol = solve(prob, Rodas4(), reltol=1e-12, saveat=0.1)

Plots.plot(sol, vars=(1:36), labels=false)

solcol = [(sol[i] .- minimum(u0)) ./ (maximum(cc) .- minimum(u0)) for i in 1:n]

using GLMakie; GLMakie.activate!()
using ColorSchemes

begin
    cmap = reverse(ColorSchemes.RdYlBu); #ColorSchemes.viridis 
    cols = [get(cmap, solcol[i]) for i in 1:n]
    nodes = right_cortical_nodes.ID;

    f = Figure(resolution=(2000, 500))
    g1 = f[1, 1] = GridLayout()
    # g2 = f[2, 1] = GridLayout()
    for k in 1:n
        ax = Axis3(g1[1,k+1], 
                   aspect = :data, 
                   azimuth = 1.0pi, 
                   elevation=0.0pi,
                   protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        for (i, j) in enumerate(nodes)
            plot_roi!(j, cols[k][i])
        end
        ax = Axis3(g1[2,k+1], 
                   aspect = :data, 
                   azimuth = 0.0pi, 
                   elevation=0.0pi,
                   protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        for (i, j) in enumerate(nodes)
            plot_roi!(j, cols[k][i])
        end
    end
    # rowsize!(f.layout, 1, Auto(0.8))
    # rowsize!(f.layout, 2, Auto(0.8))
    Colorbar(g1[1:2, 1], limits = (minimum(u0), maximum(cc)), colormap = cmap,
    label = "SUVR", labelsize=36, flipaxis=false,
    ticksize=18, ticks=collect(0.0:0.5:3.5), ticklabelsize=25, labelpadding=3)

    # ax = Axis(g2[1:3,1:6],
    #         xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.05), xgridwidth = 2,
    #         xticklabelsize = 25, xticks = LinearTicks(5), xticksize=18,
    #         xlabel="Time", xlabelsize = 36,
    #         yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
    #         yticklabelsize = 25, yticks = collect(0:0.5:3.5), yticksize=18,
    #         ylabel="SUVR", ylabelsize = 36
    # )
    # GLMakie.ylims!(ax, minimum(u0) - 0.1, 4.0)
    # GLMakie.xlims!(ax, 0.0, 20.05)
    # # hideydecorations!(ax, label=false, ticks=false, ticklabels=false)
    # hidespines!(ax, :t, :r)
    # for i in 1:36
    #     lines!(allsol.t, allsol[i, :], linewidth=2)
    # end

    f
end
save(projectdir("visualisation/inference/pstpred/output/tauneg-preditions-cortex.jpeg"), f)

#-------------------------------------------------------------------------------
# Connectome + ODE Tau Pos
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

subsuvr = [calc_suvr(data, i) for i in tau_pos]
subdata = [normalise(sd, u0, cc) for sd in subsuvr]
#-------------------------------------------------------------------------------
# Subject params
#-------------------------------------------------------------------------------
pst = deserialize(projectdir("adni/chains/local-fkpp/pst-taupos-4x2000.jls"));
meanpst = mean(pst);
sub = 10
ρ = meanpst["ρ[$sub]", :mean]
α = meanpst["α[$sub]", :mean]
prob = ODEProblem(NetworkLocalFKPP, subdata[sub][:,1], (0.0,40.0), [ρ,α])

ts = range(0.0, 40.0, 5)
n = length(ts)
sol = solve(prob, Rodas4(), reltol=1e-12, saveat=ts);
allsol = solve(prob, Rodas4(), reltol=1e-12, saveat=0.1)

# Plots.plot(sol, vars=(1:36), labels=false)

solcol = [(sol[i] .- minimum(u0)) ./ (maximum(cc) .- minimum(u0)) for i in 1:n]

using GLMakie; GLMakie.activate!()
using ColorSchemes

begin
    cmap = reverse(ColorSchemes.RdYlBu); #ColorSchemes.viridis 
    cols = [get(cmap, solcol[i]) for i in 1:n]
    nodes = right_cortical_nodes.ID;

    f = Figure(resolution=(2000, 500))
    g1 = f[1, 1] = GridLayout()
    g2 = f[2, 1] = GridLayout()
    for k in 1:n
        ax = Axis3(g1[1,k+1], 
                   aspect = :data, 
                   azimuth = 1.0pi, 
                   elevation=0.0pi,
                   protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        for (i, j) in enumerate(nodes)
            plot_roi!(j, cols[k][i])
        end
        ax = Axis3(g1[2,k+1], 
                   aspect = :data, 
                   azimuth = 0.0pi, 
                   elevation=0.0pi,
                   protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        for (i, j) in enumerate(nodes)
            plot_roi!(j, cols[k][i])
        end
    end
    # rowsize!(f.layout, 1, Auto(0.8))
    # rowsize!(f.layout, 2, Auto(0.8))
    Colorbar(g1[1:2, 1], limits = (minimum(u0), maximum(cc)), colormap = cmap,
    label = "SUVR", labelsize=36, flipaxis=false,
    ticksize=18, ticks=collect(0.0:0.5:3.5), ticklabelsize=25, labelpadding=3)

    ax = Axis(g2[1:3,1:6],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.05), xgridwidth = 2,
            xticklabelsize = 25, xticks = LinearTicks(5), xticksize=18,
            xlabel="Time", xlabelsize = 36,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticks = collect(0:0.5:3.5), yticksize=18,
            ylabel="SUVR", ylabelsize = 36
    )
    GLMakie.ylims!(ax, minimum(u0) - 0.1, 4.0)
    GLMakie.xlims!(ax, 0.0, 40.05)
    # hideydecorations!(ax, label=false, ticks=false, ticklabels=false)
    hidespines!(ax, :t, :r)
    for i in 1:36
        lines!(allsol.t, allsol[i, :], linewidth=2, color=(:grey, 0.5))
    end
    f
end
save(projectdir("visualisation/inference/pstpred/output/taupos-preditions-cortex.jpeg"), f)

