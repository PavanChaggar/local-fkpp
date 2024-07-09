using Connectomes
using DifferentialEquations
using DrWatson
using Distributions
using DataFrames, CSV
using LinearAlgebra
include(projectdir("functions.jl"))

connectome_path = Connectomes.connectome_path()
all_c = filter(Connectome(connectome_path; norm=true, weight_function = (n, l) -> n), 1e-2);

subcortex = filter(x -> get_lobe(x) == "subcortex", all_c.parc)
cortex = filter(x -> get_lobe(x) != "subcortex", all_c.parc)

c = slice(all_c, cortex) |> filter

right_cortical_nodes = filter(x -> get_hemisphere(x) == "right", c.parc)
rIDs =  get_node_id.(right_cortical_nodes)
L = laplacian_matrix(c);

function NetworkDiffusion(du, u, p, t)
    du .= -p[1] * L * (u .- p[2])
end

gmm_moments = CSV.read(projectdir("adni/data/component_moments.csv"), DataFrame)
ubase, upath = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(ubase)
cc = quantile.(upath, .99)

p0 = fill(minimum(u0), 72)
seed_regions = ["entorhinal"] # "Left-Amygdala", "Right-Amygdala", "Left-Hippocampus", "Right-Hippocampus"]
seed = findall(x -> get_label(x) ∈ seed_regions, cortex)
seed_value = mean([maximum(cc), minimum(u0)])
p0[seed] .= seed_value 

r = 1.0
prob = ODEProblem(NetworkDiffusion, p0, (0.0,10.0), r)

ts = range(0.0, 10.0, 5)
n = length(ts)
sol = solve(prob, Rodas4(), reltol=1e-12, saveat=ts);
allsol = solve(prob, Rodas4(), reltol=1e-12, saveat=0.1)

using Plots
Plots.plot(sol, vars=(1:36), labels=false)
Plots.plot(allsol, vars=(1:36), labels=false)

using GLMakie, ColorSchemes

cmap = reverse(ColorSchemes.RdYlBu);
cols = [get(cmap, sol[i]) for i in 1:n]
nodes = get_node_id.(right_cortical_nodes);
line_cols = Makie.wong_colors()[1]
begin
    f = Figure(resolution=(1200, 800))
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
        plot_roi!(rIDs, sol[k], cmap)
        # for (i, j) in enumerate(nodes)
        #     plot_roi!(j, cols[k][i])
        # end
        ax = Axis3(g1[2,k+1], 
                   aspect = :data, 
                   azimuth = 0.0pi, 
                   elevation=0.0pi,
                   protrusions=(1.0,1.0,50.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(rIDs, sol[k], cmap)
        # for (i, j) in enumerate(nodes)
        #     plot_roi!(j, cols[k][i])
        # end
    end
    # rowsize!(f.layout, 1, 10)
    # rowsize!(f.layout, 2, 10)
    c = Colorbar(g1[1:2, 1], limits = (0, 1), colormap = cmap,
    label = "Concentration", labelsize=36, flipaxis=false,
    ticksize=18, ticks=collect(0.0:0.2:1.0), ticklabelsize=25, labelpadding=3)

    
    ax = Axis(g2[1:3,1:6],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.05), xgridwidth = 2,
            xticklabelsize = 25, xticks = LinearTicks(5), xticksize=18,
            xlabel="Time", xlabelsize = 36,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticks = collect(0:0.2:1.0), yticksize=18,
            ylabel="Concentration", ylabelsize = 36
    )
    GLMakie.ylims!(ax, 0.0, 1.0)
    GLMakie.xlims!(ax, 0.0, 10.05)
    # hideydecorations!(ax, label=false, ticks=false, ticklabels=false)
    hidespines!(ax, :t, :r)
    for i in 1:36
        lines!(allsol.t, allsol[i, :], linewidth=2, color=(line_cols, 0.5))
    end

    # for (label, layout) in zip(["A", "B"], [g1, g2])
    #     Label(layout[1, 1, TopLeft()], label,
    #         textsize = 50,
    #         font = "TeX Gyre Heros Bold",
    #         padding = (0, 25, 25, 0),
    #         halign = :right)
    # end
    f
end
save(projectdir("visualisation/models/output/diffusion.jpeg"), f)

begin 
    p1 = Vector{Mesh}(undef, length(nodes))
    p2 = Vector{Mesh}(undef, length(nodes))

    f = Figure(size=(1200, 600))
    ax = Axis3(f[2:3,1], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax)
    hidespines!(ax)
    p1 .= plot_roi!(nodes, solcol[1], cmap)
    plot_roi!(subcortical_nodes, ones(5) .* 0.75, ColorSchemes.Greys)

    ax = Axis3(f[4:5,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax)
    hidespines!(ax)
    p2 .= plot_roi!(nodes, solcol[1], cmap)
    plot_roi!(subcortical_nodes, ones(5) .* 0.75, ColorSchemes.Greys)

    c = Colorbar(f[2:5, 0], limits = (1.0, maximum(cc)), colormap = cmap,
        vertical = true, label = "SUVR", labelsize=36, flipaxis=false,
        ticksize=18, ticklabelsize=36, labelpadding=3, ticks = 1:0.5:maximum(cc))

    ax = Axis(f[1,2:4])
    hidedecorations!(ax)
    hidespines!(ax)
    text!(L"\frac{ds_i}{dt} = - \rho \sum_j^{R} L_{ij} (s_{j} - s_{0, j}) + \alpha (s_i - s_{0, i}) [(s_{\infty,i} - s_{0,i}) - (s_i - s_{0,i})]", 
        fontsize=20, 
        align = (:center, :center))
    #rowsize!(f.layout, 1, Auto(0.15))

    ax = Axis(f[2:5,2:4],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 36, xticks = LinearTicks(5), xticksize=18,
            xlabel="Time / years", xlabelsize = 36,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 36, yticks = 1:0.5:3.5, yticksize=18)
    hidespines!(ax, :t, :r)
    hideydecorations!(ax, ticks=false, grid=false)
    GLMakie.ylims!(ax, 1.0, maximum(cc))
    GLMakie.xlims!(ax, 0, 30)

    for i in 1:36
        lines!(sol.t, sol[i, :], color=Makie.wong_colors()[1])
    end

    x = Observable(0.0)
    vlines!(ax, x, color=(:red, 0.5), linewidth=5)

    # sublayout = GridLayout(width = 75)
    # f[1,3] = sublayout

    # sublayout = GridLayout(height = 50)
    # f[6, 1:2] = sublayout
    f
end


r = 0.25
prob = ODEProblem(NetworkDiffusion, p0, (0.0,30.0), [r, minimum(u0)])
ts = LinRange(0.0, 30., 480)
sol = solve(prob, Rodas4(), reltol=1e-12, saveat=ts)
Plots.plot(sol, label=false)
n = length(sol)

solcol = [(sol[i] .- 1.0) ./ (maximum(cc) .- 1.0) for i in 1:n]
cmap = ColorSchemes.viridis;
cols = [get(cmap, solcol[i]) for i in 1:n]

nodes = get_node_id.(right_cortical_nodes);
subcortical_nodes = get_node_id.(right_subcortical_nodes);

begin 
    p1 = Vector{Mesh}(undef, length(nodes))
    p2 = Vector{Mesh}(undef, length(nodes))

    f = Figure(size=(1200, 600), figure_padding = 20)
    ax = Axis3(f[2:3,1], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax)
    hidespines!(ax)
    p1 .= plot_roi!(nodes, solcol[1], cmap)
    plot_roi!(subcortical_nodes, ones(5) .* 0.75, ColorSchemes.Greys)

    ax = Axis3(f[4:5,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax)
    hidespines!(ax)
    p2 .= plot_roi!(nodes, solcol[1], cmap)
    plot_roi!(subcortical_nodes, ones(5) .* 0.75, ColorSchemes.Greys)

    c = Colorbar(f[2:5, 0], limits = (1.0, maximum(cc)), colormap = cmap,
        vertical = true, label = "SUVR", labelsize=36, flipaxis=false,
        ticksize=18, ticklabelsize=36, labelpadding=3, ticks = 1:0.5:maximum(cc))

    ax = Axis(f[1,2:4])
    hidedecorations!(ax)
    hidespines!(ax)
    text!(L"\frac{\mathrm{d}s_i}{\mathrm{d}t} = - \rho \sum_{j=1}^{R} L_{ij} (s_{j} - s_{0})", # + \alpha (s_i - s_{0}) [(s_{\infty} - s_{0}) - (s_i - s_{0})]", 
        fontsize=20, 
        align = (:center, :center))
    #rowsize!(f.layout, 1, Auto(0.15))

    ax = Axis(f[2:5,2:4],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 36, xticks = LinearTicks(5), xticksize=18,
            xlabel="Time / years", xlabelsize = 36,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 36, yticks = 1:0.5:3.5, yticksize=18)
    hidespines!(ax, :t, :r)
    hideydecorations!(ax, ticks=false, grid=false)
    GLMakie.ylims!(ax, 1.0, maximum(cc))
    GLMakie.xlims!(ax, 0, 30)

    for i in 1:36
        lines!(sol.t, sol[i, :], color=Makie.wong_colors()[1])
    end

    x = Observable(0.0)
    vlines!(ax, x, color=(:red, 0.5), linewidth=5)

    # sublayout = GridLayout(width = 75)
    # f[1,3] = sublayout

    # sublayout = GridLayout(height = 50)
    # f[6, 1:2] = sublayout
    f
end

frames = 10 * 48

record(f, projectdir("visualisation/models/output/diffusion-video.mp4"), 1:frames; framerate=48) do i
    x[] = ts[i]
    for k in 1:36
        p1[k].color[] = cols[i][k]
        p2[k].color[] = cols[i][k]
    end
end