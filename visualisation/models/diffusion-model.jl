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
    du .= -p[1] * L * u
end

p0 = zeros(72)
seed_regions =  ["entorhinal"] # "Left-Amygdala", "Right-Amygdala", "Left-Hippocampus", "Right-Hippocampus"]
seed = findall(x -> get_label(x) ∈ seed_regions, cortex)
p0[seed] .= 0.5

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

cmap = ColorSchemes.viridis;
cols = [get(cmap, sol[i]) for i in 1:n]
nodes = get_node_id.(right_cortical_nodes);
line_cols = Makie.wong_colors()[1]
begin
    f = Figure(size=(1200, 800))
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