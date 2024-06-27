using Connectomes
using DifferentialEquations
using DrWatson
using Distributions
include(projectdir("functions.jl"))
include(projectdir("braak-regions.jl"))

connectome_path = Connectomes.connectome_path()
all_c = filter(Connectome(connectome_path; norm=true, weight_function = (n, l) -> n), 1e-2);

subcortex = filter(x -> x.Lobe == "subcortex", all_c.parc)
cortex = filter(x -> x.Lobe != "subcortex", all_c.parc)

c = slice(all_c, cortex) |> filter

right_cortical_nodes = filter(x -> get_hemisphere(x) == "right", c.parc)
rIDs =  get_node_id.(right_cortical_nodes)

L = laplacian_matrix(c)

function NetworkFKPP(du, u, p, t)
    du .= -p[1] * L * u .+ p[2] .* u .* (1 .- u)
end

p0 = zeros(72)
seed_regions =  ["entorhinal"] # "Left-Amygdala", "Right-Amygdala", "Left-Hippocampus", "Right-Hippocampus"]
seed = findall(x -> get_label(x) ∈ seed_regions, cortex)
p0[seed] .= 0.5

r = 0.15
a = 1.5
prob = ODEProblem(NetworkFKPP, p0, (0.0,10.0), [r,a])

ts = range(0.0, 10.0, 5)
n = length(ts)
sol = solve(prob, Rodas4(), reltol=1e-12, saveat=ts);
allsol = solve(prob, Rodas4(), reltol=1e-12, saveat=0.1)

using Plots
Plots.plot(sol, vars=(1:36), labels=false)
Plots.plot(allsol, vars=(1:36), labels=false)

using GLMakie, ColorSchemes

cmap = ColorSchemes.viridis
cols = [get(cmap, sol[i]) for i in 1:n]
nodes = rIDs

braak_stages = map(getbraak, braak)
right_braak_stages = [filter(x -> x ∈ nodes, br) for br in braak_stages]
braak_colors = Makie.wong_colors();

begin
    f = Figure(resolution=(1200, 1200))
    g1 = f[1, 1] = GridLayout()
    g2 = f[2:3, 1] = GridLayout()
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
                   protrusions=(1.0,1.0,50.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        for (i, j) in enumerate(nodes)
            plot_roi!(j, cols[k][i])
        end
    end
    # rowsize!(f.layout, 1, Auto(0.8))
    # rowsize!(f.layout, 2, Auto(0.8))
    c = Colorbar(g1[1:2, 1], limits = (0, 1), colormap = cmap, flipaxis=false,
    ticksize=18, ticks=collect(0.0:0.2:1.0), ticklabelsize=25, labelpadding=3)
    Label(g1[1:2, 0], "Concentration", fontsize=35, rotation = pi/2)

    ax = Axis(g2[1:2,1:4],
    xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.05), xgridwidth = 2,
    xticklabelsize = 25, xticks = LinearTicks(5), xticksize=18,
    xlabel="Time / yrs", xlabelsize = 36,
    yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
    yticklabelsize = 25, yticks = collect(0:0.2:1.0), yticksize=18
)
    Label(g2[1:2, 0], "Conc.", fontsize=35, rotation = pi/2)
    GLMakie.ylims!(ax, 0, 1)
    GLMakie.xlims!(ax, 0.0, 10.05)
    hidexdecorations!(ax, grid=false, ticks=false)
    hidespines!(ax, :t, :r)
    for i in 1:36
        lines!(allsol.t, allsol[i, :], linewidth=2, color=(braak_colors[1], 0.5))
    end
    ax = Axis(g2[3:4,1:4],
    xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.05), xgridwidth = 2,
    xticklabelsize = 25, xticks = LinearTicks(5), xticksize=18,
    xlabel="Time / yrs", xlabelsize = 30,
    yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
    yticklabelsize = 25, yticks = collect(0:0.2:1.), yticksize=18
    )
    Label(g2[3:4, 0], "Conc.", fontsize=35, rotation = pi/2)
    GLMakie.ylims!(ax, 0.0, 1.0)
    GLMakie.xlims!(ax, 0.0, 10.05)
    # hideydecorations!(ax, label=false, ticks=false, ticklabels=false)
    hidespines!(ax, :t, :r)
    for (j, braak_stage) in enumerate(right_braak_stages)
        for i in braak_stage
            mean_braak_sol = vec(mean(allsol[braak_stage,:], dims=1))
            lines!(allsol.t, mean_braak_sol, linewidth=5, color=braak_colors[j])
        end
    end

    braak_labels = [LineElement(color = bc, linestyle = nothing, linewidth=3) for bc in braak_colors[1:5]]

    Legend(g2[5, 1:4],
    braak_labels,
    ["Braak 1", "Braak 2/3", "Braak 4", "Braak 5", "Braak 6"],
    patchsize = (100, 30), colgap = 10, orientation = :horizontal, fontsize=30)
    f
end
save(projectdir("visualisation/models/output/global-fkpp-braak.jpeg"), f)