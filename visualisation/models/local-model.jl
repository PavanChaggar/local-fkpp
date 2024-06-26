using Connectomes
using DifferentialEquations
using DrWatson
using Distributions
include(projectdir("functions.jl"))
include(projectdir("braak-regions.jl"))
connectome_path = Connectomes.connectome_path()
all_c = filter(Connectome(connectome_path; norm=true, weight_function = (n, l) -> n), 1e-2);

subcortex = filter(x -> get_lobe(x) == "subcortex", all_c.parc)
cortex = filter(x -> get_lobe(x) != "subcortex", all_c.parc)

c = slice(all_c, cortex) |> filter

right_cortical_nodes = filter(x -> get_hemisphere(x) == "right", c.parc)
rIDs =  get_node_id.(right_cortical_nodes)
L = laplacian_matrix(c);

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in get_node_id.(cortex)]

gmm_moments = CSV.read(projectdir("adni/data/component_moments.csv"), DataFrame)
ubase, upath = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(ubase)
cc = quantile.(upath, .99)

function NetworkExFKPP(du, u, p, t; L = L, u0 = u0, cc = cc)
    du .= -p[1] * L * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

p0 = mean.(ubase);
seed_regions = ["entorhinal"] # "Left-Amygdala", "Right-Amygdala", "Left-Hippocampus", "Right-Hippocampus"]
seed = findall(x -> get_label(x) ∈ seed_regions, cortex)
seed_value = mean([cc[seed] p0[seed]], dims=2)
p0[seed] .= seed_value 

r = 0.15
a = 1.1
prob = ODEProblem(NetworkExFKPP, p0, (0.0,8.0), [r,a])

ts = range(0.0, 8.0, 5)
n = length(ts)
sol = solve(prob, Rodas4(), reltol=1e-12, saveat=ts);
allsol = solve(prob, Rodas4(), reltol=1e-12, saveat=0.1)

nodes = get_node_id.(right_cortical_nodes);

braak_stages = map(getbraak, braak)

right_braak_stages = [filter(x -> x ∈ nodes, br) for br in braak_stages]

# using Plots
# Plots.plot(sol, vars=(1:36), labels=false)

using GLMakie; GLMakie.activate!()
using ColorSchemes

begin
    cmap = ColorSchemes.viridis; 
    braak_colors = Makie.wong_colors()
    endsol = sol[end]
    solcol = [(sol[i] .- minimum(u0)) ./ (maximum(cc) .- minimum(u0)) for i in 1:n]
    braak_sol = reduce(hcat, [(allsol[i] .- u0) ./ (endsol .- u0) for i in 1:length(allsol)])
    f = Figure(size=(1200, 1200))
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
        plot_roi!(rIDs, solcol[k], cmap)

        ax = Axis3(g1[2,k+1], 
                   aspect = :data, 
                   azimuth = 0.0pi, 
                   elevation=0.0pi,
                   protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(rIDs, solcol[k], cmap)
    end
    # rowsize!(f.layout, 1, Auto(0.8))
    # rowsize!(f.layout, 2, Auto(0.8))
    Colorbar(g1[1:2, 1], limits = (minimum(u0), maximum(cc)), colormap = cmap, flipaxis=false,
    ticksize=18, ticks=collect(0.0:0.5:3.5), ticklabelsize=25, labelpadding=3)
    Label(g1[1:2, 0], "SUVR", fontsize=35, rotation = pi/2)
    
    ax = Axis(g2[1:2,1:4],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.05), xgridwidth = 2,
            xticklabelsize = 25, xticks = LinearTicks(5), xticksize=18,
            xlabel="Time", xlabelsize = 36,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticks = collect(0:1.:4.0), yticksize=18
    )
    Label(g2[1:2, 0], "SUVR", fontsize=35, rotation = pi/2)
    GLMakie.ylims!(ax, minimum(u0) - 0.1, 4.1)
    GLMakie.xlims!(ax, 0.0, 8.05)
    hidexdecorations!(ax, grid=false, ticks=false)
    hidespines!(ax, :t, :r)
    for i in 1:36
        lines!(allsol.t, allsol[i, :], linewidth=2, color=(braak_colors[1], 0.5))
    end
    ax = Axis(g2[3:4,1:4],
    xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.05), xgridwidth = 2,
    xticklabelsize = 25, xticks = LinearTicks(5), xticksize=18,
    xlabel="Time", xlabelsize = 30,
    yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
    yticklabelsize = 25, yticks = collect(0:0.2:1.), yticksize=18
    )
    Label(g2[3:4, 0], "Conc.", fontsize=35, rotation = pi/2)
    GLMakie.ylims!(ax, 0.0, 1.0)
    GLMakie.xlims!(ax, 0.0, 8.05)
    # hideydecorations!(ax, label=false, ticks=false, ticklabels=false)
    hidespines!(ax, :t, :r)
    for (j, braak_stage) in enumerate(right_braak_stages)
        for i in braak_stage
            mean_braak_sol = vec(mean(braak_sol[braak_stage,:], dims=1))
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
save(projectdir("visualisation/models/output/local-fkpp.jpeg"), f)


r = 0.1
a = 0.8
prob = ODEProblem(NetworkExFKPP, p0, (0.0,12.0), [r,a])

ts = range(0.0, 12.0, 4)
n = length(ts)
sol = solve(prob, Rodas4(), reltol=1e-12, saveat=ts);
allsol = solve(prob, Rodas4(), reltol=1e-12, saveat=0.1)

begin
    cmap = ColorSchemes.viridis
    braak_colors = Makie.wong_colors()
    endsol = sol[end]
    solcol = [(sol[i] .- minimum(u0)) ./ (maximum(cc) .- minimum(u0)) for i in 1:n]
    braak_sol = reduce(hcat, [(allsol[i] .- u0) ./ (endsol .- u0) for i in 1:length(allsol)])
    f = Figure(size=(600, 800))
    g1 = f[1, 1] = GridLayout()
    g2 = f[2:4, 1] = GridLayout()
    for k in 1:n
        ax = Axis3(g1[1,k+1], 
                   aspect = :data, 
                   azimuth = 1.0pi, 
                   elevation=0.0pi,
                   protrusions=(0.0,0.0,0.0,0.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(rIDs, solcol[k], cmap)

        ax = Axis3(g1[2,k+1], 
                   aspect = :data, 
                   azimuth = 0.0pi, 
                   elevation=0.0pi,
                   protrusions=(0.0,0.0,0.0,0.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(rIDs, solcol[k], cmap)
    end
    # rowsize!(f.layout, 1, Auto(0.8))
    # rowsize!(f.layout, 2, Auto(0.8))
    Colorbar(g1[1:2, 1], limits = (minimum(u0), maximum(cc)), colormap = cmap, flipaxis=false,
    ticksize=18, ticks=collect(0.0:0.5:3.5), ticklabelsize=20, labelpadding=3)
    Label(g1[1:2, 0], "SUVR", fontsize=25, rotation = pi/2)
    
    ax = Axis(g2[1:2,1:4],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.05), xgridwidth = 2,
            xticklabelsize = 20, xticks = LinearTicks(4), xticksize=18,
            xlabel="Time / Years", xlabelsize = 36,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 20, yticks = collect(1:1:3.0), yticksize=18
    )
    Label(g2[1:2, 0], "SUVR", fontsize=25, rotation = pi/2)
    GLMakie.ylims!(ax, minimum(u0) - 0.1, 4.1)
    GLMakie.xlims!(ax, 0.0, 12.05)
    hidexdecorations!(ax, grid=false, ticks=false)
    hidespines!(ax, :t, :r)
    for i in 1:36
        lines!(allsol.t, allsol[i, :], linewidth=1.5, color=(braak_colors[1], 0.75))
    end
    ax = Axis(g2[3:4,1:4],
    xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.05), xgridwidth = 2,
    xticklabelsize = 20, xticks = LinearTicks(4), xticksize=18,
    xlabel="Time / Years", xlabelsize = 25,
    yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
    yticklabelsize = 20, yticks = collect(0:0.5:1.), yticksize=18
    )
    Label(g2[3:4, 0], "Conc.", fontsize=25, rotation = pi/2)
    GLMakie.ylims!(ax, 0.0, 1.05)
    GLMakie.xlims!(ax, 0.0, 12.05)
    # hideydecorations!(ax, label=false, ticks=false, ticklabels=false)
    hidespines!(ax, :t, :r)
    for (j, braak_stage) in enumerate(right_braak_stages)
        for i in reverse(braak_stage)
            mean_braak_sol = vec(mean(braak_sol[braak_stage,:], dims=1))
            lines!(allsol.t, mean_braak_sol, linewidth=5, color=reverse(braak_colors[1:5])[j])
        end
    end

    braak_labels = [LineElement(color = bc, linestyle = nothing, linewidth=3) for bc in reverse(braak_colors[1:5])]

    Legend(g2[5, 1:4],
        braak_labels,
        ["Braak 1", "Braak 2/3", "Braak 4", "Braak 5", "Braak 6"],
        patchsize = (20, 40), colgap = 10, orientation = :horizontal, fontsize=30)
        
    f
end
save(projectdir("visualisation/models/output/local-fkpp-small.jpeg"), f)

prob = ODEProblem(NetworkExFKPP, p0, (0.0,8.0), [r,a])
ts = LinRange(0.0, 8., 480)
sol = solve(prob, Rodas4(), reltol=1e-12, saveat=ts)
n = length(sol)

solcol = [(sol[i] .- minimum(u0)) ./ (maximum(cc) .- minimum(u0)) for i in 1:n]
cmap = reverse(ColorSchemes.RdYlBu);
cols = [get(cmap, solcol[i]) for i in 1:n]
nodes = get_node_id.(right_cortical_nodes)

begin 
    p1 = Vector{Mesh}(undef, length(nodes))
    p2 = Vector{Mesh}(undef, length(nodes))

    f = Figure(resolution=(2560, 1200))
    ax = Axis3(f[1:2,1], aspect = :data, azimuth = 1.0pi, elevation=0.0pi)
    hidedecorations!(ax)
    hidespines!(ax)
    p1 .= plot_roi!(nodes, solcol[1], cmap)

    ax = Axis3(f[3:4,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi)
    hidedecorations!(ax)
    hidespines!(ax)
    p2 .= plot_roi!(nodes, solcol[1], cmap)

    c = Colorbar(f[5, 1], limits = (minimum(u0), maximum(cc)), colormap = cmap,
        vertical = false, label = "SUVR", labelsize=36, flipaxis=false,
        ticksize=18, ticklabelsize=36, labelpadding=3)

    ax = Axis(f[1,2])
    hidedecorations!(ax)
    hidespines!(ax)
    text!(L"\frac{ds_i}{dt} = - \rho \sum_j^{R} L_{ij} (s_{j} - s_{0, j}) + \alpha (s_i - s_{0, i}) [(s_{\infty,i} - s_{0,i}) - (s_i - s_{0,i})]", 
        fontsize=40, 
        align = (:center, :center))
    #rowsize!(f.layout, 1, Auto(0.15))

    ax = Axis(f[2:5,2],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 36, xticks = LinearTicks(4), xticksize=18,
            xlabel="Time / years", xlabelsize = 36,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 36, yticks = LinearTicks(4), yticksize=18,
            ylabel="SUVR", ylabelsize = 36
    )

    GLMakie.ylims!(ax, minimum(u0)-0.1, maximum(cc)+0.1)
    GLMakie.xlims!(ax, 0, 8)

    for i in 1:36
        lines!(sol.t, sol[i, :], color=Makie.wong_colors()[1])
    end

    x = Observable(0.0)
    vlines!(ax, x, color=(:red, 0.5), linewidth=5)

    sublayout = GridLayout(width = 75)
    f[1,3] = sublayout

    sublayout = GridLayout(height = 50)
    f[6, 1:2] = sublayout
    f
end

frames = 10 * 48

record(f, projectdir("visualisation/models/output/local-fkpp-video.mp4"), 1:frames; framerate=48) do i
    x[] = ts[i]
    for k in 1:36
        p1[k].color[] = cols[i][k]
        p2[k].color[] = cols[i][k]
    end
end