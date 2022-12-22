using GLMakie
using Connectomes
using DifferentialEquations
using ColorSchemes
using DrWatson
using Distributions
include(projectdir("adni/adni.jl"))

connectome_path = Connectomes.connectome_path()
all_c = filter(Connectome(connectome_path; norm=true), 1e-2);

subcortex = filter(x -> x.Lobe == "subcortex", all_c.parc)
cortex = filter(x -> x.Lobe != "subcortex", all_c.parc)

c = slice(all_c, cortex) |> filter

cortex.rID = collect(1:72)

right_cortical_nodes = filter(x -> x.Hemisphere == "right", c.parc)

L = laplacian_matrix(c)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for is in c.parc.ID]

gmm_moments = CSV.read(projectdir("data/adni-data/component_moments.csv"), DataFrame)
norm, path = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(norm)
cc  = quantile.(path, .95)

function NetworkExFKPP(du, u, p, t; L = L, u0 = u0, cc = cc)
    du .= -p[1] * L * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

p0 = mean.(norm);
seed = filter(x -> x.Label == "entorhinal", cortex)
seed_value = mean([cc[seed.rID] p0[seed.rID]], dims=2)
p0[seed.rID] .= seed_value 

r = 0.1
a = 1.0
prob = ODEProblem(NetworkExFKPP, p0, (0.0,20.0), [r,a, u0, cc])

ts = collect(0:5:20)
n = length(ts)
sol = solve(prob, Rodas4(), reltol=1e-12, saveat=ts);
allsol = solve(prob, Rodas4(), reltol=1e-12, saveat=0.1)

solcol = [sol[i] ./ maximum(cc) for i in 1:n]
cmap = reverse(ColorSchemes.RdYlBu);
cols = [get(cmap, solcol[i]) for i in 1:n]
nodes = right_cortical_nodes.ID;

begin
    f = Figure(resolution=(2000, 2500))
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
                   protrusions=(1.0,1.0,50.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        for (i, j) in enumerate(nodes)
            plot_roi!(j, cols[k][i])
        end
    end
    # rowsize!(f.layout, 1, Auto(0.8))
    # rowsize!(f.layout, 2, Auto(0.8))
    c = Colorbar(g1[1:2, 1], limits = (0, maximum(cc)), colormap = cmap,
    label = "SUVR", labelsize=36, flipaxis=false,
    ticksize=18, ticks=collect(0.0:0.5:3.5), ticklabelsize=25, labelpadding=3)

    
    ax = Axis(g2[1:3,1:6],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 25, xticks = LinearTicks(4), xticksize=18,
            xlabel="Time / years", xlabelsize = 36,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticks = collect(0:0.5:3.5), yticksize=18,
            ylabel="SUVR", ylabelsize = 36
    )

    GLMakie.ylims!(ax, 0, 3.5)
    GLMakie.xlims!(ax, 0, 20)

    for i in 1:36
        lines!(allsol.t, allsol[i, :], linewidth=2)
    end

    f
end
save(projectdir("adni/visualisation/models/local-fkpp-$(r)-$(a).jpeg"), f)