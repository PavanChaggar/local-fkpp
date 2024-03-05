using Connectomes
using GLMakie
using ColorSchemes
using Colors
using DrWatson
using CSV, DataFrames
using Distributions
include(projectdir("functions.jl"))
GLMakie.activate!()

c = Connectomes.connectome_path() |> Connectome

cortex = filter(x -> x.Lobe != "subcortex", c.parc)
right_cortex = filter(x -> x.Hemisphere == "right", cortex)
right_nodes = get_node_id.(right_cortex)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in get_node_id.(cortex)]
gmm_moments = CSV.read(projectdir("adni/data/component_moments.csv"), DataFrame)
norm, path = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(norm)
cc = quantile.(path, .99)

describe(u0)
std(u0)

describe(cc)
std(cc)

scaled_cc = (cc .- minimum(u0)) ./ (maximum(cc) .- minimum(u0))

begin
    GLMakie.activate!()
    cmap = ColorSchemes.RdYlBu |> reverse
    f = Figure(size=(1000,400))

    ax = Axis3(f[1,1], 
               aspect = :data, 
               azimuth = 0.0pi, 
               elevation=0.0pi)
               #protrusions=(0.0,1.0,50.0,1.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(right_nodes, scaled_cc, cmap)
    
    ax = Axis3(f[1,2], 
               aspect = :data, 
               azimuth = 1.0pi, 
               elevation=0.0pi)
               #protrusions=(1.0,1.0,50.0,1.0))
               
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(right_nodes, scaled_cc, cmap)

    Colorbar(f[1, 0], limits = (minimum(u0), maximum(cc)), colormap = cmap,
    vertical = true, label = "SUVR", labelsize=25, flipaxis=false,
    ticksize=18, ticklabelsize=20, labelpadding=3)
    f
end

save(projectdir("visualisation/models/output/carrying-capacities.jpeg"), f)