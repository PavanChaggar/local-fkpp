using Connectomes
using GLMakie
using ColorSchemes
using Colors
using DrWatson
using CSV, DataFrames
using Distributions
include(projectdir("functions.jl"))
include(projectdir("adni/inference/inference-preamble.jl"))
GLMakie.activate!()

right_cortex = filter(x -> x.Hemisphere == "right", cortex)
right_nodes = get_node_id.(right_cortex)

ubase, upath = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(ubase)
cc = quantile.(upath, .99)

describe(u0)
std(u0)

describe(cc)
std(cc)

scaled_cc = (cc .- minimum(u0)) ./ (maximum(cc) .- minimum(u0))

begin
    GLMakie.activate!()
    cmap = ColorSchemes.RdYlBu |> reverse
    f = Figure(size=(700, 1150),font = "CMU Serif", fontsize=50)

    ax = Axis3(f[1,1], 
               aspect = :data, 
               azimuth = 0.0pi, 
               elevation=0.0pi,
               protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(right_nodes, scaled_cc, cmap)
    
    ax = Axis3(f[2,1], 
               aspect = :data, 
               azimuth = 1.0pi, 
               elevation=0.0pi,
               protrusions=(0.0,0.0,0.0,0.0))
               
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(right_nodes, scaled_cc, cmap)

    Colorbar(f[3, 1], limits = (0.90, maximum(cc)), colormap = cmap,
    vertical = false, label = "SUVR", labelsize=50, flipaxis=false,
    ticksize=20, ticklabelsize=45, labelpadding=3, ticks=1:0.5:3.5)
end
save(projectdir("visualisation/models/output/carrying-capacities-mm-vertical.jpeg"), f, px_per_unit = 5)