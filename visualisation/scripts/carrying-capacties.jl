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

gmm_weights = CSV.read(projectdir("adni/data/component_weights.csv"), DataFrame)
dktweights= filter(x -> x.Column1 ∈ dktnames, gmm_weights)

function get_dkt_weights(weights::DataFrame, dktnames)
    _weights = dropmissing(weights)
    w = Vector{Vector{Float64}}()
    for (i, name) in enumerate(dktnames)
        _df = filter(x -> x.Column1 == name, _weights)
        _w = [_df.Comp_0[1], _df.Comp_1[1]]
        @assert _w[1] > _w[2]
        push!(w, _w)
    end
    w
end

weights = get_dkt_weights(dktweights, dktnames)

ubase, upath = get_dkt_moments(gmm_moments, dktnames)
mm = [MixtureModel([u0, ui], w) for (u0, ui, w) in zip(ubase, upath, weights)]
u0 = mean.(ubase)
cc = quantile.(mm, .99)

# u0 = mean.(norm)
# cc = quantile.(path, .99)

describe(u0)
std(u0)

describe(cc)
std(cc)

scaled_cc = (cc .- minimum(u0)) ./ (maximum(cc) .- minimum(u0))

begin
    GLMakie.activate!()
    cmap = ColorSchemes.RdYlBu |> reverse
    f = Figure(size=(400,800))

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

    Colorbar(f[3, 1], limits = (minimum(u0), maximum(cc)), colormap = cmap,
    vertical = false, label = "SUVR", labelsize=25, flipaxis=false,
    ticksize=18, ticklabelsize=20, labelpadding=3)
    f
end

save(projectdir("visualisation/models/output/carrying-capacities-mm-vertical.jpeg"), f)