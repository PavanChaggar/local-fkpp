using Connectomes
using ADNIDatasets
using DrWatson 
using CSV, DataFrames
using LinearAlgebra
using GLMakie, ColorSchemes
include(projectdir("functions.jl"))
include(projectdir("adni/inference/inference-preamble.jl"))

L = laplacian_matrix(c)

q = 1 ./ (cc .- u0)

Lw = L * diagm(q)

e = eigen(Lw)
d = abs.(e.vectors[:,1]) ./ maximum(abs.(e.vectors[:,1]))
begin
    GLMakie.activate!()
    right_nodes = get_node_id.(filter(x -> get_hemisphere(x) == "right", cortex))
    left_nodes = get_node_id.(filter(x -> get_hemisphere(x) == "left", cortex))
    cmap = ColorSchemes.viridis
    f = Figure(size=(600,350))
    for (i, nodes) in enumerate([right_nodes, left_nodes])
        ax = Axis3(f[i,1], 
                aspect = :data, 
                azimuth = 0.0pi, 
                elevation=0.0pi,
                protrusions=(0.0,0.0,0.0,0.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, d, cmap)
        
        ax = Axis3(f[i,2], 
                aspect = :data, 
                azimuth = 1.0pi, 
                elevation=0.0pi,
                protrusions=(0.0,0.0,0.0,0.0))
                
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, d, cmap)
    end

    Colorbar(f[1:2, 3], limits = (0, 0.21), colormap = cmap,
    vertical = true, label = "Eigenvalues", labelsize=25, flipaxis=true,
    ticksize=10, ticklabelsize=15, labelpadding=3, ticks=[0,  0.21])
    f
end
save(projectdir("visualisation/models/output/weighted_eigenvector.jpeg"), f)
