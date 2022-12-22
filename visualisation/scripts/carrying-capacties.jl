using Connectomes
using GLMakie
using ColorSchemes
using Colors
using DrWatson
using CSV, DataFrames
using Distributions
include(projectdir("adni/adni.jl"))
GLMakie.activate!()

c = Connectomes.connectome_path() |> Connectome

cortex = filter(x -> x.Lobe != "subcortex", c.parc)
left_cortex = filter(x -> x.Hemisphere == "left", cortex)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in left_cortex.ID]
gmm_moments = CSV.read(projectdir("data/adni-data/component_moments.csv"), DataFrame)
norm, path = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(norm)
cc = quantile.(path, .95)

describe(u0)
std(u0)

describe(cc)
std(cc)

begin
    GLMakie.activate!()
    cmap = ColorSchemes.RdYlBu |> reverse
    f = Figure(resolution=(2560,840))

    ax = Axis3(f[1,1], 
               aspect = :data, 
               azimuth = 0.0pi, 
               elevation=0.0pi)
               #protrusions=(0.0,1.0,50.0,1.0))
    hidedecorations!(ax)
    hidespines!(ax)
    for (i, j) in enumerate(left_cortex.ID)
        w =  cc[i] / maximum(cc)
        plot_roi!(j, get(cmap,w))
    end
    ax = Axis3(f[1,2], 
               aspect = :data, 
               azimuth = 1.0pi, 
               elevation=0.0pi)
               #protrusions=(1.0,1.0,50.0,1.0))
               
    hidedecorations!(ax)
    hidespines!(ax)
    for (i, j) in enumerate(left_cortex.ID)
        w =  cc[i] / maximum(cc)
        plot_roi!(j, get(cmap,w))
    end

    # ax = Axis3(f[2,1], 
    #            aspect = :data, 
    #            azimuth = 0.0pi, 
    #            elevation=0.0pi,
    #            protrusions=(1.0,1.0,1.0,1.0))
    # hidedecorations!(ax)
    # hidespines!(ax)
    # for (i, j) in enumerate(42:82)
    #     w =  cc[i] / maximum(cc)
    #     plot_roi!(j, get(cmap,w))
    # end
    # ax = Axis3(f[2,2], 
    #            aspect = :data, 
    #            azimuth = 1.0pi, 
    #            elevation=0.0pi,
    #            protrusions=(1.0,1.0,1.0,1.0))
    # hidedecorations!(ax)
    # hidespines!(ax)
    # for (i, j) in enumerate(42:82)
    #     w =  cc[i] / maximum(cc)
    #     plot_roi!(j, get(cmap,w))
    # end
    

    c = Colorbar(f[1, 0], limits = (0, maximum(cc)), colormap = cmap,
    vertical = true, label = "SUVR", labelsize=25, flipaxis=false,
    ticksize=18, ticklabelsize=20, labelpadding=3)
    f
end

save(projectdir("adni/visualisation/models/carrying-capacities.jpeg"), f)