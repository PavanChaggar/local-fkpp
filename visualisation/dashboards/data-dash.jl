using Pkg
Pkg.activate("/Users/pavanchaggar/Projects/local-fkpp")
using GLMakie
using Connectomes
using DifferentialEquations
using ColorSchemes
using DrWatson
using Distributions
using Serialization
using Colors
using CSV, DataFrames
using ADNIDatasets
include(projectdir("functions.jl"))

#-------------------------------------------------------------------------------
# Connectome and ROIs
#-------------------------------------------------------------------------------
connectome_path = Connectomes.connectome_path()
c = filter(Connectome(connectome_path; norm=true), 1e-2);

subcortex = filter(x -> x.Lobe == "subcortex", c.parc)
cortex = filter(x -> x.Lobe != "subcortex", c.parc)
rightctx = filter(x -> x.Hemisphere == "right", cortex)

mtl_regions = ["entorhinal", "Left-Amygdala", "Right-Amygdala"]
mtl = findall(x -> x ∈ mtl_regions, cortex.Label)
neo_regions = ["inferiortemporal", "middletemporal"]
neo = findall(x -> x ∈ neo_regions, cortex.Label)
#-------------------------------------------------------------------------------
# Data 
#-------------------------------------------------------------------------------
sub_data_path = projectdir("adni/data/AV1451_Diagnosis-STATUS-STIME-braak-regions.csv")
alldf = CSV.read(sub_data_path, DataFrame)

posdf = filter(x -> x.STATUS == "POS", alldf)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in cortex.ID]

data = ADNIDataset(posdf, dktnames; min_scans=3)
n_data = length(data)

function regional_mean(data, rois, sub)
    subsuvr = calc_suvr(data, sub)
    mean(subsuvr[rois,end])
end

mtl_cutoff = 1.375
neo_cutoff = 1.395

mtl_pos = filter(x -> regional_mean(data, mtl, x) >= mtl_cutoff, 1:n_data)
neo_pos = filter(x -> regional_mean(data, neo, x) >= neo_cutoff, 1:n_data)

tau_pos = findall(x -> x ∈ unique([mtl_pos; neo_pos]), 1:n_data)
tau_neg = findall(x -> x ∉ tau_pos, 1:n_data)

n_pos = length(tau_pos)
n_neg = length(tau_neg)

gmm_moments = CSV.read(projectdir("adni/data/component_moments.csv"), DataFrame)
ubase, upath = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(ubase)
cc = quantile.(upath, .95)

subjects = tau_neg
n_subjects = length(subjects)

data_all = [calc_suvr(data, i) for i in subjects]
[normalise!(data_all[i], u0) for i in 1:n_subjects]
scaleddata = [data_all[i] .- u0 for i in 1:n_subjects]
scaledcc = cc .- u0
alltimes = [get_times(data, i) for i in subjects]

#-------------------------------------------------------------------------------
# Figure
#-------------------------------------------------------------------------------
cmap = reverse(ColorSchemes.Spectral);

p1 = Vector{Mesh}(undef, 36)
p2 = Vector{Mesh}(undef, 36)

f = Figure(resolution=(1200,600))
ax = Axis3(f[1,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi)
hidedecorations!(ax)
hidespines!(ax)
for (i, j) in enumerate(rightctx.ID)
    w =  scaleddata[1][i] / scaledcc[i]
    p1[i] = plot_roi!(j, get(cmap,w))
end
f
ax = Axis3(f[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi)
hidedecorations!(ax)
hidespines!(ax)
for (i, j) in enumerate(rightctx.ID)
    w =  scaleddata[1][i] / scaledcc[i]
    p2[i] = plot_roi!(j, get(cmap,w))
end

Colorbar(f[2, 1:2], limits = (0, 1), colormap = cmap,
    vertical = false, label = "SUVR", labelsize=20, flipaxis=false,
    ticksize=18, ticklabelsize=20, labelpadding=3)
f

sg = SliderGrid(f[3, 1:3],
    (label = "Subject",
    range = 1:1:n_subjects,
    startvalue = 1,
    format = "{:1}"
    ),
    (label = "Time",
    range = 1:1:5,
    startvalue = 1,
    format = "{:1}"
    ),
    (label = "Node",
    range = 1:1:length(rightctx.ID),
    startvalue = 1,
    format = "{:1}"
    )
)

ax = Axis(f[1:2,3],
        xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 1.0,
        xticklabelsize = 20, xticks = LinearTicks(6), xticksize=18,
        xlabel="Time / years", xlabelsize = 20, xminorticksvisible = true,
        xminorgridvisible = true,
        yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 1,
        yticklabelsize = 20, yticks = LinearTicks(6), yticksize=18,
        ylabel="b.c. SUVR", ylabelsize = 20, yminorticksvisible = true,
        yminorgridvisible = true,
)
GLMakie.ylims!(ax, 0.0, 1.0)
GLMakie.xlims!(ax, 0.0, 5.0)
x = Observable(0.0)
vlines!(ax, x, color=(:red, 0.5), linewidth=5)
meancutoff = (mtl_cutoff - mean(u0[mtl]) )/ (mean(cc[mtl])- mean(u0[mtl]))
hlines!(ax, meancutoff)
onany(sg.sliders[1].value, sg.sliders[2].value) do val, t
    for (i, j) in enumerate(rightctx.ID)
        w =  scaleddata[val][i,t] / scaledcc[i]
        p1[i].color[] = get(cmap,w)
        p2[i].color[] = get(cmap,w)
    end
    x[] = alltimes[val][t]
end

point = lift(sg.sliders[1].value, sg.sliders[3].value) do x, y
    Point2f.(alltimes[x], scaleddata[x][rightctx.ID[y],:] / scaledcc[rightctx.ID[y]])
end

title = lift(sg.sliders[3].value) do i
    ax.title[] = c.parc.Label[rightctx.ID[i]]
end

GLMakie.scatter!(point, markersize=10, color=:red)

wait(display(f))