using Connectomes
using ADNIDatasets
using CSV, DataFrames
using DrWatson: projectdir
using Distributions
using Serialization
using DelimitedFiles
using MCMCChains

#-------------------------------------------------------------------------------
# Hierarchical Distributions -- ADNI
#-------------------------------------------------------------------------------
pst = deserialize(projectdir("adni/chains/local-fkpp/pst-taupos-4x2000-vc.jls"));
pst2 = deserialize(projectdir("adni/chains/local-fkpp/pst-tauneg-4x2000-vc.jls"));
pst3 = deserialize(projectdir("adni/chains/local-fkpp/pst-abneg-4x2000-vc.jls"));
prior = deserialize(projectdir("adni/chains/local-fkpp/prior-taupos-8000.jls"));

[p[:numerical_error] |> sum for p in [pst, pst2, pst3]]

using CairoMakie; CairoMakie.activate!()
using Colors

begin
    n_samples = 4000
    f = Figure(resolution=(2000, 750), fontsize=50)
    g1 = f[1, 1] = GridLayout()
    g2 = f[1, 2] = GridLayout()

    colors = alphacolor.(Makie.wong_colors(), 0.75)
    _category_label = [L"A\beta^-", L"A\beta^+ \tau P^-", L"A\beta^+ \tau P^+"]
    
    category_labels = reduce(vcat, fill.(_category_label, n_samples))
    data_array = reduce(vcat, [vec(pst3[:Pm]), vec(pst2[:Pm]), vec(pst[:Pm])])
    
    ax = Axis(g2[1,1], 
            xticklabelsize=30, xlabelsize=30, xlabel="1 / yr", 
            yticklabelsize=40,
            titlesize=40, title="Diffusion", xticks=0.0:0.025:0.075,
            xminorticks=0.0:0.0125:0.075, xminorticksvisible=true, 
            xticksize=20, xminorticksize=15, xgridcolor=RGBAf(0, 0, 0, 0.25))
            xlims!(ax, -0.005, 0.077)
    hideydecorations!(ax)
    hidexdecorations!(ax, grid=false, minorticks=false, label=false, ticks=false, ticklabels=false)
    hidespines!(ax, :t, :r, :l)

    rainclouds!(ax, category_labels, data_array;
                orientation = :horizontal, gap=0.0,
                plot_boxplots = true, cloud_width=0.25,
                clouds=hist, hist_bins=50,
                color = colors[indexin(category_labels, unique(category_labels))])

    category_labels = reduce(vcat, fill.(_category_label, n_samples))    
    data_array = reduce(vcat, [vec(pst3[:Am]), vec(pst2[:Am]), vec(pst[:Am])])
    ax = Axis(g1[1,1], 
            xticklabelsize=30, xlabelsize=30, xlabel="1 / yr", 
            yticklabelsize=40, ylabelsize=30, ylabel="Density",
            titlesize=40, title="Growth", xticks=-0.25:0.125:0.25, 
            xminorticks=-0.25:0.125:0.25, xminorticksvisible=true, 
            xticksize=20, xminorticksize=15, xgridcolor=RGBAf(0, 0, 0, 0.25))
            xlims!(ax, -0.275, 0.275)
        hideydecorations!(ax, label=false, ticklabels=false)
        hidexdecorations!(ax, grid=false, minorticks=false, label=false, ticks=false, ticklabels=false)
    hidespines!(ax, :t, :r, :l)

    rainclouds!(ax, category_labels, data_array;
                orientation = :horizontal, gap=0.0,
                plot_boxplots = true, cloud_width=0.25,
                clouds=hist, hist_bins=50,
                color = colors[indexin(category_labels, unique(category_labels))])

    colgap!(f.layout, 1, 50)
    f
end
save(projectdir("visualisation/inference/posteriors/output/adni-posteriors.pdf"), f)

#-------------------------------------------------------------------------------
# Hierarchical Distributions -- BF2
#-------------------------------------------------------------------------------
pst = deserialize(projectdir("biofinder/chains/local-fkpp/pst-taupos-4x1000-vc.jls"));
pst2 = deserialize(projectdir("biofinder/chains/local-fkpp/pst-tauneg-4x1000-vc.jls"));
pst3 = deserialize(projectdir("biofinder/chains/local-fkpp/pst-abneg-4x1000-vc.jls"));

[p[:numerical_error] |> sum for p in [pst, pst2, pst3]]

using CairoMakie
begin
        n_samples = 4000
        f = Figure(resolution=(2000, 750), fontsize=50)
        g1 = f[1, 1] = GridLayout()
        g2 = f[1, 2] = GridLayout()

        colors = alphacolor.(Makie.wong_colors(), 0.75)
        _category_label = [L"A\beta^-", L"A\beta^+ \tau P^-", L"A\beta^+ \tau P^+"]

        category_labels = reduce(vcat, fill.(_category_label, n_samples))
        data_array = reduce(vcat, [vec(pst3[:Pm]), vec(pst2[:Pm]), vec(pst[:Pm])])

        ax = Axis(g2[1,1], 
                xticklabelsize=30, xlabelsize=30, xlabel="1 / yr", 
                yticklabelsize=40,
                titlesize=40, title="Diffusion", xticks=0.0:0.01:0.05,
                xminorticks=0.0:0.01:0.05, xminorticksvisible=true, 
                xticksize=20, xminorticksize=15, xgridcolor=RGBAf(0, 0, 0, 0.25))
                xlims!(ax, -0.0025, 0.05)
        hideydecorations!(ax)
        hidexdecorations!(ax, grid=false, minorticks=false, label=false, ticks=false, ticklabels=false)
        hidespines!(ax, :t, :r, :l)

        rainclouds!(ax, category_labels, data_array;
                        orientation = :horizontal, gap=-1.5,
                        plot_boxplots = true, cloud_width=0.5,
                        color = colors[indexin(category_labels, unique(category_labels))])

        category_labels = reduce(vcat, fill.(_category_label, n_samples))    
        data_array = reduce(vcat, [vec(pst3[:Am]), vec(pst2[:Am]), vec(pst[:Am])])
        ax = Axis(g1[1,1], 
                xticklabelsize=30, xlabelsize=30, xlabel="1 / yr", 
                yticklabelsize=40, ylabelsize=30, ylabel="Density",
                titlesize=40, title="Growth", xticks=-0.2:0.05:0.2, 
                xminorticks=-0.15:0.05:0.15, xminorticksvisible=true, 
                xticksize=20, xminorticksize=15, xgridcolor=RGBAf(0, 0, 0, 0.25), 
                xtickformat = "{:.2f}")
                xlims!(ax, -0.17, 0.170)
                hideydecorations!(ax, label=false, ticklabels=false)
                hidexdecorations!(ax, grid=false, minorticks=false, label=false, 
                                  ticks=false, ticklabels=false)
        hidespines!(ax, :t, :r, :l)

        rainclouds!(ax, category_labels, data_array;
        orientation = :horizontal, gap=-1.5,
        plot_boxplots = true, cloud_width=0.5,
        color = colors[indexin(category_labels, unique(category_labels))])

        colgap!(f.layout, 1, 50)
        f
end 
save(projectdir("visualisation/inference/posteriors/output/bf-posteriors.pdf"), f)

# shuffled data
shuffled = [deserialize(projectdir("adni/chains/local-fkpp/shuffled/pst-tauneg-1000-shuffled-$i.jls")) for i in 1:5]

f = Figure(resolution=(1000, 500))
ax = Axis(f[1,1], title="growth")
am = reduce(vcat, [vec(Array(sh[:Am])) for sh in shuffled])
hist!(am, bins=50, normalization=:pdf, color=:grey)
hist!(pst[:Am] |> vec, bins=50, normalization=:pdf, color=(:blue, 0.5))

ax = Axis(f[1,2], title="diffusion")
pm = reduce(vcat, [vec(Array(sh[:Pm])) for sh in shuffled])
hist!(pm, bins=50, normalization=:pdf, color=:grey, label="shuffled")
hist!(pst[:Pm] |> vec, bins=50, normalization=:pdf, color=(:blue, 0.5), label="A+T+")
# hist!(pst3[:Pm] |> vec, bins=50, normalization=:pdf, color=(:green, 0.5), label="A-")
axislegend()
f
