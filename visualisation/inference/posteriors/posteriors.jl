using Connectomes
using ADNIDatasets
using CSV, DataFrames
using DrWatson: projectdir
using Distributions
using Serialization
using DelimitedFiles
using MCMCChains
using Turing
using CairoMakie; CairoMakie.activate!()
using Colors
using HypothesisTests
#-------------------------------------------------------------------------------
# Hierarchical Distributions -- ADNI
#-------------------------------------------------------------------------------
pst = deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-taupos-4x2000.jls"));
pst2 = deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-tauneg-4x2000.jls"));
pst3 = deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-abneg-4x2000.jls"));

[p[:numerical_error] |> sum for p in [pst, pst2, pst3]]

function param_test(p1, p2)
        t = ApproximateTwoSampleKSTest(vec(p1), vec(p2))
        if pvalue(t) < 0.01
                return true
        else
                return false
        end
end 

for p in [:Am, :Pm]
        param_test(pst[p], pst2[p]) |> println
        param_test(pst[p], pst3[p]) |> println
        param_test(pst2[p], pst3[p]) |> println
end

begin
        n_samples = 8000
        f = Figure(size=(2000, 2000), fontsize=50, font=:bold)
        g1 = f[1, 1] = GridLayout()
        g2 = f[2, 1] = GridLayout()
        g3 = f[3, 1] = GridLayout()
        g4 = f[4, 1] = GridLayout()
        Label(g1[1,1:3], "Transport", fontsize=50, tellwidth=false, tellheight=true)
        Label(g3[1,1:3], "Production", fontsize=50, tellwidth=false)
        colors = reverse(alphacolor.(Makie.wong_colors(), 0.75)[1:3])
        _category_label = reverse([L"A^-", L"A^+T^-", L"A^+T^+"])
        psts = [pst, pst2, pst3]
        for (i, (_label, pst, col)) in enumerate(zip(_category_label, psts, colors))
                category_labels = fill(_label, n_samples)
                transport = vec(pst[:Pm])
                production = vec(pst[:Am])

                Label(g2[i, 0], _label, fontsize=45, tellheight=false)

                ax = Axis(g2[i,1], 
                        xticklabelsize=30, xlabelsize=30, xlabel="1 / yr", 
                        yticklabelsize=40,
                        titlesize=40, xticks=0.0:0.02:0.08,
                        xminorticks=0.0:0.01:1, xminorticksvisible=true, 
                        xticksize=20, xminorticksize=15, xgridcolor=RGBAf(0, 0, 0, 0.25))
                        CairoMakie.xlims!(ax, -0.004, 0.084)
                hideydecorations!(ax)
                if i < 3
                        hidexdecorations!(ax, grid=false, minorticks=false, ticks=false)
                else
                        hidexdecorations!(ax, grid=false, minorticks=false, label=false, ticks=false, ticklabels=false)
                end
                hidespines!(ax, :t, :r, :l)

                rainclouds!(ax, category_labels, transport;
                                orientation = :horizontal, gap=0.0,
                                plot_boxplots = true, cloud_width=0.5,
                                clouds=hist, hist_bins=100,
                                color = col)
                
                ax = Axis(g2[i,2:3], 
                xticklabelsize=30, xlabelsize=30, xlabel="1 / yr", 
                yticklabelsize=40,
                titlesize=40, xticks=0.0:0.025:0.125,
                xminorticks=0.0:0.0125:1, xminorticksvisible=true, 
                xticksize=20, xminorticksize=15, xgridcolor=RGBAf(0, 0, 0, 0.25))
                CairoMakie.xlims!(ax, -0.005, 0.105)
                hideydecorations!(ax)
                if i < 3
                        hidexdecorations!(ax, grid=false, minorticks=false, ticks=false)
                else
                        hidexdecorations!(ax, grid=false, minorticks=false, label=false, ticks=false, ticklabels=false)
                end
                hidespines!(ax, :t, :r, :l)

                n_params = sum(contains.(string.(names(MCMCChains.get_sections(pst, :parameters))), "ρ"))
                for j in 1:n_params
                        CairoMakie.CairoMakie.density!(vec(Array(pst[Symbol("ρ[$j]")])), color=alphacolor(col, 0.5), strokecolor=:white, strokewidth=1)
                end

                Label(g4[i, 0], _label, fontsize=45, tellheight=false)
                ax = Axis(g4[i,1], 
                xticklabelsize=30, xlabelsize=30, xlabel="1 / yr", 
                yticklabelsize=40, ylabelsize=30, ylabel="Density", xticks=-0.20:0.1:0.20, 
                xminorticks=-0.2:0.05:0.2, xminorticksvisible=true, 
                xticksize=20, xminorticksize=15, xgridcolor=RGBAf(0, 0, 0, 0.25))
                CairoMakie.xlims!(ax, -0.22, 0.22)
                hideydecorations!(ax)
                if i < 3
                        hidexdecorations!(ax, grid=false, minorticks=false, ticks=false)
                else
                        hidexdecorations!(ax, grid=false, minorticks=false, label=false, ticks=false, ticklabels=false)
                end
                hidespines!(ax, :t, :r, :l)

                rainclouds!(ax, category_labels, production;
                                orientation = :horizontal, gap=0.0,
                                plot_boxplots = true, cloud_width=0.5,
                                clouds=hist, hist_bins=100,
                                color = col)

                ax = Axis(g4[i,2:3], 
                xticklabelsize=30, xlabelsize=30, xlabel="1 / yr", 
                yticklabelsize=40,
                titlesize=40, xticks=-1.:0.5:1.,
                xminorticks=-1.:0.25:1, xminorticksvisible=true, 
                xticksize=20, xminorticksize=15, xgridcolor=RGBAf(0, 0, 0, 0.25))
                CairoMakie.xlims!(ax, -1.1, 1.1)
                hideydecorations!(ax)
                if i < 3
                        hidexdecorations!(ax, grid=false, minorticks=false, ticks=false)
                else
                        hidexdecorations!(ax, grid=false, minorticks=false, label=false, ticks=false, ticklabels=false)
                end
                hidespines!(ax, :t, :r, :l)
                n_params = sum(contains.(string.(names(MCMCChains.get_sections(pst, :parameters))), "α"))
                for j in 1:n_params
                        CairoMakie.CairoMakie.density!(vec(Array(pst[Symbol("α[$j]")])), color=alphacolor(col, 0.5), strokecolor=:white, strokewidth=1)
                end
        end
        colgap!(g2, 2, 50.0)
        colgap!(g4, 2, 50.0)
        f
end
save(projectdir("visualisation/inference/posteriors/output/adni-posteriors-all.pdf"), f)

begin
        n_samples = 8000
        f = Figure(size=(2000, 750), fontsize=50)
        g1 = f[1, 1] = GridLayout()
        g2 = f[1, 2] = GridLayout()

        colors = alphacolor.(Makie.wong_colors(), 0.75)

        a3, a2, a1 = vec(pst3[:Am]), vec(pst2[:Am]), vec(pst[:Am])
        ax = Axis(g1[1,1], 
                xticklabelsize=30, xlabelsize=30, xlabel="1 / yr", 
                yticklabelsize=40, ylabelsize=30, ylabel="Density", yticks=(1.5:1:3.5, [L"A^-", L"A^+T^-", L"A^+T^+"]),
                titlesize=40, title="Production", xticks=-0.2:0.1:0.2, 
                xminorticks=-0.2:0.05:0.2, xminorticksvisible=true, 
                xticksize=20, xminorticksize=15, xgridcolor=RGBAf(0, 0, 0, 0.25))
        CairoMakie.xlims!(ax, -0.225, 0.225)
        CairoMakie.ylims!(ax, 0.75, 4)
        hideydecorations!(ax, label=false, ticklabels=false)
        hidexdecorations!(ax, grid=false, minorticks=false, label=false, ticks=false, ticklabels=false)
        hidespines!(ax, :t, :r, :l)

        rainclouds!(ax, 1.5 .* ones(n_samples), a3; orientation = :horizontal, gap=0.0, plot_boxplots = true, 
                                                cloud_width=0.5, clouds=hist, hist_bins=50,
                                                color = colors[1])
        rainclouds!(ax, 2.5 .* ones(n_samples), a2; orientation = :horizontal, gap=0.0, plot_boxplots = true, 
                                                cloud_width=0.5, clouds=hist, hist_bins=50,
                                                color = colors[2])
        rainclouds!(ax, 3.5 .* ones(n_samples), a1; orientation = :horizontal, gap=0.0, plot_boxplots = true, 
                                                cloud_width=0.5, clouds=hist, hist_bins=25,
                                                color = colors[3])

        bracket!(ax, median(a1), 1.25, median(a2), 1.25, text = "*", orientation = :down, textoffset = 20, linewidth = 2, color=(:grey, 0.75))
        bracket!(ax, median(a2), 1.25, median(a3), 1.25, text = "*", orientation = :down, textoffset = 20, linewidth = 2, color=(:grey, 0.75))
        bracket!(ax, median(a3), 1.05, median(a1), 1.05, text = "*", orientation = :down, textoffset = 20, linewidth = 2, color=(:grey, 0.75))

        p3, p2, p1 = vec(pst3[:Pm]), vec(pst2[:Pm]), vec(pst[:Pm])

        ax = Axis(g2[1,1], 
                xticklabelsize=30, xlabelsize=30, xlabel="1 / yr", yticks=(1.5:1:3.5, [L"A^-", L"A^+T^-", L"A^+T^+"]),
                yticklabelsize=40,
                titlesize=40, title="Transport", xticks=0.0:0.025:0.075,
                xminorticks=0.0:0.0125:1, xminorticksvisible=true, 
                xticksize=20, xminorticksize=15, xgridcolor=RGBAf(0, 0, 0, 0.25))
        CairoMakie.xlims!(ax, -0.005, 0.085)
        CairoMakie.ylims!(ax, 0.75, 4)
        hideydecorations!(ax)
        hidexdecorations!(ax, grid=false, minorticks=false, label=false, ticks=false, ticklabels=false)
        hidespines!(ax, :t, :r, :l)


        rainclouds!(ax, 1.5 .* ones(n_samples), p3; orientation = :horizontal, gap=0.0, plot_boxplots = true, 
                                                cloud_width=0.5, clouds=hist, hist_bins=50,
                                                color = colors[1])
        rainclouds!(ax, 2.5 .* ones(n_samples), p2; orientation = :horizontal, gap=0.0, plot_boxplots = true, 
                                                cloud_width=0.5, clouds=hist, hist_bins=50,
                                                color = colors[2])
        rainclouds!(ax, 3.5 .* ones(n_samples), p1; orientation = :horizontal, gap=0.0, plot_boxplots = true, 
                                         cloud_width=0.5, clouds=hist, hist_bins=25,
                                         color = colors[3])                                         
        bracket!(ax, median(p1), 1.25, median(p2), 1.25, text = "*", orientation = :down, textoffset = 20, linewidth = 2, color=(:grey, 0.75))
        bracket!(ax, median(p2), 1.25, median(p3), 1.25, text = "*", orientation = :down, textoffset = 20, linewidth = 2, color=(:grey, 0.75))
        bracket!(ax, median(p3), 1.05, median(p1), 1.05, text = "*", orientation = :down, textoffset = 20, linewidth = 2, color=(:grey, 0.5))
        colgap!(f.layout, 1, 50)
        f
end
save(projectdir("visualisation/inference/posteriors/output/adni-posteriors-with-sig.pdf"), f)

#-------------------------------------------------------------------------------
# Hierarchical Distributions -- BF2
#-------------------------------------------------------------------------------
pst = deserialize(projectdir("biofinder/chains/local-fkpp/pst-taupos-4x1000-vc.jls"));
pst2 = deserialize(projectdir("biofinder/chains/local-fkpp/pst-tauneg-4x1000-vc.jls"));
pst3 = deserialize(projectdir("biofinder/chains/local-fkpp/pst-abneg-4x1000-vc.jls"));

[p[:numerical_error] |> sum for p in [pst, pst2, pst3]]

for p in [:Am, :Pm]
        param_test(pst[p], pst2[p]) |> println
        param_test(pst[p], pst3[p]) |> println
        param_test(pst2[p], pst3[p]) |> println
end


using CairoMakie

begin
        n_samples = 4000
        f = Figure(size=(2000, 2000), fontsize=50, font=:bold)
        g1 = f[1, 1] = GridLayout()
        g2 = f[2, 1] = GridLayout()
        g3 = f[3, 1] = GridLayout()
        g4 = f[4, 1] = GridLayout()
        
        Label(g1[1,1:3], "Transport", fontsize=45, tellwidth=false, tellheight=true)
        Label(g3[1,1:3], "Production", fontsize=45, tellwidth=false)
        colors = reverse(alphacolor.(Makie.wong_colors(), 0.75)[1:3])
        _category_label = reverse([L"A^-", L"A^+T^-", L"A^+T^+"])
        psts = [pst, pst2, pst3]
        for (i, (_label, pst, col)) in enumerate(zip(_category_label, psts, colors))
                category_labels = fill(_label, n_samples)
                transport = vec(pst[:Pm])
                production = vec(pst[:Am])
                ax = Axis(g2[i,1], 
                        xticklabelsize=30, xlabelsize=30, xlabel="1 / yr", 
                        yticklabelsize=40,
                        titlesize=40, xticks=0.0:0.01:0.05,
                        xminorticks=0.0:0.005:0.05, xminorticksvisible=true, 
                        xticksize=20, xminorticksize=15, xgridcolor=RGBAf(0, 0, 0, 0.25))
                        CairoMakie.xlims!(ax, -0.0025, 0.0525)
                hideydecorations!(ax)
                if i < 3
                        hidexdecorations!(ax, grid=false, minorticks=false, ticks=false)
                else
                        hidexdecorations!(ax, grid=false, minorticks=false, label=false, ticks=false, ticklabels=false)
                end
                hidespines!(ax, :t, :r, :l)
                Label(g2[i, 0], _label, fontsize=40, tellheight=false)
                rainclouds!(ax, category_labels, transport;
                                orientation = :horizontal, gap=0.0,
                                plot_boxplots = true, cloud_width=0.5,
                                clouds=hist, hist_bins=100,
                                color = col)
                
                ax = Axis(g2[i,2:3], 
                xticklabelsize=30, xlabelsize=30, xlabel="1 / yr", 
                yticklabelsize=40,
                titlesize=40, xticks=0.0:0.02:0.08,
                xminorticks=0.0:0.01:1, xminorticksvisible=true, 
                xticksize=20, xminorticksize=15, xgridcolor=RGBAf(0, 0, 0, 0.25))
                CairoMakie.xlims!(ax, -0.004, 0.084)
                hideydecorations!(ax)
                if i < 3
                        hidexdecorations!(ax, grid=false, minorticks=false, ticks=false)
                else
                        hidexdecorations!(ax, grid=false, minorticks=false, label=false, ticks=false, ticklabels=false)
                end
                hidespines!(ax, :t, :r, :l)

                n_params = sum(contains.(string.(names(MCMCChains.get_sections(pst, :parameters))), "ρ"))
                for j in 1:n_params
                        CairoMakie.density!(vec(Array(pst[Symbol("ρ[$j]")])), color=alphacolor(col, 0.5), strokecolor=:white, strokewidth=1)
                end

                ax = Axis(g4[i,1], 
                xticklabelsize=30, xlabelsize=30, xlabel="1 / yr", 
                yticklabelsize=40, ylabelsize=30, ylabel="Density", xticks=-0.2:0.1:0.2, 
                xminorticks=-0.2:0.05:0.2, xminorticksvisible=true, 
                xticksize=20, xminorticksize=15, xgridcolor=RGBAf(0, 0, 0, 0.25))
                CairoMakie.xlims!(ax, -0.22, 0.22)
                hideydecorations!(ax)
                if i < 3
                        hidexdecorations!(ax, grid=false, minorticks=false, ticks=false)
                else
                        hidexdecorations!(ax, grid=false, minorticks=false, label=false, ticks=false, ticklabels=false)
                end
                hidespines!(ax, :t, :r, :l)
                Label(g4[i, 0], _label, fontsize=40, tellheight=false)
                rainclouds!(ax, category_labels, production;
                                orientation = :horizontal, gap=0.0,
                                plot_boxplots = true, cloud_width=0.5,
                                clouds=hist, hist_bins=100,
                                color = col)
                ax = Axis(g4[i,2:3], 
                xticklabelsize=30, xlabelsize=30, xlabel="1 / yr", 
                yticklabelsize=40,
                titlesize=40, xticks=-0.8:0.4:0.8,
                xminorticks=-1.:0.2:1, xminorticksvisible=true, 
                xticksize=20, xminorticksize=15, xgridcolor=RGBAf(0, 0, 0, 0.25))
                CairoMakie.xlims!(ax, -0.88, 0.88)
                hideydecorations!(ax)
                if i < 3
                        hidexdecorations!(ax, grid=false, minorticks=false, ticks=false)
                else
                        hidexdecorations!(ax, grid=false, minorticks=false, label=false, ticks=false, ticklabels=false)
                end
                hidespines!(ax, :t, :r, :l)

                n_params = sum(contains.(string.(names(MCMCChains.get_sections(pst, :parameters))), "α"))
                for j in 1:n_params
                        CairoMakie.density!(vec(Array(pst[Symbol("α[$j]")])), color=alphacolor(col, 0.5), strokecolor=:white, strokewidth=1)
                end
        end
        colgap!(g2, 2, 50.0)
        colgap!(g4, 2, 50.0)
        f
end
save(projectdir("visualisation/inference/posteriors/output/bf-posteriors-all.pdf"), f)


begin
        n_samples = 4000
        f = Figure(size=(2000, 750), fontsize=50)
        g1 = f[1, 1] = GridLayout()
        g2 = f[1, 2] = GridLayout()

        colors = alphacolor.(Makie.wong_colors(), 0.75)

        p3, p2, p1 = vec(pst3[:Pm]), vec(pst2[:Pm]), vec(pst[:Pm])
        ax = Axis(g2[1,1], 
                xticklabelsize=30, xlabelsize=30, xlabel="1 / yr", 
                yticklabelsize=40,
                titlesize=40, title="Transport", xticks=0.0:0.01:0.05,
                xminorticks=0.0:0.005:0.05, xminorticksvisible=true, 
                xticksize=20, xminorticksize=15, xgridcolor=RGBAf(0, 0, 0, 0.25))
                CairoMakie.xlims!(ax, -0.0025, 0.06)
                CairoMakie.ylims!(ax, 0.75, 4)

        hideydecorations!(ax, label=false, ticklabels=false)
        hidexdecorations!(ax, grid=false, minorticks=false, label=false, ticks=false, ticklabels=false)
        hidespines!(ax, :t, :r, :l)

        rainclouds!(ax, 1.5 .* ones(n_samples), p3; orientation = :horizontal, gap=0.0, plot_boxplots = true, 
                                                cloud_width=0.5, clouds=hist, hist_bins=50,
                                                color = colors[1])
        rainclouds!(ax, 2.5 .* ones(n_samples), p2; orientation = :horizontal, gap=0.0, plot_boxplots = true, 
                                                cloud_width=0.5, clouds=hist, hist_bins=50,
                                                color = colors[2])
        rainclouds!(ax, 3.5 .* ones(n_samples), p1; orientation = :horizontal, gap=0.0, plot_boxplots = true, 
                                                cloud_width=0.5, clouds=hist, hist_bins=25,
                                                color = colors[3])                                         
        bracket!(ax, median(p1), 1.25, median(p3), 1.25, text = "*", orientation = :down, textoffset = 20, linewidth = 2, color=(:grey, 0.75))
        bracket!(ax, median(p2), 1.25, median(p3), 1.25, text = "*", orientation = :down, textoffset = 20, linewidth = 2, color=(:grey, 0.75))
        bracket!(ax, median(p2), 1.05, median(p1), 1.05, text = "*", orientation = :down, textoffset = 20, linewidth = 2, color=(:grey, 0.75))
        
        a3, a2, a1 = vec(pst3[:Am]), vec(pst2[:Am]), vec(pst[:Am])

        ax = Axis(g1[1,1], 
                xticklabelsize=30, xlabelsize=30, xlabel="1 / yr", 
                yticklabelsize=40, ylabelsize=30, ylabel="Density",
                titlesize=40, title="Production", xticks=-0.2:0.1:0.2, 
                xminorticks=-0.2:0.05:0.2, xminorticksvisible=true, 
                xticksize=20, xminorticksize=15, xgridcolor=RGBAf(0, 0, 0, 0.25), 
                xtickformat = "{:.2f}")
        CairoMakie.xlims!(ax, -0.26, 0.26)
        CairoMakie.ylims!(ax, 0.75, 4)
        hideydecorations!(ax, label=false, ticklabels=false)
        hidexdecorations!(ax, grid=false, minorticks=false, label=false, 
                                ticks=false, ticklabels=false)
        hidespines!(ax, :t, :r, :l)

        rainclouds!(ax, 1.5 .* ones(n_samples), a3; orientation = :horizontal, gap=0.0, plot_boxplots = true, 
                                                cloud_width=0.5, clouds=hist, hist_bins=50,
                                                color = colors[1])
        rainclouds!(ax, 2.5 .* ones(n_samples), a2; orientation = :horizontal, gap=0.0, plot_boxplots = true, 
                                                cloud_width=0.5, clouds=hist, hist_bins=50,
                                                color = colors[2])
        rainclouds!(ax, 3.5 .* ones(n_samples), a1; orientation = :horizontal, gap=0.0, plot_boxplots = true, 
                                                cloud_width=0.5, clouds=hist, hist_bins=25,
                                                color = colors[3])                                         
        bracket!(ax, median(a1), 1.25, median(a2), 1.25, text = "*", orientation = :down, textoffset = 20, linewidth = 2, color=(:grey, 0.75))
        bracket!(ax, median(a2), 1.25, median(a3), 1.25, text = "*", orientation = :down, textoffset = 20, linewidth = 2, color=(:grey, 0.75))
        bracket!(ax, median(a3), 1.05, median(a1), 1.05, text = "*", orientation = :down, textoffset = 20, linewidth = 2, color=(:grey, 0.75))

        colgap!(f.layout, 1, 50)

        colgap!(f.layout, 1, 50)
        f
end
save(projectdir("visualisation/inference/posteriors/output/bf-posteriors-with-sig.pdf"), f)

begin
        colors = alphacolor.(Makie.wong_colors(), 0.75)

        f = Figure(size=(2000,1000));
        g1 = f[1, 2] = GridLayout()
        g2= f[1, 1] = GridLayout()
        g7 = f[1, 3] = GridLayout()

        ax11 = Axis(g1[1, 1], title="Growth", titlesize=40, xticks=-3:1:2);
        ax12 = Axis(g1[2, 1], xticks=-3:1.0:2);
        ax13 = Axis(g1[3, 1], xticks=-3:1.0:2, xlabel="1 / yr", xlabelsize=30);
        linkaxes!(ax13, ax11)
        linkaxes!(ax13, ax12)

        hideydecorations!(ax11, label=false)
        hidexdecorations!(ax11, label=false, ticks=false, grid=false)
        hidespines!(ax11, :t, :r, :l)
        for i in 1:54
                hist!(ax11, vec(pst[Symbol("α[$i]")]), bins=50, color=(colors[3], 0.6), normalization=:pdf, label=L"A\beta^+ \tau^+")
        end


        hideydecorations!(ax12, label=false)
        hidexdecorations!(ax12, label=false, ticks=false, grid=false)
        hidespines!(ax12, :t, :r, :l)
        for i in 1:18
                hist!(ax12, vec(pst2[Symbol("α[$i]")]), bins=50, color=(colors[2], 0.6), normalization=:pdf, label=L"A\beta^+ \tau^-")
        end


        hideydecorations!(ax13, label=false)
        hidexdecorations!(ax13, label=false, ticks=false, ticklabels=false, grid=false)
        hidespines!(ax13, :t, :r, :l)
        CairoMakie.xlims!(-1.5, 1.)
        for i in 1:53
                hist!(ax13, vec(pst3[Symbol("α[$i]")]), bins=50, color=(colors[1], 0.6), normalization=:pdf, label=L"A\beta^-")
        end

        ax21 = Axis(g2[1, 1], title="Diffusion", titlesize=40, ylabel="Density", ylabelsize=30, xticks=-0:0.05:0.2);
        ax22 = Axis(g2[2, 1], ylabel="Density", ylabelsize=30, xticks=-0:0.05:0.2);
        ax23 = Axis(g2[3, 1], ylabel="Density", ylabelsize=30, xlabel="mm² / yr", xlabelsize=30, xticks=-0:0.05:0.2);
        linkaxes!(ax23, ax22)
        linkaxes!(ax23, ax21)

        hideydecorations!(ax21, label=false)
        hidexdecorations!(ax21, label=false, ticks=false, grid=false)
        hidespines!(ax21, :t, :r, :l)
        for i in 1:54
                hist!(ax21, vec(pst[Symbol("ρ[$i]")]), bins=50, color=(colors[3], 0.6), normalization=:pdf, label=L"A\beta^+ \tau^+")
        end

        hideydecorations!(ax22, label=false)
        hidexdecorations!(ax22, label=false, ticks=false, grid=false)
        hidespines!(ax22, :t, :r, :l)
        for i in 1:18
                hist!(ax22, vec(pst2[Symbol("ρ[$i]")]), bins=50, color=(colors[2], 0.6), normalization=:pdf, label=L"A\beta^+ \tau^-")
        end

        hideydecorations!(ax23, label=false)
        hidexdecorations!(ax23, label=false, ticks=false, ticklabels=false, grid=false)
        hidespines!(ax23, :t, :r, :l)
        CairoMakie.xlims!(-0.001, 0.15)
        for i in 1:53
                hist!(ax23, vec(pst3[Symbol("ρ[$i]")]), bins=50, color=(colors[1], 0.6), normalization=:pdf, label=L"A\beta^-")
        end

        elem_1 = PolyElement(color = (colors[3], 0.6))
        elem_2 = PolyElement(color = (colors[2], 0.6))
        elem_3 = PolyElement(color = (colors[1], 0.6))
        legend = Legend(g7[1,1],
                [elem_1, elem_2, elem_3],
                [L"A\beta^+ \tau P^+", L"A\beta^+ \tau P^-", L"A\beta^-"],
                patchsize = (35, 35), rowgap = 10, framevisible=false, labelsize=30)
        colgap!(f.layout, 1, 50)
        f
end

#-------------------------------------------------------------------------------
# Hierarchical Distributions -- shuffled
#-------------------------------------------------------------------------------
pst = deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-taupos-4x2000.jls"));
pst2 = deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-tauneg-4x2000.jls"));
pst3 = deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-abneg-4x2000.jls"));

_pos_shuffled = [deserialize(projectdir("adni/new-chains/local-fkpp/shuffled/pos/length-free/pst-taupos-1000-shuffled-$i.jls")) for i in 1:10];
sh_idx = findall( x -> sum(x[:numerical_error]) == 0, _pos_shuffled)
pos_shuffled = chainscat(_pos_shuffled[sh_idx]...);
begin
        colors = Makie.wong_colors()

        f = Figure(size=(1000, 500), fontsize=35)
        g1 = f[1, 1] = GridLayout()

        ax = Axis(g1[1,1], title="Production", titlesize=30, xticklabelsize=30,
        xlabel="1 / yr", xlabelsize=30, ylabel="Density", ylabelsize=30)
        CairoMakie.xlims!(ax, 0., 0.22)
        CairoMakie.ylims!(ax, -4.125, 33)
        hideydecorations!(ax, label=false)
        hidespines!(ax, :t, :r, :l)

        hist!(pos_shuffled[:Am] |> vec, bins=50, normalization=:pdf, color=(:black, 0.8))
        hist!(pst[:Am] |> vec, bins=50, normalization=:pdf, color=(colors[3], 0.75))
        bracket!(ax, median(pos_shuffled[:Am]), -0.4125, median(pst[:Am]), -0.4125, text = "*", orientation = :down, fontsize=50, textoffset = 20, linewidth = 3, color=(:grey, 0.75))

        ax = Axis(g1[1,2], title="Transport", titlesize=30, xticklabelsize=30,
        xlabel="1 / yr", xlabelsize=30, ylabel="Density", ylabelsize=30)
        CairoMakie.ylims!(ax, -50., 400)
        hideydecorations!(ax)
        hidespines!(ax, :t, :r, :l)

        hist!(pos_shuffled[:Pm] |> vec, bins=25, normalization=:pdf, color=(:black, 0.8), label="Shuffled")
        hist!(pst[:Pm] |> vec, bins=50, normalization=:pdf, color=(colors[3], 0.75), label=L"A\beta^+ \tau P^+")
        bracket!(ax, median(pos_shuffled[:Pm]), -5, median(pst[:Pm]), -5, text = "*", orientation = :down, fontsize=50, textoffset = 20, linewidth = 3, color=(:grey, 0.75))
        axislegend()

        colgap!(g1, 50)
        f
end     
save(projectdir("visualisation/inference/posteriors/output/adni-pos-shuffled-with-sig.pdf"), f)


_neg_shuffled = [deserialize(projectdir("adni/new-chains/local-fkpp/shuffled/neg/length-free/pst-tauneg-1000-shuffled-$i.jls")) for i in 1:10];
sh_idx = findall( x -> sum(x[:numerical_error]) == 0, _neg_shuffled)
neg_shuffled = chainscat(_neg_shuffled[sh_idx]...);
begin
        colors = Makie.wong_colors()

        f = Figure(size=(1000, 500), fontsize=35)
        g1 = f[1, 1] = GridLayout()

        ax = Axis(g1[1,1], title="Production", titlesize=30, xticklabelsize=30,
        xlabel="1 / yr", xlabelsize=30, ylabel="Density", ylabelsize=30)
        CairoMakie.ylims!(ax, -11. .* 0.125, 11.)
        hideydecorations!(ax, label=false)
        hidespines!(ax, :t, :r, :l)

        hist!(neg_shuffled[:Am] |> vec, bins=50, normalization=:pdf, color=(:black, 0.8))
        hist!(pst2[:Am] |> vec, bins=50, normalization=:pdf, color=(colors[2], 0.75))
        bracket!(ax, median(neg_shuffled[:Am]), -0.11, median(pst2[:Am]), -0.11, text = "*", orientation = :down, fontsize=50, textoffset = 20, linewidth = 3, color=(:grey, 0.75))

        ax = Axis(g1[1,2], title="Transport", titlesize=30, xticklabelsize=30,
        xlabel="1 / yr", xlabelsize=30, ylabel="Density", ylabelsize=30)
        CairoMakie.ylims!(ax, -100 .* 0.125, 100)
        hideydecorations!(ax)
        hidespines!(ax, :t, :r, :l)
        
        hist!(neg_shuffled[:Pm] |> vec, bins=50, normalization=:pdf, color=(:black, 0.8), label="Shuffled")
        hist!(pst2[:Pm] |> vec, bins=50, normalization=:pdf, color=(colors[2], 0.75), label=L"A\beta^+ \tau P^-")
        bracket!(ax, median(neg_shuffled[:Pm]), -1.0, median(pst2[:Pm]), -1., text = "*", orientation = :down, fontsize=50, textoffset = 20, linewidth = 3, color=(:grey, 0.75))
        axislegend()
        colgap!(g1, 50)

        f
end     
save(projectdir("visualisation/inference/posteriors/output/adni-neg-shuffled-with-sig.pdf"), f)

# ab_shuffled = deserialize(projectdir("adni/chains/local-fkpp/shuffled/ab/pst-abneg-1000-shuffled-1.jls"))

# begin
#         colors = Makie.wong_colors()

#         f = Figure(size=(1000, 500), fontsize=35)
#         ax = Axis(f[1,1], title="Production", titlesize=30, xticklabelsize=30,
#         xlabel="1 / yr", xlabelsize=30, ylabel="Density", ylabelsize=30)
#         hideydecorations!(ax, label=false)
#         hidespines!(ax, :t, :r, :l)

#         am = vec(Array(ab_shuffled[:Am]))
#         hist!(am, bins=50, normalization=:pdf, color=:grey)
#         hist!(pst3[:Am] |> vec, bins=50, normalization=:pdf, color=(colors[2], 0.75))

#         ax = Axis(f[1,2], title="Transport", titlesize=30, xticklabelsize=30,
#         xlabel="1 / yr", xlabelsize=30, ylabel="Density", ylabelsize=30)
#         hideydecorations!(ax)
#         hidespines!(ax, :t, :r, :l)

#         pm = vec(Array(ab_shuffled[:Pm]))
#         hist!(pm, bins=25, normalization=:pdf, color=:grey, label="Shuffled")
#         hist!(pst3[:Pm] |> vec, bins=50, normalization=:pdf, color=(colors[2], 0.75), label=L"A\beta^+ \tau P^-")
#         axislegend()
#         f
# end     

#-------------------------------------------------------------------------------
# Hierarchical Distributions -- Schaefer
#-------------------------------------------------------------------------------

pst = deserialize(projectdir("biofinder/schaefer/pst-taupos-1x2000.jls"));
pst2 = deserialize(projectdir("biofinder/schaefer/pst-tauneg-1x2000.jls"));
pst3 = deserialize(projectdir("biofinder/schaefer/pst-abneg-1x2000.jls"));

for p in [:Am, :Pm]
        param_test(pst[p], pst2[p]) |> println
        param_test(pst[p], pst3[p]) |> println
        param_test(pst2[p], pst3[p]) |> println
end

begin
        n_samples = 2000
        f = Figure(size=(2000, 750), fontsize=50)
        g1 = f[1, 1] = GridLayout()
        g2 = f[1, 2] = GridLayout()

        colors = alphacolor.(Makie.wong_colors(), 0.75)

        a3, a2, a1 = vec(pst3[:Am]), vec(pst2[:Am]), vec(pst[:Am])
        ax = Axis(g1[1,1], 
                xticklabelsize=30, xlabelsize=30, xlabel="1 / yr", 
                yticklabelsize=40, ylabelsize=30, ylabel="Density", yticks=(1.5:1:3.5, [L"A^-", L"A^+T^-", L"A^+T^+"]),
                titlesize=40, title="Production", xticks=-0.2:0.1:0.2, 
                xminorticks=-0.2:0.05:0.2, xminorticksvisible=true, 
                xticksize=20, xminorticksize=15, xgridcolor=RGBAf(0, 0, 0, 0.25))
        CairoMakie.xlims!(ax, -0.225, 0.225)
        CairoMakie.ylims!(ax, 0.75, 4)
        hideydecorations!(ax, label=false, ticklabels=false)
        hidexdecorations!(ax, grid=false, minorticks=false, label=false, ticks=false, ticklabels=false)
        hidespines!(ax, :t, :r, :l)

        rainclouds!(ax, 1.5 .* ones(n_samples), a3; orientation = :horizontal, gap=0.0, plot_boxplots = true, 
                                                cloud_width=0.5, clouds=hist, hist_bins=25,
                                                 color = colors[1])
        rainclouds!(ax, 2.5 .* ones(n_samples), a2; orientation = :horizontal, gap=0.0, plot_boxplots = true, 
                                                cloud_width=0.5, clouds=hist, hist_bins=50,
                                                color = colors[2])
        rainclouds!(ax, 3.5 .* ones(n_samples), a1; orientation = :horizontal, gap=0.0, plot_boxplots = true, 
                                                cloud_width=0.5, clouds=hist, hist_bins=25,
                                                color = colors[3])                                         
        bracket!(ax, median(a1), 1.25, median(a2), 1.25, text = "*", orientation = :down, textoffset = 20, linewidth = 2, color=(:grey, 0.75))
        bracket!(ax, median(a2), 1.05, median(a3), 1.05, text = "*", orientation = :down, textoffset = 20, linewidth = 2, color=(:grey, 0.75))
        bracket!(ax, median(a3), 1.05, median(a1), 1.05, text = "*", orientation = :down, textoffset = 20, linewidth = 2, color=(:grey, 0.75))

        p3, p2, p1 = vec(pst3[:Pm]), vec(pst2[:Pm]), vec(pst[:Pm])

        ax = Axis(g2[1,1], 
                xticklabelsize=30, xlabelsize=30, xlabel="1 / yr", yticks=(1.5:1:3.5, [L"A^-", L"A^+T^-", L"A^+T^+"]),
                yticklabelsize=40,
                titlesize=40, title="Transport", xticks=0.0:0.025:0.075,
                xminorticks=0.0:0.0125:1, xminorticksvisible=true, 
                xticksize=20, xminorticksize=15, xgridcolor=RGBAf(0, 0, 0, 0.25))
        CairoMakie.xlims!(ax, -0.005, 0.085)
        CairoMakie.ylims!(ax, 0.75, 4)
        hideydecorations!(ax)
        hidexdecorations!(ax, grid=false, minorticks=false, label=false, ticks=false, ticklabels=false)
        hidespines!(ax, :t, :r, :l)


        rainclouds!(ax, 1.5 .* ones(n_samples), p3; orientation = :horizontal, gap=0.0, plot_boxplots = true, 
                                                cloud_width=0.5, clouds=hist, hist_bins=25,
                                                color = colors[1])
        rainclouds!(ax, 2.5 .* ones(n_samples), p2; orientation = :horizontal, gap=0.0, plot_boxplots = true, 
                                                cloud_width=0.5, clouds=hist, hist_bins=50,
                                                color = colors[2])
        rainclouds!(ax, 3.5 .* ones(n_samples), p1; orientation = :horizontal, gap=0.0, plot_boxplots = true, 
                                         cloud_width=0.5, clouds=hist, hist_bins=25,
                                         color = colors[3])                                         
        bracket!(ax, median(p1), 1.25, median(p2), 1.25, text = "*", orientation = :down, textoffset = 20, linewidth = 2, color=(:grey, 0.75))
        bracket!(ax, median(p2), 1.05, median(p3), 1.05, text = "*", orientation = :down, textoffset = 20, linewidth = 2, color=(:grey, 0.75))
        bracket!(ax, median(p3), 1.05, median(p1), 1.05, text = "*", orientation = :down, textoffset = 20, linewidth = 2, color=(:grey, 0.5))
        colgap!(f.layout, 1, 50)
        f
end
save(projectdir("visualisation/inference/posteriors/output/bf2-schaefer-with-sig.pdf"), f)