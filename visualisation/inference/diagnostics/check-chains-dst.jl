using Connectomes
using ADNIDatasets
using CSV, DataFrames
using DrWatson: projectdir
using Distributions
using Serialization
using DelimitedFiles
using MCMCChains

#-------------------------------------------------------------------------------
# Hierarchical Distributions
#-------------------------------------------------------------------------------
pst = deserialize(projectdir("adni/chains/local-fkpp/pst-taupos-4x2000.jls"));
pst2 = deserialize(projectdir("adni/chains/local-fkpp/pst-tauneg-4x2000.jls"));
pst3 = deserialize(projectdir("adni/chains/local-fkpp/pst-abneg-4x2000.jls"));

[p[:numerical_error] |> sum for p in [pst, pst2, pst3]]

using CairoMakie; CairoMakie.activate!()

begin
    f = Figure(resolution=(2000, 1000), fontsize=50)
    g1 = f[1, 1] = GridLayout()
    g2 = f[1, 2] = GridLayout()
    g3 = f[2, 1] = GridLayout()
    g4 = f[2, 2] = GridLayout()
    g5 = f[1:2, 3] = GridLayout()

    ax = Axis(g1[1,1], 
            xticklabelsize=20, xlabelsize=30, xlabel="1 / yr", 
            yticklabelsize=20, ylabelsize=30, ylabel="Density",
            titlesize=40, title="Diffusion", xticks=LinearTicks(6))
    hideydecorations!(ax, label=false)
    hidexdecorations!(ax, label=false, ticks=false, ticklabels=false)
    hidespines!(ax, :t, :r, :l)

    hist!(vec(pst[:Pm]), bins=50, color=(:blue, 0.6), label=L"A\beta^+ \tau P^+", normalization=:pdf, strokewidth=1, strokecolor=:blue)
    hist!(vec(pst2[:Pm]), bins=50, color=(:red, 0.6), label=L"A\beta^+ \tau P^-", normalization=:pdf, strokewidth=1, strokecolor=:red)
    hist!(vec(pst3[:Pm]), bins=50, color=(:green, 0.6), label=L"A\beta^-", normalization=:pdf, strokewidth=1, strokecolor=:green)
    
    ax = Axis(g3[1,1], 
            xticklabelsize=20, xlabelsize=30, xlabel="s.d.", 
            yticklabelsize=20, ylabelsize=30, ylabel="Density", xticks=LinearTicks(6))
    hideydecorations!(ax, label=false)
    hidexdecorations!(ax, label=false, ticks=false, ticklabels=false)
    hidespines!(ax, :t, :r, :l)

    hist!(vec(pst[:Ps]), bins=50, color=(:blue, 0.6), label=L"A\beta^+ \tau P^+", normalization=:pdf, strokewidth=1, strokecolor=:blue)
    hist!(vec(pst2[:Ps]), bins=50, color=(:red, 0.6), label=L"A\beta^+ \tau P^-", normalization=:pdf, strokewidth=1, strokecolor=:red)
    hist!(vec(pst3[:Ps]), bins=50, color=(:green, 0.6), label=L"A\beta^-", normalization=:pdf, strokewidth=1, strokecolor=:green)

    ax = Axis(g2[1,1], 
            xticklabelsize=20, xlabelsize=30, xlabel="1 / yr", 
            yticklabelsize=20, ylabelsize=30, ylabel="Density",
            titlesize=40, title="Growth", xticks=LinearTicks(6))
    hideydecorations!(ax)
    hidexdecorations!(ax, label=false, ticks=false, ticklabels=false)
    hidespines!(ax, :t, :r, :l)
    
    hist!(vec(pst[:Am]), bins=50, color=(:blue, 0.6), label=L"A\beta^+ \tau P^+", normalization=:pdf, strokewidth=1, strokecolor=:blue)
    hist!(vec(pst2[:Am]), bins=50, color=(:red, 0.6), label=L"A\beta^+ \tau P^-", normalization=:pdf, strokewidth=1, strokecolor=:red)
    hist!(vec(pst3[:Am]), bins=50, color=(:green, 0.6), label=L"A\beta^-", normalization=:pdf, strokewidth=1, strokecolor=:green)

    ax = Axis(g4[1,1], 
            xticklabelsize=20, xlabelsize=30, xlabel="s.d.", 
            yticklabelsize=20, ylabelsize=33, ylabel="Density", xticks=LinearTicks(6))
    hideydecorations!(ax)
    hidexdecorations!(ax, label=false, ticks=false, ticklabels=false)
    hidespines!(ax, :t, :r, :l)

    hist!(vec(pst[:As]), bins=50, color=(:blue, 0.6), label=L"A\beta^+ \tau P^+", normalization=:pdf, strokewidth=1, strokecolor=:blue)
    hist!(vec(pst2[:As]), bins=50, color=(:red, 0.6), label=L"A\beta^+ \tau P^-", normalization=:pdf, strokewidth=1, strokecolor=:red)
    hist!(vec(pst3[:As]), bins=50, color=(:green, 0.6), label=L"A\beta^-", normalization=:pdf, strokewidth=1, strokecolor=:green)

    Legend(g5[1,1], ax, framevisible = false)
    # for (label, layout) in zip(["A", "B", "C", "D"], [g1, g2, g3, g4])
    #     Label(layout[1, 1, TopLeft()], label,
    #         textsize = 30,
    #         font = "TeX Gyre Heros Bold",
    #         padding = (0, 5, 5, 0),
    #         halign = :right)
    # end
    colgap!(f.layout, 1, 50)

    f
end
save(projectdir("adni/visualisation/hier-inf/c99/hier-dsts.pdf"), f)
save(projectdir("adni/visualisation/hier-inf/png/c99/hier-dsts.png"), f)

begin
    f = Figure(resolution=(2000,1000));
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
    for i in 1:25
        hist!(ax11, vec(pst[Symbol("α[$i]")]), bins=50, color=(:red, 0.6), normalization=:pdf, label=L"A\beta^+ \tau^+")
    end

    
    hideydecorations!(ax12, label=false)
    hidexdecorations!(ax12, label=false, ticks=false, grid=false)
    hidespines!(ax12, :t, :r, :l)
    for i in 1:21
        hist!(ax12, vec(pst2[Symbol("α[$i]")]), bins=50, color=(:blue, 0.6), normalization=:pdf, label=L"A\beta^+ \tau^-")
    end

    
    hideydecorations!(ax13, label=false)
    hidexdecorations!(ax13, label=false, ticks=false, ticklabels=false, grid=false)
    hidespines!(ax13, :t, :r, :l)
    xlims!(-1.5, 1.)
    for i in 1:39
        hist!(ax13, vec(pst3[Symbol("α[$i]")]), bins=50, color=(:green, 0.6), normalization=:pdf, label=L"A\beta^-")
    end

    ax21 = Axis(g2[1, 1], title="Diffusion", titlesize=40, ylabel="Density", ylabelsize=30, xticks=-0:0.005:0.02);
    ax22 = Axis(g2[2, 1], ylabel="Density", ylabelsize=30, xticks=-0:0.005:0.02);
    ax23 = Axis(g2[3, 1], ylabel="Density", ylabelsize=30, xlabel="mm² / yr", xlabelsize=30, xticks=-0:0.005:0.02);
    linkaxes!(ax23, ax22)
    linkaxes!(ax23, ax21)

    hideydecorations!(ax21, label=false)
    hidexdecorations!(ax21, label=false, ticks=false, grid=false)
    hidespines!(ax21, :t, :r, :l)
    for i in 1:25
        hist!(ax21, vec(pst[Symbol("ρ[$i]")]), bins=50, color=(:red, 0.6), normalization=:pdf, label=L"A\beta^+ \tau^+")
    end

    hideydecorations!(ax22, label=false)
    hidexdecorations!(ax22, label=false, ticks=false, grid=false)
    hidespines!(ax22, :t, :r, :l)
    for i in 1:21
        hist!(ax22, vec(pst2[Symbol("ρ[$i]")]), bins=50, color=(:blue, 0.6), normalization=:pdf, label=L"A\beta^+ \tau^-")
    end

    hideydecorations!(ax23, label=false)
    hidexdecorations!(ax23, label=false, ticks=false, ticklabels=false, grid=false)
    hidespines!(ax23, :t, :r, :l)
    xlims!(-0.001, 1.0)
    for i in 1:39
        hist!(ax23, vec(pst3[Symbol("ρ[$i]")]), bins=50, color=(:green, 0.6), normalization=:pdf, label=L"A\beta^-")
    end

    elem_1 = PolyElement(color = (:red, 0.6))
    elem_2 = PolyElement(color = (:blue, 0.6))
    elem_3 = PolyElement(color = (:green, 0.6))
    legend = Legend(g7[1,1],
           [elem_1, elem_2, elem_3],
           [L"A\beta^+ \tau P^+", L"A\beta^+ \tau P^-", L"A\beta^-"],
           patchsize = (35, 35), rowgap = 10, framevisible=false, labelsize=30)
    colgap!(f.layout, 1, 50)
    f
end
save(projectdir("adni/visualisation/hier-inf/c99/sub-dsts.pdf"), f)
save(projectdir("adni/visualisation/hier-inf/png/c99/sub-dsts.png"), f)