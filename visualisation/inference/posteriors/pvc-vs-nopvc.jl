using DrWatson: projectdir
using Distributions
using Serialization
using DelimitedFiles
using Turing
using CairoMakie

pvc_pst = deserialize(projectdir("adni/chains-revisions/local-fkpp/pvc-ic/pst-taupos-1x2000.jls"));
pvc_pst2 = deserialize(projectdir("adni/chains-revisions/local-fkpp/pvc-ic/pst-tauneg-1x2000.jls"));
pvc_pst3 = deserialize(projectdir("adni/chains-revisions/local-fkpp/pvc-ic/pst-abneg-1x2000.jls"));

pst = deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-taupos-4x2000.jls"));
pst2 = deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-tauneg-4x2000.jls"));
pst3 = deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-abneg-4x2000.jls"));

begin
    f = Figure(size=(900, 300))
    ax = Axis(f[1,1], xlabel=L"A^{+}T^{+}", xlabelsize=20)
    hideydecorations!(ax)
    hidespines!(ax, :l, :t, :r)
    hist!(pvc_pst[:σ] |> vec, bins=50, normalization=:pdf, label="PVC")
    hist!(pst[:σ] |> vec, bins=50, normalization=:pdf, label="No PVC")

    ax = Axis(f[1,2], xlabel=L"A^{+}T^{-}", xlabelsize=20, title="Observation Noise", titlesize=20)
    hideydecorations!(ax)
    hidespines!(ax, :l, :t, :r)

    hist!(pvc_pst2[:σ] |> vec, bins=50, normalization=:pdf, label="PVC")
    hist!(pst2[:σ] |> vec, bins=50, normalization=:pdf, label="No PVC")

    ax = Axis(f[1,3], xlabel=L"A^{-}T^{-}", xlabelsize=20)
    hideydecorations!(ax)
    hidespines!(ax, :l, :t, :r)

    hist!(pvc_pst3[:σ] |> vec, bins=50, normalization=:pdf, label="PVC")
    hist!(pst3[:σ] |> vec, bins=50, normalization=:pdf, label="No PVC")
    axislegend(position=:rt)
    f
end
save(projectdir("visualisation/inference/posteriors/output-revisions/pvc-vs-nopvc-sigma.pdf"), f)

begin
    f = Figure(size=(900, 300))
    ax = Axis(f[1,1], xlabel=L"A^{+}T^{+}", xlabelsize=20)
    hideydecorations!(ax)
    hidespines!(ax, :l, :t, :r)
    hist!(pvc_pst[:Pm] .- mean(pvc_pst[:Pm]) |> vec, bins=50, normalization=:pdf, label="PVC")
    hist!(pst[:Pm] .- mean(pst[:Pm]) |> vec, bins=50, normalization=:pdf, label="No PVC")

    ax = Axis(f[1,2], xlabel=L"A^{+}T^{-}", xlabelsize=20, title="Mean Transport Coefficient", titlesize=20)
    hideydecorations!(ax)
    hidespines!(ax, :l, :t, :r)

    hist!(pvc_pst2[:Pm] .- mean(pvc_pst2[:Pm]) |> vec, bins=50, normalization=:pdf, label="PVC")
    hist!(pst2[:Pm] .- mean(pst2[:Pm]) |> vec, bins=50, normalization=:pdf, label="No PVC")

    ax = Axis(f[1,3], xlabel=L"A^{-}T^{-}", xlabelsize=20)
    hideydecorations!(ax)
    hidespines!(ax, :l, :t, :r)

    hist!(pvc_pst3[:Pm] .- mean(pvc_pst3[:Pm]) |> vec, bins=50, normalization=:pdf, label="PVC")
    hist!(pst3[:Pm] .- mean(pst3[:Pm]) |> vec, bins=50, normalization=:pdf, label="No PVC")
    axislegend(position=:rt)
    f
end
save(projectdir("visualisation/inference/posteriors/output-revisions/pvc-vs-nopvc-rho.pdf"), f)


begin
    f = Figure(size=(900, 300))
    ax = Axis(f[1,1], xlabel=L"A^{+}T^{+}", xlabelsize=20)
    hideydecorations!(ax)
    hidespines!(ax, :l, :t, :r)
    hist!(pvc_pst[:Am] .- mean(pvc_pst[:Am]) |> vec, bins=50, normalization=:pdf, label="PVC")
    hist!(pst[:Am] .- mean(pst[:Am]) |> vec, bins=50, normalization=:pdf, label="No PVC")

    ax = Axis(f[1,2], xlabel=L"A^{+}T^{-}", xlabelsize=20, title="Mean Production Coefficient", titlesize=20)
    hideydecorations!(ax)
    hidespines!(ax, :l, :t, :r)

    hist!(pvc_pst2[:Am] .- mean(pvc_pst2[:Am]) |> vec, bins=50, normalization=:pdf, label="PVC")
    hist!(pst2[:Am] .- mean(pst2[:Am]) |> vec, bins=50, normalization=:pdf, label="No PVC")

    ax = Axis(f[1,3], xlabel=L"A^{-}T^{-}", xlabelsize=20)
    hideydecorations!(ax)
    hidespines!(ax, :l, :t, :r)

    hist!(pvc_pst3[:Am] .- mean(pvc_pst3[:Am]) |> vec, bins=50, normalization=:pdf, label="PVC")
    hist!(pst3[:Am] .- mean(pst3[:Am]) |> vec, bins=50, normalization=:pdf, label="No PVC")
    axislegend(position=:rt)
    f
end
save(projectdir("visualisation/inference/posteriors/output-revisions/pvc-vs-nopvc-alpha.pdf"), f)


# ----------------------------------------------------------------------
# Data analysis 
# ----------------------------------------------------------------------
using ADNIDatasets
using Connectomes
using CSV, DataFrames
include(projectdir("functions.jl"))
include(projectdir("adni/inference/inference-preamble.jl"))

# Convenience functions for calculating longitudinal changes
function get_diff(x::Matrix{Float64})
    x[:,end] .- x[:,1]
end
function get_diff(x::Vector{Float64})
    x[end] - x[1]
end
cols = Makie.wong_colors();

pos_data
neg_data

_abneg_data = ADNIDataset(negdf, dktnames; min_scans=3, reference_region="INFERIORCEREBELLUM", qc=true)
n_subjects = length(_abneg_data)

abneg_mtl_pos = filter(x -> regional_mean(_abneg_data, mtl, x) >= mtl_cutoff, 1:n_subjects)
abneg_neo_pos = filter(x -> regional_mean(_abneg_data, neo, x) >= neo_cutoff, 1:n_subjects)

abneg_tau_pos = findall(x -> x ∈ unique([abneg_mtl_pos; abneg_neo_pos]), 1:n_subjects)
abneg_tau_neg = findall(x -> x ∉ abneg_tau_pos, 1:n_subjects)

abneg_data = _abneg_data[abneg_tau_neg]

abneg_vols = get_vol.(abneg_data)
total_vol_norm = [tp ./ sum(tp, dims=1) for tp in abneg_vols]
vols = [(vol ./ vol[:,1]) for vol in abneg_vols]

subsuvr = calc_suvr.(abneg_data)
subdata = [normalise(sd, u0, cc) for sd in subsuvr]

begin
    f = Figure(size=(600, 500))
    ax = Axis(f[2:3,1:2], xlabel="Change in SUVR", ylabel="% Change in Volume")
    ylims!(-0.05, 0.05)
    xlims!(-0.05, 0.05)
    scatter!(vec(mean(reduce(hcat, get_diff.(subdata)), dims=2)), vec(mean(reduce(hcat, get_diff.(vols)), dims=2)))
    
    ax = Axis(f[1,1:2])
    xlims!(-0.05, 0.05)
    hidexdecorations!(ax, ticks=false)
    hideydecorations!(ax, )
    density!(vec(mean(reduce(hcat, get_diff.(subdata)), dims=2)))

    ax = Axis(f[2:3,3])
    ylims!(-0.05, 0.05)
    hideydecorations!(ax, ticks=false)
    hidexdecorations!(ax, )
    density!(vec(mean(reduce(hcat, get_diff.(vols)), dims=2)), direction=:y)

    f
end

begin
    f = Figure(size=(600, 500))
    ax = Axis(f[2:3,1:2])
    ylims!(-0.5, 0.5)
    xlims!(-0.5, 0.5)
    scatter!(vec(reduce(hcat, get_diff.(subdata))), vec(reduce(hcat, get_diff.(vols))))
    
    ax = Axis(f[1,1:2])
    xlims!(-0.5, 0.5)
    hidexdecorations!(ax, ticks=false)
    hideydecorations!(ax, )
    density!(vec(reduce(hcat, get_diff.(subdata))))

    ax = Axis(f[2:3,3])
    ylims!(-0.5, 0.5)
    hidexdecorations!(ax, ticks=false)
    hideydecorations!(ax, )
    density!(vec(reduce(hcat, get_diff.(vols))), direction=:y)

    f
end