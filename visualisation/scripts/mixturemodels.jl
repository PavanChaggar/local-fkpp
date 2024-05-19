using CSV
using DataFrames
using DrWatson: projectdir
using CairoMakie 
using Connectomes
using ADNIDatasets
using Distributions
CairoMakie.activate!()
include(projectdir("functions.jl"))

parc = Parcellation(Connectomes.connectome_path())
cortex = filter(x -> get_lobe(x) != "subcortex", parc);

# sub_data_path = projectdir("adni/data/new_data/UCBERKELEYAV1451_8mm_02_17_23_AB_Status.csv")
sub_data_path = projectdir("adni/data/new_new_data/UCBERKELEY_TAU_6MM_18Dec2023_AB_STATUS.csv")
alldf = CSV.read(sub_data_path, DataFrame)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in get_node_id.(cortex)]

data = ADNIDataset(alldf, dktnames; min_scans=1)

_alldata = calc_suvr.(data)
alldata = reduce(hcat, _alldata)

gmm_moments = CSV.read(projectdir("adni/data/component_moments.csv"), DataFrame)
dktmoments = filter(x -> x.region ∈ dktnames, gmm_moments)

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
mm = [MixtureModel([u0, ui], [w...]) for (u0, ui, w) in zip(ubase, upath, weights)]
u0 = mean.(ubase)
cc = quantile.(mm, .99)

fg(x, μ, σ) = exp.(.-(x .- μ) .^ 2 ./ (2σ^2)) ./ (σ * √(2π))
function plot_density!(μ, Σ, weight; color=:blue, label="")
    d = Normal(μ, sqrt(Σ))
    x = LinRange(quantile(d, .00001),quantile(d, .99999), 200)
    lines!(x, weight .* fg(x, μ, sqrt(Σ)); color = color, label=label)
    band!(x, fill(0, length(x)), weight .* fg(x, μ, sqrt(Σ)); color = (color, 0.1))
end

cols = Makie.wong_colors();
begin
    node = 29
    data = alldata[node, :]
    moments = filter(x -> x.region == dktnames[node], gmm_moments)

    f1 = Figure(resolution=(500, 400), fontsize=20, font = "CMU Serif");
    ax = Axis(f1[1, 1], xlabel="SUVR")
    xlims!(minimum(data) - 0.05, maximum(data) + 0.05)
    hist!(vec(data), color=(:grey, 0.7), bins=100, normalization=:pdf, label="Data")
    hideydecorations!(ax)
    hidespines!(ax, :t, :r, :l)

    μ = moments.C0_mean[1]
    Σ = moments.C0_cov[1]
    plot_density!(μ, Σ, weights[node][1]; color=cols[1], label="Healthy")
    vlines!(ax, u0[node], linewidth=3, label=L"p_0", color=cols[1])

    μ = moments.C1_mean[1]
    Σ = moments.C1_cov[1]
    plot_density!(μ, Σ, weights[node][2]; color=cols[6], label="Pathological")
    vlines!(ax, cc[node], linewidth=3, label=L"p_\infty", color=cols[6])
    axislegend(; merge = true)
    f1
end
save(projectdir("visualisation/models/output/gmm-rIT.pdf"), f1)


cols = Makie.wong_colors();
begin
    f1 = Figure(size=(1000, 300), fontsize=20, font = "CMU Serif");
    node = 36
    data = alldata[node, :]
    moments = filter(x -> x.region == dktnames[node], gmm_moments)

    ax = Axis(f1[1, 1], xlabel="SUVR", title="Right Amygdala")
    CairoMakie.xlims!(minimum(data) - 0.05, maximum(data) + 0.05)
    hist!(vec(data), color=(:grey, 0.7), bins=50, normalization=:pdf, label="Data")
    hideydecorations!(ax)
    hidespines!(ax, :t, :r, :l)

    μ = moments.C0_mean[1]
    Σ = moments.C0_cov[1]
    plot_density!(μ, Σ, weights[node][1]; color=cols[1], label="Healthy")
    vlines!(ax, quantile(Normal(μ, sqrt(Σ)), 0.5), linestyle=:dash, linewidth=3, label=L"s_0", color=cols[1])

    μ = moments.C1_mean[1]
    Σ = moments.C1_cov[1]
    plot_density!(μ, Σ, weights[node][2]; color=cols[6], label="Pathological")
    vlines!(ax, quantile(mm[node], 0.99), linestyle=:dash, linewidth=3, label=L"s_\infty", color=cols[6])

    node = 29
    data = alldata[node, :]
    moments = filter(x -> x.region == dktnames[node], gmm_moments)
    
    ax = Axis(f1[1, 2], xlabel="SUVR", title="Right Inferior Temporal")
    CairoMakie.xlims!(minimum(data) - 0.05, maximum(data) + 0.05)
    hist!(vec(data), color=(:grey, 0.7), bins=50, normalization=:pdf, label="Data")
    hideydecorations!(ax)
    hidespines!(ax, :t, :r, :l)

    μ = moments.C0_mean[1]
    Σ = moments.C0_cov[1]
    plot_density!(μ, Σ, weights[node][1]; color=cols[1], label="Healthy")
    vlines!(ax, quantile(Normal(μ, sqrt(Σ)), 0.5), linestyle=:dash, linewidth=3, label=L"s_0", color=cols[1])

    μ = moments.C1_mean[1]
    Σ = moments.C1_cov[1]
    plot_density!(μ, Σ, weights[node][2]; color=cols[6], label="Pathological")
    vlines!(ax, quantile(mm[node], 0.99), linestyle=:dash, linewidth=3, label=L"s_\infty", color=cols[6])

    axislegend(ax; merge = true, patchsize=(30, 30), labelsize=22)
    f1
end
save(projectdir("visualisation/models/output/gmm-amygdala-rIT.pdf"), f1)

cols = Makie.wong_colors();
begin
    f1 = Figure(size=(250, 400), fontsize=15, font = "CMU Serif");
    node = 71
    data = alldata[node, :]
    moments = filter(x -> x.region == dktnames[node], gmm_moments)

    ax = Axis(f1[1, 1], xlabel="SUVR", title="Left Hippocampus", titlesize=15)
    CairoMakie.xlims!(minimum(data) - 0.05, quantile(mm[node], 0.99) .+ 0.05)
    hist!(vec(data), color=(:grey, 0.7), bins=40, normalization=:pdf)
    hideydecorations!(ax)
    hidexdecorations!(ax, grid=false, ticks=false, ticklabels=false)
    hidespines!(ax, :t, :r, :l)

    μ = moments.C0_mean[1]
    Σ = moments.C0_cov[1]
    plot_density!(μ, Σ, weights[node][1]; color=cols[1], label="Healthy")
    vlines!(ax, quantile(Normal(μ, sqrt(Σ)), 0.5), linestyle=:dash, linewidth=3, label=L"s_0", color=cols[1])

    μ = moments.C1_mean[1]
    Σ = moments.C1_cov[1]
    plot_density!(μ, Σ, weights[node][2]; color=cols[6], label="Pathological")
    vlines!(ax, quantile(mm[node], 0.99), linestyle=:dash, linewidth=3, label=L"s_\infty", color=cols[6])

    node = 65
    data = alldata[node, :]
    moments = filter(x -> x.region == dktnames[node], gmm_moments)
    
    ax = Axis(f1[2, 1], xlabel="SUVR", title="Left Inferior Temporal", titlesize=15)
    CairoMakie.xlims!(minimum(data) - 0.05, quantile(mm[node], 0.99) .+ 0.05)
    hist!(vec(data), color=(:grey, 0.7), bins=40, normalization=:pdf)
    hideydecorations!(ax)
    hidespines!(ax, :t, :r, :l)

    μ = moments.C0_mean[1]
    Σ = moments.C0_cov[1]
    plot_density!(μ, Σ, weights[node][1]; color=cols[1], label="Healthy")
    vlines!(ax, quantile(Normal(μ, sqrt(Σ)), 0.5), linestyle=:dash, linewidth=3, label=L"s_0", color=cols[1])

    μ = moments.C1_mean[1]
    Σ = moments.C1_cov[1]
    plot_density!(μ, Σ, weights[node][2]; color=cols[6], label="Pathological")
    vlines!(ax, quantile(mm[node], 0.99), linestyle=:dash, linewidth=3, label=L"s_\infty", color=cols[6])

    Legend(f1[3, 1], ax, framevisible=false, patchsize=(10, 10), labelsize=20, nbanks=2, rowgap = 0, tellheight=true, tellwidth=false)
    rowgap!(f1.layout, 1)
    f1
end
save(projectdir("visualisation/models/output/gmm-hc-lIT-vertical.pdf"), f1)
