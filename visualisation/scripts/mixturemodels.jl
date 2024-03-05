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

ubase, upath = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(ubase)
cc = quantile.(upath, .95)

fg(x, μ, σ) = exp.(.-(x .- μ) .^ 2 ./ (2σ^2)) ./ (σ * √(2π))
function plot_density!(μ, Σ; color=:blue, label="")
    d = Normal(μ, sqrt(Σ))
    x = LinRange(quantile(d, .00001),quantile(d, .99999), 200)
    lines!(x, fg(x, μ, sqrt(Σ)); color = color, label=label)
    band!(x, fill(0, length(x)), fg(x, μ, sqrt(Σ)); color = (color, 0.1), label=label)
end

node = 29
data = alldata[node, :]
moments = filter(x -> x.region == dktnames[node], gmm_moments)

cols = Makie.wong_colors();
begin
    f1 = Figure(resolution=(800, 400), fontsize=35, font = "CMU Serif");
    ax = Axis(f1[1, 1], xlabel="SUVR")
    xlims!(minimum(data) - 0.05, maximum(data) + 0.05)
    hist!(vec(data), color=(:grey, 0.7), bins=100, normalization=:pdf, label="Data")
    hideydecorations!(ax)
    hidespines!(ax, :t, :r, :l)

    μ = moments.C0_mean[1]
    Σ = moments.C0_cov[1]
    plot_density!(μ, Σ; color=cols[1], label="Healthy")
    vlines!(ax, quantile(Normal(μ, sqrt(Σ)), 0.5), linewidth=3, label=L"p_0", color=cols[1])

    μ = moments.C1_mean[1]
    Σ = moments.C1_cov[1]
    plot_density!(μ, Σ; color=cols[6], label="Pathological")
    vlines!(ax, quantile(Normal(μ, sqrt(Σ)), 0.95), linewidth=3, label=L"p_\infty", color=cols[6])
    axislegend(; merge = true)
    f1
end
save(projectdir("visualisation/models/output/gmm-rIT.pdf"), f1)
