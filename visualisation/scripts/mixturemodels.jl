using CSV
using DataFrames
using DrWatson: projectdir
using CairoMakie 
using Connectomes
using Distributions
using DataDrivenDiffEq
CairoMakie.activate!()
include(projectdir("adni/adni.jl"))

c = Connectome(Connectomes.connectome_path());

sub_data_path = projectdir("data/adni-data/AV1451_Diagnosis-STATUS-STIME-braak-regions.csv")
alldf = CSV.read(sub_data_path, DataFrame)
posdf = filter(x -> x.STATUS ∈ ["NEG", "POS"], alldf)
test = filter(x -> x.STATUS == "POS", alldf)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in c.parc.ID[1:end-1]]

data = ADNIDataSet(alldf, dktnames; min_scans=1)

_alldata = [calc_suvr(data, i) for i in 1:length(data)]
alldata = reduce(hcat, _alldata)

optimal_svd = optimal_shrinkage(alldata)

gmm_moments = CSV.read(projectdir("data/adni-data/component_moments.csv"), DataFrame)
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

node = 78
data = alldata[node, :]
moments = filter(x -> x.region == dktnames[node], gmm_moments)

begin
    f1 = Figure(resolution=(1000, 600), fontsize=20, font = "CMU Serif");
    ax = Axis(f1[1, 1], ylabel=L"Density", xlabel=L"SUVR", title="Population GMM for Left Putamen")
    xlims!(minimum(data) - 0.05, maximum(data) + 0.05)
    hist!(vec(data), color=(:grey, 0.7), bins=50, normalization=:pdf, label="Data")
   
    μ = moments.C0_mean[1]
    Σ = moments.C0_cov[1]
    plot_density!(μ, Σ; color=:darkblue, label="Healthy")
    vlines!(ax, quantile(Normal(μ, sqrt(Σ)), 0.5), linewidth=3, label=L"p_0", color=:darkblue)

    μ = moments.C1_mean[1]
    Σ = moments.C1_cov[1]
    plot_density!(μ, Σ; color=:darkred, label="Pathological")
    vlines!(ax, quantile(Normal(μ, sqrt(Σ)), 0.95), linewidth=3, label=L"p_\infty", color=:darkred)
    axislegend(; merge = true)
    f1
end
save(projectdir("adni/visualisation/models/gmm-lPutamen.pdf"), f1)
