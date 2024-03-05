using Connectomes
using ADNIDatasets
using CSV, DataFrames
using DrWatson: projectdir
using Distributions
using Serialization
using DelimitedFiles
using MCMCChains

lf = deserialize(projectdir("adni/chains/local-fkpp/connectome-weights/ll-taupos-2x2000-lengthfree.jls"));
diff = deserialize(projectdir("adni/chains/local-fkpp/connectome-weights/ll-taupos-2x2000-diffusive.jls"));
bal = deserialize(projectdir("adni/chains/local-fkpp/ballistic/ll-taupos-4x2000.jls"));
max_lls= [maximum(dict["data"]) for dict in [lf, diff, bal]]

using CairoMakie

f = hist(vec(lf["data"]), bins=50, label="length free")
hist!(vec(diff["data"]), bins=50, label="diffusive")
hist!(vec(bal["data"]), bins=50, label="ballistic")
axislegend()
f