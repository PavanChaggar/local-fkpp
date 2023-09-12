using Connectomes
using ADNIDatasets
using CSV, DataFrames
using DrWatson: projectdir
using Distributions
using Serialization
using DelimitedFiles, LinearAlgebra
using MCMCChains
using Random
using LinearAlgebra
using SparseArrays
include(projectdir("functions.jl"))
#-------------------------------------------------------------------------------
# Connectome and ROIs
#-------------------------------------------------------------------------------
connectome_path = Connectomes.connectome_path()
all_c = filter(Connectome(connectome_path; norm=true, weight_function = (n, l) -> n ./ l), 1e-2);

subcortex = filter(x -> x.Lobe == "subcortex", all_c.parc);
cortex = filter(x -> x.Lobe != "subcortex", all_c.parc);

c = slice(all_c, cortex) |> filter

mtl_regions = ["entorhinal", "Left-Amygdala", "Right-Amygdala"]
mtl = findall(x -> x ∈ mtl_regions, cortex.Label)
neo_regions = ["inferiortemporal", "middletemporal"]
neo = findall(x -> x ∈ neo_regions, cortex.Label)
#-------------------------------------------------------------------------------
# Data 
#-----------------------------------------------------------------------------
sub_data_path = projectdir("adni/data/new_data/UCBERKELEYAV1451_8mm_02_17_23_AB_Status.csv")
alldf = CSV.read(sub_data_path, DataFrame)

#posdf = filter(x -> x.STATUS == "POS", alldf)
posdf = filter(x -> x.AB_Status == 1, alldf)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in cortex.ID]

data = ADNIDataset(posdf, dktnames; min_scans=3)
n_data = length(data)
# Ask Jake where we got these cutoffs from? 
mtl_cutoff = 1.375
neo_cutoff = 1.395

mtl_pos = filter(x -> regional_mean(data, mtl, x) >= mtl_cutoff, 1:n_data)
neo_pos = filter(x -> regional_mean(data, neo, x) >= neo_cutoff, 1:n_data)

tau_pos = findall(x -> x ∈ unique([mtl_pos; neo_pos]), 1:n_data)
tau_neg = findall(x -> x ∉ tau_pos, 1:n_data)

n_pos = length(tau_pos)
n_neg = length(tau_neg)

mtl_pos_first = filter(x -> regional_mean(data, mtl, x, 1) >= mtl_cutoff, 1:n_data)
neo_pos_first = filter(x -> regional_mean(data, neo, x, 1) >= neo_cutoff, 1:n_data)

tau_pos_first = findall(x -> x ∈ unique([mtl_pos_first; neo_pos_first]), 1:n_data)
tau_neg_first = findall(x -> x ∉ tau_pos_first, 1:n_data)

transition_subs= filter(x -> x ∈ tau_neg_first, tau_pos)
transition_subs_idx = findall(x -> x ∈ tau_neg_first, tau_pos)

pst = deserialize(projectdir("adni/chains/local-fkpp/pst-taupos-4x2000.jls"));
meanpst = mean(pst);

as = [meanpst["α[$i]", :mean] for i in 1:31]
ps = [meanpst["ρ[$i]", :mean] for i in 1:31]

using CairoMakie

f, ax = scatter(as, ps)
ax.xlabel="production"
ax.ylabel="transport"
scatter!(as[transition_subs_idx], ps[transition_subs_idx], color=:red)
f

as[transition_subs_idx]

for k in transition_subs[[1,4]]
    t = get_times(data, k)
    f = scatter(t, [regional_mean(data, neo, k, i) for i in 1:length(get_times(data, k))])
    display(f)
end

_subdata = [calc_suvr(data, i) for i in tau_pos]
subdata = [normalise(sd, u0, cc) for sd in _subdata]
mean_subdata = mean.(subdata, dims=1)

for i in [1, 4]
    t = get_times(data, transition_subs[i])
    f = scatter(t, vec(mean_subdata[i]))
    display(f)
end
