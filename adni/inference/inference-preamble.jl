#-------------------------------------------------------------------------------
# Connectome and ROIs
#-------------------------------------------------------------------------------
connectome_path = Connectomes.connectome_path()
all_c = filter(Connectome(connectome_path; norm=true, weight_function = (n, l) -> n ), 1e-2);

subcortex = filter(x -> get_lobe(x) == "subcortex", all_c.parc);
cortex = filter(x -> get_lobe(x) != "subcortex", all_c.parc);

c = slice(all_c, cortex) |> filter

mtl_regions = ["entorhinal", "Left-Amygdala", "Right-Amygdala"]
mtl = findall(x -> x ∈ mtl_regions, get_label.(cortex))
neo_regions = ["inferiortemporal", "middletemporal"]
neo = findall(x -> x ∈ neo_regions, get_label.(cortex))
#-------------------------------------------------------------------------------
# Data 
#-----------------------------------------------------------------------------
sub_data_path = projectdir("adni/data/new_new_data/UCBERKELEY_TAU_6MM_18Dec2023_AB_STATUS.csv")

alldf = CSV.read(sub_data_path, DataFrame)

#posdf = filter(x -> x.STATUS == "POS", alldf)
posdf = filter(x -> x.AB_Status == 1, alldf)
negdf = filter(x -> x.AB_Status == 0, alldf)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in get_node_id.(cortex)]

data = ADNIDataset(posdf, dktnames; min_scans=3, qc=true)
n_data = length(data)

# Ask Jake where we got these cutoffs from? 
mtl_cutoff = 1.375
neo_cutoff = 1.395

mtl_pos = filter(x -> regional_mean(data, mtl, x) >= mtl_cutoff, 1:n_data)
neo_pos = filter(x -> regional_mean(data, neo, x) >= neo_cutoff, 1:n_data)

tau_pos = findall(x -> x ∈ unique([mtl_pos; neo_pos]), 1:n_data)
tau_neg = findall(x -> x ∉ tau_pos, 1:n_data)

pos_data = data[tau_pos]
neg_data = data[tau_neg]

n_pos = length(pos_data)
n_neg = length(neg_data)

# Regional parameters
gmm_moments = CSV.read(projectdir("adni/data/component_moments.csv"), DataFrame)

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
quantile.(mm, .99)
cc = quantile.(mm, .99)

u0 = mean.(ubase)
cc = quantile.(upath, .99)