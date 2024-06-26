using CSV
using DataFrames
using DrWatson

# Load data
taudata = CSV.read(projectdir("adni/data/new_new_data/UCBERKELEY_TAU_6MM_18Dec2023.csv"), DataFrame)
abdata =  CSV.read(projectdir("adni/data/new_new_data/UCBERKELEY_AMY_6MM_18Dec2023.csv"), DataFrame)

# replace missing ab status with 0
abstatus = abdata[:, "AMYLOID_STATUS"]
findall(x -> x isa Missing, abstatus)
abdata[:, "AMYLOID_STATUS"] = coalesce.(abdata[:,"AMYLOID_STATUS"], 0)

# check all ab positive ids are the same
zipped_abstatus = zip(findall(x -> x isa Int && x == 1, abstatus),
                      findall(x -> x isa Int && x == 1, abdata[:, "AMYLOID_STATUS"]))

allequal(allequal.(zipped_abstatus))

# make column in tau data for ab status
taudata.AB_Status = fill(-1,size(taudata, 1))

function get_sub_scans(df, id)
    filter(X -> X.RID == id, df)
end

function get_ab_status(taudf, abdf, id)
    tau_scans = get_sub_scans(taudf, id)
    ab_scans = get_sub_scans(abdf, id)
    if size(ab_scans, 1) == 0
        return -1
    else
        init_tau_date = minimum(tau_scans.SCANDATE)
        ab_scan_dates = ab_scans.SCANDATE
        nearest_ab_scan_idx = argmin(abs.(ab_scan_dates .- init_tau_date))
    
        return ab_scans[nearest_ab_scan_idx, "AMYLOID_STATUS"]
    end
end
get_ab_status(id) = get_ab_status(taudata, abdata, id)

IDs = unique(taudata.RID)
ab_stats = map(get_ab_status, IDs)

abdict = Dict(zip(IDs, ab_stats))

taudata.AB_Status = map(x -> abdict[x], taudata.RID)
CSV.write(projectdir("adni/data/new_new_data/UCBERKELEY_TAU_6MM_18Dec2023_AB_STATUS.csv"), taudata)

using ADNIDatasets, Connectomes
include(projectdir("functions.jl"))

connectome_path = Connectomes.connectome_path()
all_c = filter(Connectome(connectome_path; norm=true), 1e-2);
cortex = filter(x -> x.Lobe != "subcortex", all_c.parc);

mtl_regions = ["entorhinal", "Left-Amygdala", "Right-Amygdala"]
mtl = findall(x -> x ∈ mtl_regions, get_label.(cortex))
neo_regions = ["inferiortemporal", "middletemporal"]
neo = findall(x -> x ∈ neo_regions, get_label.(cortex))

posdf = filter(x -> x.AB_Status == 1, taudata)
dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in get_node_id.(cortex)]

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