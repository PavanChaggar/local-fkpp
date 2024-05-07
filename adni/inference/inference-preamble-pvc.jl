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
# sub_data_path = projectdir("adni/data/new_new_data/UCBERKELEY_TAU_6MM_18Dec2023_AB_STATUS.csv")
# alldf = CSV.read(sub_data_path, DataFrame)
# suvrnames = [ADNIDatasets.suvr_name.(dktnames); "INFERIORCEREBELLUM_SUVR"] # no pvc
# d = Array(dropmissing(alldf[:, suvrnames]))
# writedlm(projectdir("data-nopvc-ic.txt"), transpose(d ./ d[:,end]))
sub_data_path = projectdir("adni/data/new_new_data/UCBERKELEY_TAU_6MM_18Dec2023_AB_STATUS.csv")
alldf_nopvc = CSV.read(sub_data_path, DataFrame)

sub_data_path_pvc = projectdir("adni/data/new_new_data/pvc/UC-Berkeley-TAUPVC-6MM-Mar-30-2024-AB-Status.csv")
alldf = CSV.read(sub_data_path_pvc, DataFrame)

function check_qc(df, rid)
    _df = filter(x -> x.RID == rid, df)
    if length(_df.qc_flag) == 0
        return -1
    end

    if _df.qc_flag[1] == 2 && allequal(_df.qc_flag)
        return 2
    else
        return -1
    end
end
check_qc(rid) =  check_qc(alldf_nopvc, rid)

alldf.qc_flag = map(check_qc, alldf.RID)
# suvrnames = ADNIDatasets.suvr_name.(dktnames) # pvc
# d = Array(dropmissing(alldf[:, suvrnames]))
# writedlm(projectdir("py-analysis/data-pvc-ic.txt"), transpose(d))

#posdf = filter(x -> x.STATUS == "POS", alldf)

posdf = filter(x -> x.AB_Status == 1, alldf)
negdf = filter(x -> x.AB_Status == 0, alldf)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in get_node_id.(cortex)]

data = ADNIDataset(posdf, dktnames; min_scans=3, reference_region="INFERIORCEREBELLUM", qc=true)

n_data = length(data)

gmm_moments = CSV.read(projectdir("py-analysis/ic-pvc-moments-prob.csv"), DataFrame)
# gmm_moments = CSV.read(projectdir("py-analysis/wm-nopvc-moments-prob.csv"), DataFrame)
# gmm_moments = CSV.read(projectdir("adni/data/component_moments.csv"), DataFrame)

mtl_cutoff = mean(gmm_moments.cutoff[mtl])
neo_cutoff = mean(gmm_moments.cutoff[neo])

mtl_pos = filter(x -> regional_mean(data, mtl, x) >= mtl_cutoff, 1:n_data)
neo_pos = filter(x -> regional_mean(data, neo, x) >= neo_cutoff, 1:n_data)

tau_pos = findall(x -> x ∈ unique([mtl_pos; neo_pos]), 1:n_data)
tau_neg = findall(x -> x ∉ tau_pos, 1:n_data)

n_pos = length(tau_pos)
n_neg = length(tau_neg)

# Regional parameters
ubase, upath = get_dkt_moments(gmm_moments)
u0 = mean.(ubase)
cc = quantile.(upath, .99)