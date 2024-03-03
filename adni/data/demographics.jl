using DataFrames, CSV
using DrWatson
using Connectomes
using ADNIDatasets
using Dates
include(projectdir("functions.jl"))

connectome_path = Connectomes.connectome_path()
parc = Parcellation(connectome_path)
cortex = filter(x -> x.Lobe != "subcortex", parc);

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in get_node_id.(cortex)]

sub_data_path = projectdir("adni/data/new_new_data/UCBERKELEY_TAU_6MM_18Dec2023_AB_STATUS.csv")
alldf = CSV.read(sub_data_path, DataFrame)
posdf = filter(x -> x.AB_Status == 1, alldf)

data = ADNIDataset(posdf, dktnames; min_scans=3)

demo = CSV.read(projectdir("adni/data/new_new_data/demographics.csv"), DataFrame)
diagnostics = CSV.read(projectdir("adni/data/new_new_data/ADNIMERGE_15Jan2024.csv"), DataFrame);

#-------------------------------------------------------------------------------
# AB pos 
#-------------------------------------------------------------------------------
IDs = get_id(data)

_dmdf = reduce(vcat, [filter(x -> x.RID == id, demo) for id in IDs])
idx = [findfirst(isequal(id), _dmdf.RID) for id in IDs]
dmdf = _dmdf[idx,:]

mtl_regions = ["entorhinal", "Left-Amygdala", "Right-Amygdala"]
mtl = findall(x -> x ∈ mtl_regions, get_label.(cortex))
neo_regions = ["inferiortemporal", "middletemporal"]
neo = findall(x -> x ∈ neo_regions, get_label.(cortex))

function regional_mean(data, rois, sub)
    subsuvr = calc_suvr(data, sub)
    mean(subsuvr[rois,end])
end

mtl_cutoff = 1.375
neo_cutoff = 1.395

mtl_pos = filter(x -> regional_mean(data, mtl, x) >= mtl_cutoff, 1:97)
neo_pos = filter(x -> regional_mean(data, neo, x) >= neo_cutoff, 1:97)

tau_pos = findall(x -> x ∈ unique([mtl_pos; neo_pos]), 1:97)
tau_neg = findall(x -> x ∉ tau_pos, 1:97)

tauposdf = dmdf[tau_pos, :]
taunegdf = dmdf[tau_neg, :]

# Dataframe 
df = DataFrame(Group=String[], Age=Float64[], Gender=Float64[], Education=Float64[], CN = Float64[], MCI = Float64[], AD = Float64[])
#-------------------------------------------------------------------------------
# Tau Pos
posdf3 = filter(x -> x.RID ∈ IDs, posdf)
idx = [findfirst(isequal(id), posdf3.RID) for id in IDs]
posdfinit = posdf3[idx, :]
posdfinit_taupos = posdfinit[tau_pos,:]

diag_idx = [findfirst(isequal(id), diagnostics.RID) for id in posdfinit_taupos.RID]
pos_diag_df = diagnostics[diag_idx, :]
pdx_taupos = [count(==(pdx), pos_diag_df.DX_bl) for pdx in ["CN",  "SMC", "EMCI", "LMCI", "AD"]]
percent_pdx_taupos = pdx_taupos ./57

Group = "tau +"
age = mean(year.([data.SubjectData[i].scan_dates[1] for i in tau_pos]) .- (tauposdf.PTDOBYY))
Gender = count(==(2), tauposdf.PTGENDER) ./ length(tauposdf.PTGENDER)
Education = mean(tauposdf.PTEDUCAT)

push!(df, (Group, age, Gender, Education, 
          sum(percent_pdx_taupos[1:2]), sum(percent_pdx_taupos[3:4]), percent_pdx_taupos[5]))
#-------------------------------------------------------------------------------
# Tau neg
posdfinit_tauneg = posdfinit[tau_neg,:]

neg_diag_idx = [findfirst(isequal(id), diagnostics.RID) for id in posdfinit_tauneg.RID]
neg_diag_df = diagnostics[neg_diag_idx, :]
pdx_tauneg = [count(==(pdx), neg_diag_df.DX_bl) for pdx in ["CN",  "SMC", "EMCI", "LMCI", "AD"]]
percent_pdx_tauneg = pdx_tauneg ./ 40

Group = "tau -"
age = mean(year.([data.SubjectData[i].scan_dates[1] for i in tau_neg]) .- (taunegdf.PTDOBYY))
Gender = count(==(2), taunegdf.PTGENDER) ./ length(taunegdf.PTGENDER)
Education = mean(taunegdf.PTEDUCAT)
push!(df, (Group, age, Gender, Education, sum(percent_pdx_tauneg[1:2]), sum(percent_pdx_tauneg[3:4]), percent_pdx_tauneg[5]))

#-------------------------------------------------------------------------------
# AB neg 
#-------------------------------------------------------------------------------
negdf = filter(x -> x.AB_Status == 0, alldf)

negdata = ADNIDataset(negdf, dktnames; min_scans=3)
negIDs = get_id(negdata)

_dmdfneg = reduce(vcat, [filter(x -> x.RID == id, demo) for id in negIDs])
idxneg = [findfirst(isequal(id), _dmdfneg.RID) for id in negIDs]

dmdfneg = _dmdfneg[idxneg,:]

Group = "Ab -"
age = mean(year.([negdata.SubjectData[i].scan_dates[1] for i in 1:65]) .- (dmdfneg.PTDOBYY))
Gender = count(==(2), dmdfneg.PTGENDER) ./ length(dmdfneg.PTGENDER)
Education = mean(dmdfneg.PTEDUCAT)

negdf3 = filter(x -> x.RID ∈ negIDs, negdf)
idx = [findfirst(isequal(id), negdf3.RID) for id in negIDs]
abnegdfinit = negdf3[idx, :]

ab_diag_idx = [findfirst(isequal(id), diagnostics.RID) for id in abnegdfinit.RID]
ab_diag_df = diagnostics[ab_diag_idx, :]
pdx_ab = [count(==(pdx), ab_diag_df.DX_bl) for pdx in ["CN",  "SMC", "EMCI", "LMCI", "AD"]]
percent_pdx_ab = pdx_ab ./ 65

push!(df, (Group, age, Gender, Education, sum(percent_pdx_ab[1:2]), sum(percent_pdx_ab[3:4]), percent_pdx_ab[5]))

using PrettyTables
formatter = (v, i, j) -> round(v, digits = 3);
(df, digits=3)

pretty_table(df; formatters = ft_printf("%5.4f"), backend=Val(:latex))
