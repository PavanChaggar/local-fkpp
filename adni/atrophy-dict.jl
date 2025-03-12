using CSV, DataFrames
using DrWatson
using Connectomes

connectome_path = Connectomes.connectome_path()
all_c = filter(Connectome(connectome_path; norm=true, weight_function = (n, l) -> n ), 1e-2);

subcortex = filter(x -> get_lobe(x) == "subcortex", all_c.parc);
cortex = filter(x -> get_lobe(x) != "subcortex", all_c.parc);

dktdict = Connectomes.node2FS()

df = CSV.read(projectdir("adni/data/ucsf-smri.csv"), DataFrame)
dict_df = CSV.read(projectdir("adni/data/adni-dictionary.csv"), DataFrame)

# ucsf_dict_df = filter(x -> x.CRFNAME == "Cross-Sectional FreeSurfer (7.x)", dict_df)

# df_names = filter(x -> startswith(x, "ST"), names(df))[2:end]
ucsf_fldn_dict_df = filter(x -> startswith(x.FLDNAME, "ST") && x.CRFNAME == "Cross-Sectional FreeSurfer (7.x)", dict_df)

function make_dkt_name(names)
    words = split(names, "_")
    if words[end-1] == "rh" || words[end-1] == "Right"
        hem = "right"
    elseif words[end-1] == "lh" || words[end-1] == "Left"
        hem = "left"
    end
    return lowercase(hem * words[end])
end
dktnames = [dktdict[i] for i in get_node_id.(cortex)]
ucsf_dkt_names = make_dkt_name.(dktnames)

push!(ucsf_dkt_names, "icv")
push!(dktnames, "icv")

ucsf_dkt_dict = Dict(zip(ucsf_dkt_names, dktnames))

roi_df = filter(x -> contains(lowercase(x.TEXT), ucsf_dkt_names[end]), ucsf_fldn_dict_df)

function make_ucsf_name(rois)
    fldnames = Vector{String}()
    labels = Vector{String}()
    for roi in rois
        roi_df = filter(x -> contains(lowercase(x.TEXT), roi), ucsf_fldn_dict_df)
    
        for _df in eachrow(roi_df)
            push!(fldnames, _df.FLDNAME)
            push!(labels, prod(split(_df.TEXT)[1:2]) * "_" * ucsf_dkt_dict[roi])
        end
    end
    return fldnames, labels
end

fldnames, labels = make_ucsf_name(ucsf_dkt_names)

labels

newdf = deepcopy(df)
[rename!(newdf, f => l) for (f, l) in zip(fldnames, labels)];

testdf = newdf[:, [names(df)[1:20]; labels]]
CSV.write(projectdir("adni/data/ucsf-FS-smri.csv"), newdf[:, [names(df)[1:20]; labels]])