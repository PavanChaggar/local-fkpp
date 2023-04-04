using CSV
using DataFrames
using DrWatson

# Load data
taudata = CSV.read(projectdir("adni/data/new_data/UCBERKELEYAV1451_8mm_02_17_23.csv"), DataFrame)
abdata =  CSV.read(projectdir("adni/data/new_data/UCBERKELEYAV45_8mm_02_17_23.csv"), DataFrame)

# replace missing ab status with 0
abstatus = abdata[:, "SUMMARYSUVR_COMPOSITE_REFNORM_0.78CUTOFF"]
abdata[:, "SUMMARYSUVR_COMPOSITE_REFNORM_0.78CUTOFF"] = coalesce.(abdata[:,"SUMMARYSUVR_COMPOSITE_REFNORM_0.78CUTOFF"], 0)

# check all ab positive ids are the same
zipped_abstatus = zip(findall(x -> x isa Int && x == 1, abstatus),
                      findall(x -> x isa Int && x == 1, abdata[:, "SUMMARYSUVR_COMPOSITE_REFNORM_0.78CUTOFF"]))

allequal(allequal.(zipped_abstatus))

# make column in tau data for ab status
taudata.AB_Status = fill(0,size(taudata, 1))
taudata.AB_Status

# retrieve ab status for a subject
function get_ab_status(df, id)
    sub_df = filter(X -> X.RID == id, df)
    status = sub_df[:,"SUMMARYSUVR_COMPOSITE_REFNORM_0.78CUTOFF"]
    if length(status) == 0 
        return 0
    end
    if 0 ∈ status 
        return 0 
    else
        return 1
    end
end
get_ab_status(id::Int) = get_ab_status(abdata, id)

IDs = unique(taudata.RID)
tau_abstatus = map(get_ab_status, IDs)

abdict = Dict(zip(IDs, tau_abstatus))

taudata.AB_Status = map(x -> abdict[x], taudata.RID)

CSV.write(projectdir("adni/data/new_data/UCBERKELEYAV1451_8mm_02_17_23_AB_Status.csv"), taudata)