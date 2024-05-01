using Connectomes
using ADNIDatasets
using CSV, DataFrames
using DrWatson: projectdir
using Serialization
using DelimitedFiles
using CairoMakie
include(projectdir("functions.jl"))
 
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Connectome and ROIs
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
# suvrnames = [ADNIDatasets.suvr_name.(dktnames); "INFERIORCEREBELLUM_SUVR"] # no pvc
# d = Array(dropmissing(alldf[:, suvrnames]))
# writedlm(projectdir("data-nopvc-ic.txt"), transpose(d ./ d[:,end]))

# sub_data_path_pvc = projectdir("adni/data/new_new_data/pvc/UC-Berkeley-TAUPVC-6MM-Mar-30-2024-AB-Status.csv")
# alldf = CSV.read(sub_data_path_pvc, DataFrame)
# suvrnames = [ADNIDatasets.suvr_name.(dktnames); "CEREBRAL_WHITE_MATTER_SUVR"] # pvc
# d = Array(dropmissing(alldf[:, suvrnames]))
# writedlm(projectdir("data-pvc-wm.txt"), transpose(d ./ d[:,end]))

#posdf = filter(x -> x.STATUS == "POS", alldf)

posdf = filter(x -> x.AB_Status == 1, alldf)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in get_node_id.(cortex)]

ic_data = ADNIDataset(posdf, dktnames; min_scans=3, reference_region="INFERIORCEREBELLUM")
ic_data_with_wm = ADNIDataset(posdf, [dktnames; "ERODED_SUBCORTICALWM"; "CEREBELLUM_CORTEX"]; min_scans=3, reference_region="INFERIORCEREBELLUM")
wm_data = ADNIDataset(posdf, dktnames; min_scans=3, reference_region="ERODED_SUBCORTICALWM")
n_data = length(ic_data)

ic_ref_vols = [d ./ d[1] for d in get_ref_vol.(ic_data)]
wm_ref_vols = [d ./ d[1] for d in get_ref_vol.(wm_data)]

ic_vols = [d ./ d[:,1] for d in get_vol.(ic_data)]
wm_vols = [d ./ d[:,1] for d in get_vol.(wm_data)]

ic_gmm_moments = CSV.read(projectdir("adni/data/component_moments.csv"), DataFrame)
ic_mtl_cutoff = 1.375
ic_neo_cutoff = 1.395

ic_mtl_pos = filter(x -> regional_mean(ic_data, mtl, x) >= ic_mtl_cutoff, 1:n_data)
ic_neo_pos = filter(x -> regional_mean(ic_data, neo, x) >= ic_neo_cutoff, 1:n_data)

ic_tau_pos = findall(x -> x ∈ unique([ic_mtl_pos; ic_neo_pos]), 1:n_data)
ic_tau_neg = findall(x -> x ∉ ic_tau_pos, 1:n_data)

wm_gmm_moments = CSV.read(projectdir("py-analysis/wm-nopvc-moments-prob.csv"), DataFrame)
wm_mtl_cutoff = mean(wm_gmm_moments.cutoff[mtl])
wm_neo_cutoff = mean(wm_gmm_moments.cutoff[neo])

wm_mtl_pos = filter(x -> regional_mean(wm_data, mtl, x) >= wm_mtl_cutoff, 1:n_data)
wm_neo_pos = filter(x -> regional_mean(wm_data, neo, x) >= wm_neo_cutoff, 1:n_data)

wm_tau_pos = findall(x -> x ∈ unique([wm_mtl_pos; wm_neo_pos]), 1:n_data)
wm_tau_neg = findall(x -> x ∉ wm_tau_pos, 1:n_data)

negdf = filter(x -> x.AB_Status == 0, alldf)
ic_negdata = ADNIDataset(negdf, dktnames; min_scans=3, reference_region="INFERIORCEREBELLUM")
wm_negdata = ADNIDataset(negdf, dktnames; min_scans=3, reference_region="ERODED_SUBCORTICALWM")

ic_neg_ref_vols = [d ./ d[1] for d in get_ref_vol.(ic_negdata)]
wm_neg_ref_vols = [d ./ d[1] for d in get_ref_vol.(wm_negdata)]

function get_diff(x::Matrix{Float64})
    x[:,end] .- x[:,1]
end
function get_diff(x::Vector{Float64})
    x[end] - x[1]
end

begin
    f = Figure(size=(1000, 500))
    ax = Axis(f[1,1], xlabel="Inferior Cerbellum", ylabel="White Matter")
    CairoMakie.scatter!(mean(get_initial_conditions.(ic_data[ic_tau_pos])), mean(get_initial_conditions.(wm_data[ic_tau_pos])), label="A+T+")
    CairoMakie.scatter!(mean(get_initial_conditions.(ic_data[ic_tau_neg])), mean(get_initial_conditions.(wm_data[ic_tau_neg])), label="A+T-")
    CairoMakie.scatter!(mean(get_initial_conditions.(ic_negdata)), mean(get_initial_conditions.(wm_negdata)), label="A-")
    axislegend(ax, position=:rb)

    ax = Axis(f[1,2], xlabel="Inferior Cerbellum", ylabel="White Matter")
    CairoMakie.scatter!(mean(get_diff.(calc_suvr.(ic_data[ic_tau_pos]))), mean(get_diff.(calc_suvr.(wm_data[ic_tau_pos]))), label="A+T+")
    CairoMakie.scatter!(mean(get_diff.(calc_suvr.(ic_data[ic_tau_neg]))), mean(get_diff.(calc_suvr.(wm_data[ic_tau_neg]))), label="A+T-")
    CairoMakie.scatter!(mean(get_diff.(calc_suvr.(ic_negdata))), mean(get_diff.(calc_suvr.(wm_negdata))), label="A-")
    axislegend(ax, position=:rb)
    f
end


begin
    f = Figure(size=(1000, 500), fontsize=25)
    ax = Axis(f[1,1], xlabel="Group", ylabel="SUVR", xticks=(1:3, ["A+T+", "A+T-", "A-"]))
    Label(f[0, 1], "White Matter", tellwidth=false)
    ylims!(0.75, 1.8)
    rainclouds!(ones(72), mean(get_initial_conditions.(ic_data[ic_tau_pos])), whiskerwidth=1, markersize=5)
    rainclouds!(ones(72) .* 2, mean(get_initial_conditions.(ic_data[ic_tau_neg])), whiskerwidth=1, markersize=5)
    rainclouds!(ones(72) .* 3, mean(get_initial_conditions.(ic_negdata)), whiskerwidth=1, markersize=5)

    ax = Axis(f[1,2], xlabel="Group", ylabel="SUVR", xticks=(1:3, ["A+T+", "A+T-", "A-"]))
    Label(f[0, 2], "White Matter", tellwidth=false)
    ylims!(0.75, 1.8)
    rainclouds!(ones(72), mean(get_initial_conditions.(wm_data[wm_tau_pos])), whiskerwidth=1, markersize=5)
    rainclouds!(ones(72) .* 2, mean(get_initial_conditions.(wm_data[wm_tau_neg])), whiskerwidth=1, markersize=5)
    rainclouds!(ones(72) .* 3, mean(get_initial_conditions.(wm_negdata)), whiskerwidth=1, markersize=5)
    f
end
save("suvr-wm-vs-ic.pdf", f)

begin
    f = Figure(size=(1000, 500), fontsize=25)
    ax = Axis(f[1,1], xlabel="Group", ylabel="Δ SUVR", xticks=(1:3, ["A+T+", "A+T-", "A-"]))
    Label(f[0, 1], "Inferior Cerbellum", tellwidth=false)
    ylims!(-0.05, 0.25)
    rainclouds!(ones(72), mean(get_diff.(calc_suvr.(ic_data[ic_tau_pos]))), whiskerwidth=1, markersize=5)
    rainclouds!(ones(72) .* 2, mean(get_diff.(calc_suvr.(ic_data[ic_tau_neg]))), whiskerwidth=1, markersize=5)
    rainclouds!(ones(72) .* 3, mean(get_diff.(calc_suvr.(ic_negdata))), whiskerwidth=1, markersize=5)

    ax = Axis(f[1,2], xlabel="Group", ylabel="Δ SUVR", xticks=(1:3, ["A+T+", "A+T-", "A-"]))
    Label(f[0, 2], "White Matter", tellwidth=false)
    ylims!(-0.05, 0.25)
    rainclouds!(ones(72), mean(get_diff.(calc_suvr.(wm_data[wm_tau_pos]))), whiskerwidth=1, markersize=5)
    rainclouds!(ones(72) .* 2, mean(get_diff.(calc_suvr.(wm_data[wm_tau_neg]))), whiskerwidth=1, markersize=5)
    rainclouds!(ones(72) .* 3, mean(get_diff.(calc_suvr.(wm_negdata))), whiskerwidth=1, markersize=5)

    f
end
save("suvr-wm-vs-ic-delta.pdf", f)

cols = Makie.wong_colors();
begin
    f = Figure(size=(1000, 500), fontsize=20)
    ax = Axis(f[1,1], xlabel="Δ REF SUVR", ylabel="Δ MTL SUVR")
    Label(f[0,1], "A+T+", tellwidth=false, fontsize=30)
    xlims!(ax, -0.35, 0.35); ylims!(ax, -0.5, 0.5)
    for (d, ic) in zip(wm_data[wm_tau_pos], ic_data[wm_tau_pos])
        scatter!(get_diff(get_ref_suvr(d)), get_diff(vec(mean(calc_suvr(ic)[mtl,:], dims=1))), markersize=15, color=cols[1])
    end

    ax = Axis(f[1,2], xlabel="Δ REF SUVR", ylabel="Δ MTL SUVR")
    Label(f[0,2], "A+T-", tellwidth=false, fontsize=30)
    xlims!(ax, -0.35, 0.35); ylims!(ax, -0.5, 0.5)
    # for (d, vol) in zip(wm_data, wm_ref_vols)
    #     scatter!(get_diff(get_ref_suvr(d)), get_diff(vol), color=:blue, markersize=15)
    # end
    for (d, ic) in zip(wm_data[wm_tau_neg], ic_data[wm_tau_neg])
        scatter!(get_diff(get_ref_suvr(d)), get_diff(vec(mean(calc_suvr(ic)[mtl,:], dims=1))), markersize=15, color=cols[1])
    end
    f
end
save("suvr-wm-vs-mtl.pdf", f)

begin
    f = Figure(size=(1000, 500), fontsize=20)
    ax = Axis(f[1,1], xlabel="Δ REF SUVR", ylabel="Δ MTL SUVR")
    Label(f[0,1], "A+T+", tellwidth=false, fontsize=30)
    xlims!(ax, -0.35, 0.35); ylims!(ax, -0.5, 0.5)
    for (cg, ic) in zip(ic_data_with_wm[wm_tau_pos], ic_data[wm_tau_pos])
        scatter!(get_diff(calc_suvr(cg)[end,:]), get_diff(vec(mean(calc_suvr(ic)[mtl,:], dims=1))), markersize=15, color=cols[1])
    end

    ax = Axis(f[1,2], xlabel="Δ REF SUVR", ylabel="Δ MTL SUVR")
    Label(f[0,2], "A+T-", tellwidth=false, fontsize=30)
    xlims!(ax, -0.35, 0.35); ylims!(ax, -0.5, 0.5)
    # for (d, vol) in zip(wm_data, wm_ref_vols)
    #     scatter!(get_diff(get_ref_suvr(d)), get_diff(vol), color=:blue, markersize=15)
    # end
    for (cg, ic) in zip(ic_data_with_wm[wm_tau_neg], ic_data[wm_tau_neg])
        scatter!(get_diff(calc_suvr(cg)[end,:]), get_diff(vec(mean(calc_suvr(ic)[mtl,:], dims=1))), markersize=15, color=cols[1])
    end
    f
end
save("suvr-cg-vs-mtl.pdf", f)

begin
    f = Figure(size=(1000, 500), fontsize=20)
    ax = Axis(f[1,1], xlabel="Δ REF VOL", ylabel="Δ REF SUVR")
    Label(f[0,1], "A+T+", tellwidth=false, fontsize=30)
    xlims!(ax, -0.25, 0.1); ylims!(ax, -0.4, 0.4)
    ref_suvr = reduce(vcat, [get_diff(d[end,:]) for d in calc_suvr.(ic_data_with_wm[ic_tau_pos])])
    ref_vol = reduce(vcat, get_diff.([d[end,:] ./ d[end,1] for d in get_vol.(ic_data_with_wm[ic_tau_pos])]))
    scatter!(ref_vol, ref_suvr)

    ax = Axis(f[1,2], xlabel="Δ REF VOL", ylabel="Δ REF SUVR")
    xlims!(ax, -0.25, 0.1); ylims!(ax, -0.4, 0.4)
    Label(f[0,2], "A+T-", tellwidth=false, fontsize=30)
    # xlims!(ax, -0.35, 0.35); ylims!(ax, -0.5, 0.5)
    ref_suvr = reduce(vcat, [get_diff(d[end,:]) for d in calc_suvr.(ic_data_with_wm[ic_tau_neg])])
    ref_vol = reduce(vcat, get_diff.([d[end,:] ./ d[end,1] for d in get_vol.(ic_data_with_wm[ic_tau_neg])]))
    scatter!(ref_vol, ref_suvr)
    f
end
save("ic-suvr-vs-vol.pdf", f)

begin
    f = Figure(size=(1000, 500), fontsize=20)
    ax = Axis(f[1,1], xlabel="REF SUVR", ylabel="MTL SUVR")
    Label(f[0,1], "A+T+", tellwidth=false, fontsize=30)
    xlims!(ax, -0.35, 0.35); ylims!(ax, -0.5, 0.5)
    for (d, ic) in zip(wm_data[wm_tau_pos], ic_data[wm_tau_pos])
        scatter!(get_diff(get_ref_suvr(d)), get_diff(vec(mean(calc_suvr(ic)[mtl,:], dims=1))), markersize=15, color=cols[1])
    end

    ax = Axis(f[1,2], xlabel="REF SUVR", ylabel="MTLSUVR")
    Label(f[0,2], "A+T-", tellwidth=false, fontsize=30)
    xlims!(ax, -0.35, 0.35); ylims!(ax, -0.5, 0.5)
    # for (d, vol) in zip(wm_data, wm_ref_vols)
    #     scatter!(get_diff(get_ref_suvr(d)), get_diff(vol), color=:blue, markersize=15)
    # end
    for (d, ic) in zip(wm_data[wm_tau_neg], ic_data[wm_tau_neg])
        scatter!(get_diff(get_ref_suvr(d)), get_diff(vec(mean(calc_suvr(ic)[mtl,:], dims=1))), markersize=15, color=cols[1])
    end
    f
end

begin
    f = Figure(size=(1000, 500), fontsize=20)
    ax = Axis(f[1,1], xlabel="REF SUVR", ylabel="MTL SUVR")
    Label(f[0,1], "A+T+", tellwidth=false, fontsize=30)
    xlims!(ax, -0.35, 0.35); ylims!(ax, -0.5, 0.5)
    for (d, ic) in zip(wm_data[wm_tau_pos], ic_data[wm_tau_pos])
        scatter!(get_diff(get_ref_suvr(d)), get_diff(vec(mean(calc_suvr(ic)[mtl,:], dims=1))), markersize=15, color=cols[1])
    end

    ax = Axis(f[1,2], xlabel="REF SUVR", ylabel="MTLSUVR")
    Label(f[0,2], "A+T-", tellwidth=false, fontsize=30)
    xlims!(ax, -0.35, 0.35); ylims!(ax, -0.5, 0.5)
    # for (d, vol) in zip(wm_data, wm_ref_vols)
    #     scatter!(get_diff(get_ref_suvr(d)), get_diff(vol), color=:blue, markersize=15)
    # end
    for (d, ic) in zip(wm_data[wm_tau_neg], ic_data[wm_tau_neg])
        scatter!(get_diff(get_ref_suvr(d)), get_diff(vec(mean(calc_suvr(ic)[mtl,:], dims=1))), markersize=15, color=cols[1])
    end
    f
end

begin
    f = Figure(size=(1000, 500))
    ax = Axis(f[1,1], xlabel="CEREBELLAR VOL", ylabel="MTL VOL")
    for (group, col) in zip([ic_tau_pos, ic_tau_neg], [:red, :blue])
        for (d_vol, ref_vol) in zip(ic_vols[group], ic_ref_vols[group])
            scatter!(get_diff(ref_vol), get_diff(vec(mean(d_vol[mtl,:], dims=1))), color=col, markersize=15)
        end
    end
    ax = Axis(f[1,2], xlabel="WHITE MATTER VOL", ylabel="MTL VOL")
    for (group, col) in zip([wm_tau_pos, wm_tau_neg], [:red, :blue])
        for (d_vol, ref_vol) in zip(wm_vols[group], wm_ref_vols[group])
            scatter!(get_diff(ref_vol), get_diff(vec(mean(d_vol[mtl,:], dims=1))), color=col, markersize=15)
        end
    end
    f
end

begin
    f = Figure(size=(1000, 500))
    ax = Axis(f[1,1], xlabel="REF", ylabel="MTL")
    xlims!(ax, -0.35, 0.15); ylims!(ax, -0.25, 0.25)
    Label(f[0,1], "A+T+", tellwidth=false, fontsize=30)
    for (d, vol) in zip(ic_vols[ic_tau_pos], ic_ref_vols[ic_tau_pos])
        scatter!(get_diff(vol), get_diff(vec(mean(d[mtl,:], dims=1))), color=:red, markersize=15, label="Cerbellum")
    end

    for (d, vol) in zip(wm_vols[wm_tau_pos], wm_ref_vols[wm_tau_pos])
        scatter!(get_diff(vol), get_diff(vec(mean(d[mtl,:], dims=1))), color=:blue, markersize=15, label="White matter")
    end

    axislegend(ax, merge=true, position=:lt)
    ax = Axis(f[1,2], xlabel="REF", ylabel="MTL")
    Label(f[0,2], "A+T-", tellwidth=false, fontsize=30)
    xlims!(ax, -0.35, 0.15); ylims!(ax, -0.25, 0.25)
    for (d, vol) in zip(ic_vols[ic_tau_neg], ic_ref_vols[ic_tau_neg])
        scatter!(get_diff(vol), get_diff(vec(mean(d[mtl,:], dims=1))), color=:red, markersize=15)
    end

    for (d, vol) in zip(wm_vols[wm_tau_neg], wm_ref_vols[wm_tau_neg])
        scatter!(get_diff(vol), get_diff(vec(mean(d[mtl,:], dims=1))), color=:blue, markersize=15)
    end
    f
end
save("vol-ref-vs-mtl.pdf", f)

begin
    f = Figure(size=(1250, 500))
    ax = Axis(f[1,1], ylabel="Δ Volume")
    Label(f[0,1], "A+T+", tellwidth=false, fontsize=30)
    ic_ref_vol_diff = get_diff.(ic_ref_vols[ic_tau_pos])
    wm_ref_vol_diff = get_diff.(wm_ref_vols[wm_tau_pos])
    rainclouds!(ones(length(ic_ref_vol_diff)),      ic_ref_vol_diff, whiskerwidth=1, markersize=5)
    rainclouds!(ones(length(wm_ref_vol_diff)) .* 2, wm_ref_vol_diff, whiskerwidth=1, markersize=5)


    ax = Axis(f[1,2], ylabel="Δ Volume")
    Label(f[0,2], "A+T-", tellwidth=false, fontsize=30)
    ic_ref_vol_diff = get_diff.(ic_ref_vols[ic_tau_neg])
    wm_ref_vol_diff = get_diff.(wm_ref_vols[wm_tau_neg])
    rainclouds!(ones(length(ic_ref_vol_diff)),      ic_ref_vol_diff, whiskerwidth=1, markersize=5)
    rainclouds!(ones(length(wm_ref_vol_diff)) .* 2, wm_ref_vol_diff, whiskerwidth=1, markersize=5)

    ax = Axis(f[1,3], ylabel="Δ Volume")
    Label(f[0,3], "A-", tellwidth=false, fontsize=30)
    ic_ref_vol_diff = get_diff.(ic_neg_ref_vols)
    wm_ref_vol_diff = get_diff.(wm_neg_ref_vols)
    rainclouds!(ones(length(ic_ref_vol_diff)),      ic_ref_vol_diff, whiskerwidth=1, markersize=5)
    rainclouds!(ones(length(wm_ref_vol_diff)) .* 2, wm_ref_vol_diff, whiskerwidth=1, markersize=5)
    f
end
save("vol-wm-vs-mtl.pdf", f)