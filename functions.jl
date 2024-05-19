using CSV, DataFrames, Distributions
function get_dkt_moments(gmm_moments, dktnames)
    μ_1 = Vector{Float64}()
    μ_2 = Vector{Float64}()

    σ_1 = Vector{Float64}()
    σ_2 = Vector{Float64}()
    
    for name in dktnames[1:end]
        roi = filter( x -> x.region == name, gmm_moments)
        push!(μ_1, roi.C0_mean[1])
        push!(μ_2, roi.C1_mean[1])
        push!(σ_1, sqrt(roi.C0_cov[1]))
        push!(σ_2, sqrt(roi.C1_cov[1]))
    end
    Normal.(μ_1, σ_1), Normal.(μ_2, σ_2)
end


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

function normalise!(data, lower)
    for i in 1:size(data, 1)
        lower_mask = data[i,:] .< lower[i]
        data[i, lower_mask] .= lower[i]
    end
end

function normalise(data, lower)
    _data = deepcopy(data)
    normalise!(_data, lower)
    _data
end

function normalise!(data, lower, upper)
    for i in 1:size(data, 1)
        lower_mask = data[i,:] .< lower[i]
        data[i, lower_mask] .= lower[i]
        upper_mask = data[i,:] .> upper[i]
        data[i, upper_mask] .= upper[i]
    end
end

function normalise(data, lower, upper)
    _data = deepcopy(data)
    normalise!(_data, lower, upper)
    _data
end

function regional_mean(data, rois, sub)
    subsuvr = calc_suvr(data, sub)
    mean(subsuvr[rois,end])
end

function regional_mean(data, rois, sub, scan)
    subsuvr = calc_suvr(data, sub)
    mean(subsuvr[rois,scan])
end