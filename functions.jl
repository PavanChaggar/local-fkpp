using CSV, DataFrames, Distributions

function get_stats(df, dktnames)
    norm = Vector{Float64}()
    path = Vector{Float64}()
    for roi in dktnames[1:end-1]
        roidf = filter(x -> x.region == roi, df)
        push!(norm, roidf.norm_mean_SUVR[1])
        push!(path, roidf.path_95CI_SUVR[1])
    end
    norm, path
end

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

function make_idata(m, pst, data, args...)
    # m_predict = f(fill(missing, length(vec(data))), args...)
    # pst_pred = predict(m_predict, pst)
    
    chains_params = Turing.MCMCChains.get_sections(pst, :parameters)
    loglikelihoods = pointwise_loglikelihoods(m, chains_params)
    #nms = string.(keys(pst_pred))
    nms = keys(loglikelihoods)
    loglikelihoods_vals = getindex.(Ref(loglikelihoods), nms)
    n_samples, n_chains = size(pst[:n_steps])
    loglikelihoods_arr = Array{Float64}(undef, n_chains, n_samples, length(data))
    for j in 1:n_chains
        for i in 1:length(data)
            loglikelihoods_arr[j,:,i] .= loglikelihoods_vals[i]
        end
    end
    from_mcmcchains(pst;
                    # posterior_predictive=pst_pred,
                    log_likelihood=Dict("ll" => loglikelihoods_arr),
                    library="Turing",
                    observed_data=Dict("data" => data))
end


function get_loglikelihoods(f, m, pst, data, args...)
    m_predict = f(fill(missing, length(vec(data))), args...)
    pst_pred = predict(m_predict, pst)
    
    chains_params = Turing.MCMCChains.get_sections(pst, :parameters)
    loglikelihoods = pointwise_loglikelihoods(m, chains_params)
    nms = string.(keys(pst_pred))
    loglikelihoods_vals = getindex.(Ref(loglikelihoods), nms)
    loglikelihoods_arr = permutedims(cat(loglikelihoods_vals...; dims=3), (2, 1, 3));
    vec(sum(loglikelihoods_arr, dims=3))
end

function get_number_of_params(model)
    n = 0
    vi = Turing.VarInfo(model)
    for v in vi.metadata
        n += length(v.vals)
    end
    n
end

function calc_aic(f, m, pst, data, args...)
    ll = get_loglikelihoods(f, m, pst, data, args...)
    np = get_number_of_params(m)
    2np - 2 * maximum(ll)
end


function normalise!(data, cutoff)
    for i in 1:size(data, 1)
        mask = data[i,:] .< cutoff[i]
        data[i, mask] .= cutoff[i]
    end
end

function normalise(data, cutoff)
    _data = deepcopy(data)
    normalise!(_data, cutoff)
    _data
end

function regional_mean(data, rois, sub)
    subsuvr = calc_suvr(data, sub)
    mean(subsuvr[rois,end])
end