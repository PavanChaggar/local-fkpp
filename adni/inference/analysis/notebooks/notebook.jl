### A Pluto.jl notebook ###
# v0.19.20

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 5a1e2ffa-9a8b-11ed-330a-6b360d663ddb
begin
	using Pkg
	Pkg.activate("/Users/pavanchaggar/Projects/local-fkpp/")
end

# ╔═╡ 5d463896-c3d8-4a7e-b3ff-59da170fda03
begin
	using Connectomes
	using ADNIDatasets
	using CSV, DataFrames
	using DrWatson: projectdir
	using DifferentialEquations
	using Distributions
	using Serialization
	using LinearAlgebra
	using Random
	using LinearAlgebra
	using MCMCChains
	using PlutoUI
end

# ╔═╡ e8333ba1-f967-43c0-b4bc-c89a189e5ea2
using CairoMakie; CairoMakie.activate!()

# ╔═╡ a56c1d2d-c02a-4bd8-8653-1f3df706dbd4
begin
	connectome_path = Connectomes.connectome_path()
	all_c = filter(Connectome(connectome_path; norm=true), 1e-2);
	
	subcortex = filter(x -> x.Lobe == "subcortex", all_c.parc);
	cortex = filter(x -> x.Lobe != "subcortex", all_c.parc);
	
	c = slice(all_c, cortex) |> filter
	
	mtl_regions = ["entorhinal", "Left-Amygdala", "Right-Amygdala"]
	mtl = findall(x -> x ∈ mtl_regions, cortex.Label)
	neo_regions = ["inferiortemporal", "middletemporal"]
	neo = findall(x -> x ∈ neo_regions, cortex.Label)
end;

# ╔═╡ 212a1707-2726-41cc-bc04-8cbf73392bb9
begin
	sub_data_path = projectdir("adni/data/AV1451_Diagnosis-STATUS-STIME-braak-regions.csv");
	alldf = CSV.read(sub_data_path, DataFrame)
	
	posdf = filter(x -> x.STATUS == "POS", alldf)
	
	dktdict = Connectomes.node2FS()
	dktnames = [dktdict[i] for i in cortex.ID]
	
	data = ADNIDataset(posdf, dktnames; min_scans=2, max_scans=2)
	
	# Ask Jake where we got these cutoffs from? 
	mtl_cutoff = 1.375
	neo_cutoff = 1.395
	
	function regional_mean(data, rois, sub)
	    subsuvr = calc_suvr(data, sub)
	    mean(subsuvr[rois,1])
	end
	
	mtl_pos = filter(x -> regional_mean(data, mtl, x) >= mtl_cutoff, 1:50)
	neo_pos = filter(x -> regional_mean(data, neo, x) >= neo_cutoff, 1:50)
	
	tau_pos = findall(x -> x ∈ unique([mtl_pos; neo_pos]), 1:50)
	tau_neg = findall(x -> x ∉ tau_pos, 1:50)
	
	n_pos = length(tau_pos)
	n_neg = length(tau_neg)
end

# ╔═╡ ab10d78c-9182-4ff9-bd47-e8cb728a5497
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

# ╔═╡ 9816f6a6-cd70-4eea-a806-52f8a35fbf68
begin
	gmm_moments = CSV.read(projectdir("adni/data/component_moments.csv"), DataFrame)
	ubase, upath = get_dkt_moments(gmm_moments, dktnames)
	u0 = mean.(ubase)
	cc = quantile.(upath, .99)
end;

# ╔═╡ 5cefaa15-4f05-4429-9b8d-9af6534e6b03
function normalise!(data, lower, upper)
    for i in 1:size(data, 1)
        lower_mask = data[i,:] .< lower[i]
        data[i, lower_mask] .= lower[i]
        upper_mask = data[i,:] .> upper[i]
        data[i, upper_mask] .= upper[i]
    end
end

# ╔═╡ 365da038-da05-484d-9096-f75d61a694c4
begin
	L = laplacian_matrix(c)
	
	function NetworkLocalFKPP(du, u, p, t; L = L, u0 = u0, cc = cc)
	    du .= -p[1] * L * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
	end
	
	subdata = [calc_suvr(data, i) for i in tau_pos]
	for i in 1:n_pos
	    normalise!(subdata[i], u0, cc)
	end
	
	initial_conditions = [sd[:,1] for sd in subdata]
	times =  [get_times(data, i) for i in tau_pos]
	
	prob = ODEProblem(NetworkLocalFKPP, 
	                  initial_conditions[1], 
	                  (0.,10.), 
	                  [1.0,1.0])
	                  
	sol = solve(prob, Tsit5())
end;

# ╔═╡ 62ecebf4-522b-4e8e-8d29-ee166fda917e
begin
	pst = deserialize(projectdir("adni/chains/local-fkpp/pst-taupos-4x2000.jls"));
	
	meanpst = mean(pst)
	
	Pm, Am = meanpst[:Pm, :mean], meanpst[:Am, :mean]
	
	
	probs = [ODEProblem(NetworkLocalFKPP, initial_conditions[i], (0.,5.), [Pm, Am]) for i in 1:n_pos];
	sols = [solve(probs[i], Tsit5(), saveat=times[i]) for i in 1:n_pos];
end;

# ╔═╡ 040f92b5-2b3f-4a5e-a4db-68ec3e12b4fb
begin
	f = Figure(resolution=(600, 500))
	ax = Axis(f[1,1], 
	          xlabel="SUVR", 
	          ylabel="Prediction", 
	          titlesize=26, xlabelsize=20, ylabelsize=20)
	xlims!(ax, 0.9, 2.5)
	ylims!(ax, 0.9, 2.5)
	lines!(0.9:0.1:2.5, 0.9:0.1:2.5, color=(:grey, 0.75), linewidth=2, linestyle=:dash)

	for i in 1:n_pos
	    scatter!(subdata[i][:,2], sols[i][2], color=(:grey, 0.3))
	end
f
end

# ╔═╡ 8b131e58-e642-4756-9892-1721cc4e6db7
function get_diff(d)
    d[:,end] .- d[:,1]
end

# ╔═╡ 84d8872f-9b17-4259-9550-e1b1ffcb418c
begin
	f2 = Figure(resolution=(600, 500))
	ax2 = Axis(f2[1,1], 
	          xlabel="δ SUVR", 
	          ylabel="δ Prediction", 
	          titlesize=26, xlabelsize=20, ylabelsize=20)
	xlims!(ax2, -0.25, 0.5)
	ylims!(ax2, -0.25, 0.5)
	for i in 1:n_pos
	    scatter!(get_diff(subdata[i]), get_diff(sols[i]), color=(:grey, 0.2))
	end
	f2
end

# ╔═╡ 994e5a72-be90-4d62-b61a-bde998d9a221
md"#### subject $(sub = @bind sub PlutoUI.Slider(1:n_pos, show_value=true))"

# ╔═╡ d5e8b4c0-3dd8-49a8-be8a-3597ba29e402
begin
	f3= Figure(resolution=(1000, 500))
	ax3 = Axis(f3[1,1], 
	          xlabel="SUVR", 
	          ylabel="Prediction", 
	          titlesize=26, xlabelsize=20, ylabelsize=20)
	xlims!(ax3, 0.9, 2.5)
	ylims!(ax3, 0.9, 2.5)
	lines!(0.9:0.1:2.5, 0.9:0.1:2.5, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
	scatter!(subdata[sub][:,2], sols[sub][2], color=(:grey, 0.7))
	ax4 = Axis(f3[1,2], 
	          xlabel="δ SUVR", 
	          ylabel="δ Prediction", 
	          titlesize=26, xlabelsize=20, ylabelsize=20)
	xlims!(ax4, -0.25, 0.5)
	ylims!(ax4, -0.25, 0.5)
	lines!(-0.25:0.1:0.5, -0.25:0.1:0.5, color=(:grey, 0.75), linewidth=2, linestyle=:dash)

	scatter!(get_diff(subdata[sub]), get_diff(sols[sub]), color=(:grey, 0.7))
	f3
end

# ╔═╡ d5261194-3a63-4a49-aa4c-4e3d11e4fbd3
md"#### node $(@bind node PlutoUI.Slider(1:72, show_value=true))"

# ╔═╡ 0d78d8d0-a539-4839-b01b-a3f0f4aace22
begin
	f6= Figure(resolution=(1000, 500))
	ax6 = Axis(f6[1,1], 
	          xlabel="SUVR", 
	          ylabel="Prediction", 
	          titlesize=26, xlabelsize=20, ylabelsize=20)
	xlims!(ax6, 0.9, 2.5)
	ylims!(ax6, 0.9, 2.5)
	lines!(0.9:0.1:2.5, 0.9:0.1:2.5, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
	node_suvr = [subdata[i][node,2] for i in 1:n_pos]
	sol_suvr = [sols[i][node,2] for i in 1:n_pos]
	scatter!(node_suvr, sol_suvr, color=(:grey, 0.7))
	ax7 = Axis(f6[1,2], 
	          xlabel="δ SUVR", 
	          ylabel="δ Prediction", 
	          titlesize=26, xlabelsize=20, ylabelsize=20)
	xlims!(ax7, -0.25, 0.5)
	ylims!(ax7, -0.25, 0.5)
	lines!(-0.25:0.1:0.5, -0.25:0.1:0.5, color=(:grey, 0.75), linewidth=2, linestyle=:dash)
	node_diffs = [get_diff(subdata[sub])[node] for sub in 1:n_pos]
	sol_diffs = [get_diff(sols[sub])[node] for sub in 1:n_pos]
	scatter!(node_diffs, sol_diffs , color=(:grey, 0.7))
	f6
end

# ╔═╡ db66ee10-446a-43dd-9383-ee74110cee36
c.parc.Label[node]

# ╔═╡ Cell order:
# ╠═5a1e2ffa-9a8b-11ed-330a-6b360d663ddb
# ╠═5d463896-c3d8-4a7e-b3ff-59da170fda03
# ╠═a56c1d2d-c02a-4bd8-8653-1f3df706dbd4
# ╠═212a1707-2726-41cc-bc04-8cbf73392bb9
# ╠═ab10d78c-9182-4ff9-bd47-e8cb728a5497
# ╠═9816f6a6-cd70-4eea-a806-52f8a35fbf68
# ╠═5cefaa15-4f05-4429-9b8d-9af6534e6b03
# ╠═365da038-da05-484d-9096-f75d61a694c4
# ╠═62ecebf4-522b-4e8e-8d29-ee166fda917e
# ╠═e8333ba1-f967-43c0-b4bc-c89a189e5ea2
# ╠═040f92b5-2b3f-4a5e-a4db-68ec3e12b4fb
# ╠═8b131e58-e642-4756-9892-1721cc4e6db7
# ╟─84d8872f-9b17-4259-9550-e1b1ffcb418c
# ╠═d5e8b4c0-3dd8-49a8-be8a-3597ba29e402
# ╠═994e5a72-be90-4d62-b61a-bde998d9a221
# ╟─0d78d8d0-a539-4839-b01b-a3f0f4aace22
# ╟─d5261194-3a63-4a49-aa4c-4e3d11e4fbd3
# ╟─db66ee10-446a-43dd-9383-ee74110cee36
