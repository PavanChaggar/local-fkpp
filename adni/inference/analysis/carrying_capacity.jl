using OrdinaryDiffEq 
using CSV, DataFrames
using Connectomes
using Serialization, Turing
using CairoMakie
using DrWatson

connectome_path = Connectomes.connectome_path()
all_c = filter(Connectome(connectome_path; norm=true, weight_function = (n, l) -> n ), 1e-2);

subcortex = filter(x -> get_lobe(x) == "subcortex", all_c.parc);
cortex = filter(x -> get_lobe(x) != "subcortex", all_c.parc);

c = slice(all_c, cortex) |> filter

L = laplacian_matrix(c)

gmm_moments = CSV.read(projectdir("adni/data/component_moments.csv"), DataFrame)
ubase, upath = get_dkt_moments(gmm_moments, dktnames)
u0 = mean.(ubase)
cc = quantile.(upath, .99)

function NetworkLocalFKPP(du, u, p, t; L = L, u0 = u0, cc = cc)
    du .= -p[1] * L * (u .- u0) .+ p[2] .* (u .- u0) .* ((cc .- u0) .- (u .- u0))
end

prob = ODEProblem(NetworkLocalFKPP, 
                  u0 .+ 0.1, 
                  (0.,10), 
                  [1.0,1.0])
                  
sol = solve(prob, Tsit5())


pst = mean(deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-taupos-4x2000.jls")));
pst2 = mean(deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-tauneg-4x2000.jls")));
pst3 = mean(deserialize(projectdir("adni/new-chains/local-fkpp/length-free/pst-abneg-4x2000.jls")));

ccs = Vector{Vector{Float64}}()
for p in [pst, pst2, pst3]
    prob = ODEProblem(NetworkLocalFKPP, 
                    u0 .+ 0.1, 
                    (0.,100), 
                    [p[:Pm , :mean], 0.1])
                    
    sol = solve(prob, Tsit5())
    push!(ccs, sol[end])
end

begin
    f = Figure(size=(600, 300))
    ax = Axis(f[1,1], 
              yticks=(1:3, [L"A^+T^+", L"A^+T^-", L"A^-"]), 
              yticklabelsize=20,
              xlabel="Carrying Capacity SUVR", xlabelsize=20)
    ylims!(ax, 0.5, 3.5)
    scatter!(ccs[1], ones(72) .* 1.0)
    scatter!(ccs[2], ones(72) .* 2.0)
    scatter!(ccs[3], ones(72) .* 3.0)
    f
end
save(projectdir("visualisation/models/output/carrying_capacities.pdf"), f)