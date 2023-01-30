using Pkg
cd("/home/chaggar/Projects/local-fkpp")
Pkg.activate(".")

using Distributed
using Turing, DrWatson

addprocs(2)
file_path = projectdir("adni/inference/local-fkpp/taupos.jl")
@everywhere include($(file_path))

pst = sample(m, Turing.NUTS(0.8), MCMCDistributed(), 500, 2)
serialize(projectdir("adni/chains/local-fkpp/pst-taupos-distributed.jls"), pst)