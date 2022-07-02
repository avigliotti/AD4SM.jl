module AD4SM


# using LinearAlgebra, Printf
# using Distributed, SparseArrays
# using ProgressMeter, Dates#, StatsBase

include("adiff.jl")
include("materials.jl")
include("elements.jl")
include("solvers.jl")

export adiff, Materials, Elements, Solvers

VER = "0.0.3 candidate"
export VER



end # module
