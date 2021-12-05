module AD4SM


using LinearAlgebra, Printf
using Distributed, SparseArrays
using ProgressMeter, Dates#, StatsBase

include("adiff.jl")
include("materials.jl")
include("elements.jl")
include("solvers.jl")

export adiff, Materials, Elements, Solvers

VER = "0.0.2"
export VER

# using .adiff, .Materials, .Elements 
using .adiff, .Materials, .Elements 


end # module
