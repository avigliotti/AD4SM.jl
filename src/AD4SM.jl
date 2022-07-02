module AD4SM

<<<<<<< HEAD
=======

using LinearAlgebra, Printf
using Distributed, SparseArrays
using ProgressMeter, Dates#, StatsBase
>>>>>>> candidate

include("adiff.jl")
include("materials.jl")
include("elements.jl")
include("solvers.jl")

export adiff, Materials, Elements, Solvers

VER = "0.0.3"
export VER



end # module
