__precompile__()

module Materials

using LinearAlgebra 
using ..adiff


# union of all materials
Material = Union{}
Mat3D    = Union{}
Mat2D    = Union{}
Mat1D    = Union{}

dims(mat::M) where M<:Mat3D = 3
dims(mat::M) where M<:Mat2D = 2
dims(mat::M) where M<:Mat1D = 1

# support for elastic materials
include("elasticmaterials.jl")

# support for phase field materials
include("phasefieldmaterials.jl")


export Hooke,Hooke1D,Hooke2D,MooneyRivlin,NeoHooke,Ogden,PhaseField
export getϕ

end
