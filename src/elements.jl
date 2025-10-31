__precompile__()

module Elements

using ..AD4SM.adiff
import ..Materials.getϕ

using StaticArrays, SparseArrays
using LinearAlgebra:I
import LinearAlgebra.×

export CElem, CEElem, CPElem, C1DE, C2DE, C3DE, C1DP, C2DP, C3DP,
       C1D, C2D, C3D
export getϕ, getσ, getP, detJ, getF, ×, getV



# ---------------------------------------------------------------------------
# Inline static dot product
# ---------------------------------------------------------------------------

@inline function dot(a::SVector{N,S}, b::SVector{N,T}) where {N,S,T}
    s = zero(T)
    @inbounds @simd for ii in 1:N
        s += a[ii] * b[ii]
    end
    return s
end

const ⋅ = dot
# ---------------------------------------------------------------------------
# Abstract type hierarchy
# ---------------------------------------------------------------------------

abstract type AbstractElement end
abstract type AbstractContinuumElem <: AbstractElement end
abstract type AbstractCElem{D,P,M,T,N} <: AbstractContinuumElem end

# ---------------------------------------------------------------------------
# CEElem — Mechanical elements
# ---------------------------------------------------------------------------

"""
CEElem{D,P,M,T,N}

Generic D-dimensional continuum finite element for displacement-based
mechanical analysis.

Type parameters:
- `D` : spatial dimension (1, 2, or 3)
- `P` : number of quadrature points
- `M` : material model type
- `T` : numeric type
- `N` : number of element nodes

Fields:
- `nodes` : vector of nodal IDs (any integer type)
- `∇N`    : NTuple{D,NTuple{P,SVector{N,T}}} — shape-function derivatives
- `wgt`   : NTuple{P,T} — quadrature weights
- `V`     : reference volume/area/length
- `mat`   : material model
"""
struct CEElem{D,P,M,T,N} <: AbstractCElem{D,P,M,T,N}
    nodes::Vector{I} where I
    ∇N::NTuple{D,NTuple{P,SVector{N,T}}}
    wgt::NTuple{P,T}
    V::T
    mat::M
end

# ---------------------------------------------------------------------------
# CPElem — Phase-field / scalar-field elements
# ---------------------------------------------------------------------------

"""
CPElem{D,P,M,T,N}

Generic D-dimensional continuum finite element for scalar fields
(e.g., phase, concentration, or temperature).

Type parameters:
- `D` : spatial dimension (1, 2, or 3)
- `P` : number of quadrature points
- `M` : material model type
- `T` : numeric type
- `N` : number of element nodes

Fields:
- `nodes` : vector of nodal IDs (any integer type)
- `N`     : NTuple{P,SVector{N,T}} — shape-function values
- `∇N`    : NTuple{D,NTuple{P,SVector{N,T}}} — shape-function derivatives
- `wgt`   : NTuple{P,T} — quadrature weights
- `V`     : reference volume/area/length
- `mat`   : material model
"""
struct CPElem{D,P,M,T,N} <: AbstractCElem{D,P,M,T,N}
    nodes::Vector{I} where I
    N::NTuple{P,SVector{N,T}}
    ∇N::NTuple{D,NTuple{P,SVector{N,T}}}
    wgt::NTuple{P,T}
    V::T
    mat::M
end

# ---------------------------------------------------------------------------
# Type aliases for convenience
# ---------------------------------------------------------------------------

const C1DE{P,M,T,N}  = CEElem{1,P,M,T,N}
const C2DE{P,M,T,N}  = CEElem{2,P,M,T,N}
const C3DE{P,M,T,N}  = CEElem{3,P,M,T,N}

const C1DP{P,M,T,N} = CPElem{1,P,M,T,N}
const C2DP{P,M,T,N} = CPElem{2,P,M,T,N}
const C3DP{P,M,T,N} = CPElem{3,P,M,T,N}

const C1D = Union{C1DE,C1DP}
const C2D = Union{C2DE,C2DP}
const C3D = Union{C3DE,C3DP}

# const CElem{D,P,M,T,N} = Union{CEElem{D,P,M,T,N}, CPElem{D,P,M,T,N}}
const CElem = AbstractCElem

# ---------------------------------------------------------------------------
# Constructors for CEElem
# ---------------------------------------------------------------------------

"""
Create a 1D mechanical element.
"""
function C1DE(nodes, Nx, wgt, V, mat)
    P  = length(wgt)
    Nn = length(Nx[1])
    return C1DE{P,typeof(mat),eltype(wgt),Nn}(
        nodes,
        (ntuple(ii -> SVector{Nn}(Nx[ii]), P),),
        wgt, V, mat)
end

"""
Create a 2D mechanical element.
"""
function C2DE(nodes, Nx, Ny, wgt, V, mat)
    P  = length(wgt)
    Nn = length(Nx[1])
    return C2DE{P,typeof(mat),eltype(wgt),Nn}(
        nodes,
        ( ntuple(ii -> SVector{Nn}(Nx[ii]), P),
          ntuple(ii -> SVector{Nn}(Ny[ii]), P) ),
        wgt, V, mat)
end

"""
Create a 3D mechanical element.
"""
function C3DE(nodes, Nx, Ny, Nz, wgt, V, mat)
    P = length(wgt)
    Nn = length(Nx[1])
    return C3DE{P,typeof(mat),eltype(wgt),Nn}(
        nodes,
        ( ntuple(ii -> SVector{Nn}(Nx[ii]), P),
          ntuple(ii -> SVector{Nn}(Ny[ii]), P),
          ntuple(ii -> SVector{Nn}(Nz[ii]), P) ),
        wgt, V, mat)
end

# ---------------------------------------------------------------------------
# Constructors for CPElem
# ---------------------------------------------------------------------------

"""
Create a 1D scalar-field element.
"""
function C1DP(nodes, N, Nx, wgt, V, mat)
    P = length(wgt)
    Nn = length(N[1])
    return CPElem{1,P,typeof(mat),eltype(wgt),Nn}(
        nodes,
        ntuple(ii -> SVector{Nn}(N[ii]), P),
        (ntuple(ii -> SVector{Nn}(Nx[ii]), P),),
        wgt, V, mat)
end

"""
Create a 2D scalar-field element.
"""
function C2DP(nodes, N, Nx, Ny, wgt, V, mat)
    P = length(wgt)
    Nn = length(N[1])
    return CPElem{2,P,typeof(mat),eltype(wgt),Nn}(
        nodes,
        ntuple(ii -> SVector{Nn}(N[ii]), P),
        ( ntuple(ii -> SVector{Nn}(Nx[ii]), P),
          ntuple(ii -> SVector{Nn}(Ny[ii]), P) ),
        wgt, V, mat)
end

"""
Create a 3D scalar-field element.
"""
function C3DP(nodes, N, Nx, Ny, Nz, wgt, V, mat)
    P = length(wgt)
    Nn = length(N[1])
    return CPElem{3,P,typeof(mat),eltype(wgt),Nn}(
        nodes,
        ntuple(ii -> SVector{Nn}(N[ii]), P),
        ( ntuple(ii -> SVector{Nn}(Nx[ii]), P),
          ntuple(ii -> SVector{Nn}(Ny[ii]), P),
          ntuple(ii -> SVector{Nn}(Nz[ii]), P) ),
        wgt, V, mat)
end

include("./elements.toolkit.jl")
include("./elasticelements.jl")
include("./phasefieldelements.jl")

end # module Elements
