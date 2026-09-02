
export makeϕrKt_d

include("./phasefieldelements.1stord.jl")
include("./phasefieldelements.2ndord.jl")

# get free energy density for the elment without history
function getϕ(elem::CPElem{D,P,M,T,N} where {D,M,T}, u0::AbstractArray, d0::AbstractArray) where {P,N}

  d0= SVector{N}(d0)
  ϕ = 0
  @inline for ii=1:P
    F    = getF(elem, u0, ii)
    d,∇d = get_d_and_∇d(elem,d0,ii)
    ϕ   += elem.wgt[ii]getϕ(F, d, ∇d, elem.mat)
  end
  ϕ
end
# get free energy density for the elment with history
function getϕ(elem::CPElem{D,P,M,T,N}, u0::AbstractArray, d0::AbstractArray, ϕmax::Vector) where {D,P,M,T,N}

  u0  = SMatrix{D,N}(u0)
  d0  = SVector{N}(d0)
  wgt = elem.wgt
  ϕ   = 0
  @inline for ii=1:P
    F    = getF(elem, u0, ii)
    d,∇d = get_d_and_∇d(elem,d0,ii)
    ϕii,ϕmax[ii] = getϕ(F, d, ∇d, elem.mat, ϕmax[ii])
    ϕ   += wgt[ii]ϕii
  end
  ϕ
end
#
# calling getϕ with dual numbers on 3D elements
#
# these functions are optimized in case getϕ is called with a dual type for 
# the displacement field trough the use of the × operators for the chain 
# derivative, the other use the standard implementation common for all
# on newer CPU this might disppear
#
# without history
"""
    getϕ(elem::C3DP, u0::AbstractArray{D}, d0::AbstractArray) where D<:adiff.D2

Optimized 3D phase-field free-energy evaluation using local 3×3 kinematics at
Gauss-point level and the `×` operator for chain-rule propagation with respect
to the displacement DOFs.
"""
function getϕ(elem::C3DP{P,M,T,N} where {M,T}, u0::AbstractArray{D}, d0::AbstractArray) where {P,N,D<:adiff.D2}

  u0  = SMatrix{3,N}(adiff.D1.(u0))
  d0  = SVector{N}(d0)
  ϕ  = zero(D) 
  @inline for ii=1:P
    F    = getF(elem, u0, ii)
    d,∇d = get_d_and_∇d(elem,d0, ii)
    valF = adiff.val.(F)
    δϕ   = getϕ(adiff.D2(valF), d, ∇d, elem.mat)
    ϕ   += elem.wgt[ii] * (δϕ × F)
  end
  ϕ
end
#
# with history
"""
    getϕ(elem::C3DP, u0::Array{D}, d0::Array, ϕmax::Array) where D<:adiff.D2

Optimized 3D phase-field free-energy evaluation with history, using local 3×3
kinematics at Gauss-point level and the `×` operator for chain-rule
propagation with respect to the displacement DOFs.
"""
function getϕ(elem::C3DP{P,<:Any,<:Any,N}, u0::Array{D}, d0::Array, ϕmax::Array) where {P,N,D<:adiff.D2}

  u0  = SMatrix{3,N}(adiff.D1.(u0))
  d0  = SVector{N}(d0)
  ϕ  = zero(D) 
  @inline for ii=1:P
    F    = getF(elem, u0, ii)
    d,∇d = get_d_and_∇d(elem,d0, ii)
    valF = adiff.val.(F)
    δϕ,ϕmax[ii] = getϕ(adiff.D2(valF), d, ∇d, elem.mat, ϕmax[ii])
    ϕ   += elem.wgt[ii] * (δϕ × F)
  end
  ϕ
end
#
# 
# functions for array of elements
# 
function makeϕrKt(elems::Vector{<:CPElem{D,P,M,S,N}} where {P,M,S}, u::Array{T}, d::Array{T}) where {D,N,T}
  nElems = length(elems)
  G      = D*N      # number of gradient components
  H      = (G+1)G÷2 # number of hessian components

  Φ = Vector{adiff.D2{G,H,T}}(undef, nElems)
  Threads.@threads for ii=1:nElems
    Φ[ii] = getϕ(elems[ii], adiff.D2(u[:,elems[ii].nodes]), d[elems[ii].nodes])
  end

  makeϕrKt(Φ, elems, u)
end
function makeϕrKt_d(elems::Vector{<:CPElem{D,P,M,S,N}} where {D,P,M,S}, u::Array{T}, d::Array{T}) where {N,T}
  nElems = length(elems)
  M      = (N+1)N÷2

  Φ = Vector{adiff.D2{N,M,T}}(undef, nElems)
  Threads.@threads for ii=1:nElems
    nodes = elems[ii].nodes
    Φ[ii] = getϕ(elems[ii], u[:,elems[ii].nodes], adiff.D2(d[elems[ii].nodes]))
  end
  makeϕrKt(Φ, elems, d)
end
# with history
function makeϕrKt_d(elems::Vector{<:CPElem{D,P,M,S,N}} where {D,P,M,S}, u::Array{T}, d::Array{T}, ϕmax::Array) where {N,T}

  nElems = length(elems)
  M      = (N+1)N÷2

  Φ = Vector{adiff.D2{N,M,T}}(undef, nElems)
  Threads.@threads for ii=1:nElems
    nodes = elems[ii].nodes
    Φ[ii] = getϕ(elems[ii], u[:,elems[ii].nodes], adiff.D2(d[elems[ii].nodes]), ϕmax[ii])
  end
  makeϕrKt(Φ, elems, d)
end
#
function getd(elem::CPElem{<:Any,P}, d0::Array{T}) where {P,T}
  d       = zero(T)
  for ii=1:P
    d += elem.wgt[ii]*(elem.N[ii]⋅d0)
  end  
  d/elem.V
end
# getVd
function getVd(elem::CPElem{D,P,M,S,N} where {D,M,S}, d0::Array{T}) where {T,P,N}
  d0 = SVector{N}(d0)
  Vd = zero(T)
  for ii=1:P
    Vd += elem.wgt[ii]elem.N[ii]⋅d0
  end
  Vd
end
function getVd(elems::Vector{<:CPElem}, d::Array{T}) where T
  Vd = zero(T)
  for elem in elems
    Vd += getVd(elem, d[elem.nodes])
  end
  Vd
end

