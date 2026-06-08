
export makeϕrKt_d

include("./phasefieldelements.1stord.jl")
include("./phasefieldelements.2ndord.jl")

# functions for phase field
#
# get the average of d over the element
function getd(elem::CPElem{P}, d0::Vector{T}) where {P,T}
  d       = zero(T)
  for ii=1:P
    d += elem.wgt[ii]*(elem.N[ii]⋅d0)
  end  
  d/elem.V
end
#
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
function makeϕrKt(elems::Vector{<:CPElem{D,P,M,S,N}} where {D,P,M,S}, u::Array{T}, d::Array{T}) where {N,T}
  nElems = length(elems)
  M      = (N+1)N÷2

  Φ = Vector{adiff.D2{N,M,T}}(undef, nElems)
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
#= getδϕu
function getδϕu(elem::C3DP{P,<:PhaseField}, u0::Array{T}, d0::Array{T})  where {P,T}

  u, v, w = u0[1:3:end], u0[2:3:end], u0[3:3:end]
  N       = lastindex(u0)  
  wgt     = elem.wgt
  val     = zero(T)
  grad    = zeros(T,N)
  hess    = zeros(T,(N+1)N÷2)
  δF      = zeros(T,N,9)

  for ii=1:P
    N,Nx,Ny,Nz = elem.N[ii],elem.Nx[ii],elem.Ny[ii],elem.Nz[ii]
    δF[1:3:N,1] = δF[2:3:N,2] = δF[3:3:N,3] = Nx
    δF[1:3:N,4] = δF[2:3:N,5] = δF[3:3:N,6] = Ny
    δF[1:3:N,7] = δF[2:3:N,8] = δF[3:3:N,9] = Nz

    F    = [Nx⋅u Ny⋅u Nz⋅u;
            Nx⋅v Ny⋅v Nz⋅v;
            Nx⋅w Ny⋅w Nz⋅w ] + I
    d    = N⋅d0
    ∇d   = [Nx⋅d0, Ny⋅d0, Nz⋅d0]
    ϕ    = getϕ(adiff.D2(F), d, ∇d, elem.mat)::adiff.D2{9, 45, T}
    val += wgt[ii]ϕ.v
    for jj=1:9,i1=1:N
      grad[i1] += wgt[ii]*ϕ.g[jj]*δF[i1,jj]
      for kk=1:9,i2=1:i1
        hess[(i1-1)i1÷2+i2] += wgt[ii]*ϕ.h[jj,kk]*δF[i1,jj]*δF[i2,kk]
      end   
    end
  end

  adiff.D2(val, adiff.Grad(grad), adiff.Grad(hess))
end
=#
# getδϕd(elem::C3Ds{P,<:PhaseField}, u0::Array, d0::Array) where P = getϕ(elem, u0, adiff.D2(d0))
# getδϕd(elem::C2Ds{P,<:PhaseField}, u0::Array, d0::Array) where P = getϕ(elem,u0,adiff.D2(d0))
# getδϕd(elem::Rod{<:PhaseField}, u0::Array, d0::Array)                = getϕ(elem,u0,adiff.D2(d0))
# getδϕu(elem::Rod{<:PhaseField}, u0::Array, d0::Array)               = getϕ(elem,adiff.D2(u0),d0)
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

