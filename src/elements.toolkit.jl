# ---------------------------------------------------------------------------
# Mechanical evaluation functions
# ---------------------------------------------------------------------------

"""
Compute deformation gradient F = I + ∇u at Gauss point `ii`.
"""
@inline getF(elem::CElem, u::Matrix, ii::Integer) = get∇u(elem,u,ii) + I
@inline getF(elem::CElem, u::Matrix)              = get∇u(elem,u) + I
@inline function get∇u(elem::CElem{D,P,M,S,N} where S, u::Matrix{T}, ii::Integer) where {D,P,M,T,N}
    ∇u = @MMatrix zeros(T, D, D)
    u = SMatrix{D,N}(u[1:D,:])
    @inbounds for jj in 1:D
        for kk in 1:D
          ∇u[jj, kk] = elem.∇N[kk][ii]⋅u[kk,:]
        end
    end
    return SMatrix{D,D,T}(∇u)
end
function get∇u(elem::CElem{D,P,M,T,N}, u::Matrix{T}) where {D,P,M,T,N}
    ∇u = @MMatrix zeros(T, D, D)
    for ii=1:P
      ∇u .+= get∇u(elem, u, ii)
    end
    return SMatrix{D,D,T}(∇u/P)
end
get∇u(elems::Array{<:CElem}, u::Matrix) = [get∇u(elem, u[:,elem.nodes]) for elem in elems]
getF(elems::Array{<:CElem}, u::Matrix)  = [getF(elem, u[:,elem.nodes]) for elem in elems]

"""
Compute determinant of deformation gradient J = det(F).
"""
@inline detJ(F::SMatrix{2,2,T}) where {T} = F[1,1]F[2,2] - F[1,2]F[2,1]
@inline detJ(F::SMatrix{3,3,T}) where {T} = F[1,1]*(F[2,2]F[3,3]-F[2,3]F[3,2]) -
                                            F[1,2]*(F[2,1]F[3,3]-F[2,3]F[3,1]) +
                                            F[1,3]*(F[2,1]F[3,2]-F[2,2]F[3,1])

"""
Compute total integrated Jacobian over the element (useful for volume change).
"""
@inline function getV(elem::CElem{D,P,M,T,N}, u::Matrix{T}) where {D,P,M,T,N}
    total = zero(T)
    @inbounds for ii in 1:P
        total += elem.wgt[ii] * detJ(getF(elem, u, ii))
    end
    return total
end

# function for retriving stress
"""
# getσ(elem, u)

Calculates the volume average Cauchy Stress in the element
"""
function getσ(elem::CElem{D,P,M,S,N} where {S,M}, u::AbstractArray{T}) where {P,D,N,T}
  u = SMatrix{D,N,T}(u)
  σ = @MMatrix zeros(T, D, D)
  @inline for ii=1:P
    σ .+= elem.wgt[ii]*getσ(elem, u, ii)
  end
  SMatrix{D,D}(σ/elem.V)
end
"""
# getσ(elem, u, ii)

Calculates the Cauchy Stress at the integration point ii
"""
function getσ(elem::CElem{D,P,M,S,N} where {P,S,M}, u::AbstractArray{T}, ii::Int) where {D,N,T}

  u  = SMatrix{D,N,T}(u)
  F  = getF(elem, u, ii)
  δϕ = getϕ(adiff.D1(F), elem.mat)
  P  = reshape(adiff.grad(δϕ), size(F))
  J  = detJ(F)

  return 1/J*P*F'
end

# ---------------------------------------------------------------------------
# Scalar-field evaluation functions
# ---------------------------------------------------------------------------

"""
Return scalar field value `d` at Gauss point `ii`.
"""
@inline function get_d(elem::CPElem{D,P,M,T,N}, d::Vector{T}, ii::Integer) where {D,P,M,T,N}
    return dot(elem.N[ii], SVector{N}(d[elem.nodes]))
end

"""
Return scalar field gradient ∇d at Gauss point `ii`.
"""
@inline function get_∇d(elem::CPElem{D,P,M,T,N}, d::Vector{T}, ii::Integer) where {D,P,M,T,N}
    ∇d = @MVector zeros(T, D)
    nodal_d = SVector{N}(d[elem.nodes])
    @inbounds for jj in 1:D
        ∇d[jj] = dot(elem.∇N[jj][ii], nodal_d)
    end
    return SVector{D,T}(∇d)
end

"""
Return scalar field value and gradient `(d, ∇d)` at Gauss point `ii`.
"""
@inline function get_d_and_∇d(elem::CPElem{D,P,M,T,N}, d::Vector{T}, ii::Integer) where {D,P,M,T,N}
    nodal_d = SVector{N}(d[elem.nodes])
    d_val = dot(elem.N[ii], nodal_d)
    ∇d = @MVector zeros(T, D)
    @inbounds for jj in 1:D
        ∇d[jj] = dot(elem.∇N[jj][ii], nodal_d)
    end
    return d_val, SVector{D,T}(∇d)
end

function makeϕrKt(Φ::Vector{<:adiff.D2}, elems::Vector{<:AbstractContinuumElem}, u)

  N  = length(u) 
  Nt = 0
  for ϕ in Φ
    # Nt += length(ϕ.g)*length(ϕ.g)
    Nt += length(ϕ.g.v)*length(ϕ.g.v)
  end

  II = zeros(Int, Nt)
  JJ = zeros(Int, Nt)  
  Kt = zeros(Nt)  
  r  = zeros(N)
  ϕ  = 0
  indxs  = LinearIndices(u)

  N1 = 1
  for (ii,elem) in enumerate(elems)    
    idxii     = indxs[:, elem.nodes][:]    
    ϕ        += adiff.val(Φ[ii]) 
    r[idxii] += adiff.grad(Φ[ii]) 
    nii       = length(idxii)
    Nii       = nii*nii
    oneii     = ones(nii)
    idd       = N1:N1+Nii-1
    II[idd]   = idxii * transpose(oneii)
    JJ[idd]   = oneii * transpose(idxii)
    Kt[idd]   = adiff.hess(Φ[ii])
    N1       += Nii
  end

  ϕ, r, dropzeros(sparse(II,JJ,Kt,N,N))
end
function makeϕr(Φ::Vector{<:adiff.Duals}, elems::Vector{<:AbstractContinuumElem}, u)

  indxs = LinearIndices(u)  
  r     = zeros(length(u))
  ϕ     = 0
  for (ii,elem) in enumerate(elems)    
    idxii     = indxs[:, elem.nodes][:]    
    ϕ        += adiff.val(Φ[ii]) 
    r[idxii] += adiff.grad(Φ[ii]) 
  end
  ϕ, r
end

# ------------------------------------------------------------------
# Operator × for chaining AD quantities
# ------------------------------------------------------------------

"""
×(ϕ::adiff.D2{N,M,T}, F::Array{adiff.D1{P,T}})

Chain an `adiff.D2` free-energy object `ϕ` with a set of deformation-gradient
AD objects `F` to produce a new `adiff.D2` representing the scalar energy
evaluated as a function of the underlying nodal DOFs.

This implements the necessary chain-rule accumulation of first and second
derivatives for element-level assembly.
"""
function ×(ϕ::adiff.D2{N,M,T},F::Array{adiff.D1{P,T}}) where {N,M,P,T}
  val  = ϕ.v
  grad = adiff.Grad(zeros(T,P))
  hess = adiff.Grad(zeros(T,(P+1)P÷2))
  for ii=1:N
    grad += ϕ.g[ii]*F[ii].g
  end
  for ii=2:N, jj=1:ii-1
    hess += ϕ.h[ii,jj]*(F[ii].g*F[jj].g + F[jj].g*F[ii].g)
  end  
  for ii=1:N
    hess += ϕ.h[ii,ii]F[ii].g*F[ii].g
  end
  adiff.D2(val, grad, hess)
end

"""
×(ϕ::adiff.D1{N,T}, F::Array{adiff.D1{P,T}})

Chain an `adiff.D1` scalar object with deformation-gradient AD objects `F`.
Returns an `adiff.D1` with propagated first derivatives.
"""
function ×(ϕ::adiff.D1{N,T},F::Array{adiff.D1{P,T}}) where {N,P,T}
  val  = ϕ.v
  grad = adiff.Grad(zeros(T,P))
  for ii=1:N
    grad += ϕ.g[ii]*F[ii].g
  end
  adiff.D1(val, grad)
end


