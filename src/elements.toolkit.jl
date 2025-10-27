# ---------------------------------------------------------------------------
# Mechanical evaluation functions
# ---------------------------------------------------------------------------

"""
Compute deformation gradient F = I + ∇u at Gauss point `ii`.
"""
@inline getF(elem::CElem{D,P,M,T,N}, u::Matrix{T}, ii::Integer) where {D,P,M,T,N} = get∇u(elem,u,ii) + I
@inline function get∇u(elem::CElem{D,P,M,T,N}, u::Matrix{T}, ii::Integer) where {D,P,M,T,N}
    ∇u = @MMatrix zeros(T, D, D)
    u = SMatrix{D,N}(u)
    @inbounds for jj in 1:D
        for kk in 1:D
          ∇u[jj, kk] = elem.∇N[kk][ii]⋅u[kk,:]
        end
    end
    return SMatrix{D,D,T}(∇u)
end
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


