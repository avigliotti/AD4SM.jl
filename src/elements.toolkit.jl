# ---------------------------------------------------------------------------
# Mechanical evaluation functions
# ---------------------------------------------------------------------------

"""
Compute deformation gradient F = I + ∇u at Gauss point `ii`.
"""
@inline getF(elem::CElem, u::AbstractArray, ii::Integer) = get∇u(elem,u,ii) + I
@inline getF(elem::CElem, u::AbstractArray)              = get∇u(elem,u) + I
getF(elems::Array{<:CElem},  u::AbstractArray) = [getF(elem, u[:,elem.nodes]) for elem in elems]
@inline function get∇u(elem::CElem{D,P,M,<:Any,N}, u::AbstractArray{T}, ii::Integer) where {D,P,M,T,N}
  # ∇u = @MMatrix zeros(T, D, D)
  ∇u = zeros(T, D, D)
  u  = SMatrix{D,N}(u[1:D,:])
  @inbounds for jj in 1:D, kk in 1:D
    ∇u[jj,kk] = elem.∇N[kk][ii]⋅u[jj,:]
  end
  return SMatrix{D,D,T}(∇u)
end
"""
get∇u(elem::CElem{D,P,M,T,N}, u::AbstractArray{T}) where {D,P,M,T,N}

Compute the average of ∇u over the element
"""
function get∇u(elem::CElem{D,P,M,T,N}, u::AbstractArray{T}) where {D,P,M,T,N}
    ∇u = @MMatrix zeros(T, D, D)
    for ii=1:P
      ∇u .+= elem.wgt[ii]get∇u(elem, u, ii)
    end
    return SMatrix{D,D,T}(∇u/elem.V)
end
get∇u(elems::Array{<:CElem}, u::AbstractArray) = [get∇u(elem, u[:,elem.nodes]) for elem in elems]
"""
Compute the gradient of the scalar field at integration point ii
"""
function get∇n(elem::CElem{D,P,M,<:Any,N}, n::AbstractArray{T}, ii::Integer) where {D,P,M,N,T}
  ∇n = @MVector zeros(T,D)
  n  = SVector{N,T}(n)
  @inbounds for jj=1:D
    ∇n[jj] = elem.∇N[jj][ii]⋅n
  end
  return SVector{D,T}(∇n)
end
"""
Compute the average of the gradient of the scalar field over the elment
"""
function get∇n(elem::CElem{D,P,M,<:Any,N}, n::AbstractArray{T}) where {D,P,M,N,T}
  ∇n = @MVector zeros(T,D)
  n  = SVector{N,T}(n)
  @inbounds for ii=1:P, jj=1:D
    ∇n[jj] += elem.wgt[ii]*(elem.∇N[jj][ii]⋅n)
  end
  return SVector{D,T}(∇n/elem.V)
end
"""
Compute determinant of deformation gradient J = det(F).
"""
@inline detJ(F::SMatrix{2,2,T} where T) = F[1,1]F[2,2] - F[1,2]F[2,1]
@inline detJ(F::SMatrix{3,3,T} where T) = (F[1,1]*(F[2,2]F[3,3]-F[2,3]F[3,2]) -
                                           F[1,2]*(F[2,1]F[3,3]-F[2,3]F[3,1]) +
                                           F[1,3]*(F[2,1]F[3,2]-F[2,2]F[3,1]) )

detJ(elem::CElem, u::AbstractArray, ii::Integer) = detJ(getF(elem,u,ii))
function detJ(elem::CElem{D,P,<:Any,<:Any,N}, u::AbstractArray{T}) where {D,P,T,N}
    u = SMatrix{D,N,T}(u[1:D,:])
    J = zero(T)
    @inbounds for ii=1:P
      J += elem.wgt[ii]detJ(elem, u, ii)
    end
    return J/elem.V
end
detJ(elems::Array{<:CElem}, u::AbstractArray)  = [detJ(elem, u[:,elem.nodes]) for elem in elems]

"""
Compute the current volume
getV(elem::CElem{D,P,M,T,N}, u::AbstractArray{T}) where {D,P,M,T,N}
getV(elems::AbstractArray{<:CElem}, u::AbstractArray)
"""
@inline function getV(elem::CElem{<:Any,P}, u::AbstractArray{T}) where {P,T}
    total = zero(T)
    @inbounds for ii in 1:P
        total += elem.wgt[ii] * detJ(getF(elem, u, ii))
    end
    return total
end
getV(elems::AbstractArray{<:CElem}, u::AbstractArray) = sum(getV(elem, u[:,elem.nodes]) for elem in elems)

# function for retriving stress
"""
# getσ(elem, u)

Calculates the volume average Cauchy Stress in the element
"""
function getσ(elem::CElem{D,P,M,S,N} where {S,M}, u::AbstractArray{T}) where {P,D,N,T}
  u = SMatrix{D,N,T}(u[1:D,:])
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

  u  = SMatrix{D,N,T}(u[1:D,:])
  F  = getF(elem, u, ii)
  δϕ = getϕ(adiff.D1(F), elem.mat)
  P  = reshape(adiff.grad(δϕ), size(F))
  J  = detJ(F)

  return 1/J*P*F'
end
function getσ(elems::Vector{<:C3D}, u::Matrix{D}) where D<:Real

  nElems = length(elems)
  #         xx  yy  zz  xy  zx  yz 
  idx = [1,  5,  9,  2,  3,  6]
  σ   = zeros(6,nElems)
  @inbounds for (ii,elem) in enumerate(elems)
    σii = getσ(elems[ii], u[:, elems[ii].nodes])
    σ[:, ii] = σii[idx]
  end

  return σ
end
function getσ(elems::Vector{<:C2D}, u::Matrix{D}) where D<:Real

  nElems = length(elems)
  #         xx  yy  xy
  idx = [1,  4,  2]
  σ   = zeros(3,nElems)
  for (ii,elem) in enumerate(elems)
    @inbounds σ[:, ii] = getσ(elems[ii], u[:, elems[ii].nodes])[idx]
  end

  return σ
end

function getP(elem::CElem{D,P,M,S,N} where {S,M}, u::AbstractArray{T}) where {P,D,N,T}
  u = SMatrix{D,N,T}(u[1:D,:])
  P1K = @MMatrix zeros(T, D, D)
  @inline for ii=1:P
    P1K .+= elem.wgt[ii]*getP(elem, u, ii)
  end
  SMatrix{D,D}(P1K/elem.V)
end
function getP(elem::CElem{D,P,M,S,N} where {P,S,M}, u::AbstractArray{T}, ii::Int) where {D,N,T}

  u   = SMatrix{D,N,T}(u[1:D,:])
  F   = getF(elem, u, ii)
  δϕ  = getϕ(adiff.D1(F), elem.mat)
  P1K = reshape(adiff.grad(δϕ), size(F))

  return P1K'
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
@inline function get_d_and_∇d(elem::CPElem{D,P,M,T,N}, d::SVector, ii::Integer) where {D,P,M,T,N}
    d_val = elem.N[ii]⋅d
    ∇d = @MVector zeros(T, D)
    @inbounds for jj in 1:D
        ∇d[jj] = elem.∇N[jj][ii]⋅d
    end
    return (d_val, SVector{D,T}(∇d))
end

function makeϕrKt(Φ::Vector{<:adiff.D2}, elems::Vector{<:AbstractContinuumElem}, u)

  N  = length(u) 
  Nt = 0
  for ϕ in Φ
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

  # Create the matrix
  K_approx = sparse(II, JJ, Kt, N, N)
  
  # Enforce strict symmetry to handle potential truncation errors during assembly
  K_sym = (K_approx + K_approx') / 2

  ϕ, r, dropzeros(K_sym)
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
function ×(ϕ::adiff.D2{N,M,T},F::AbstractArray{adiff.D1{P,T}}) where {N,M,P,T}
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
function ×(ϕ::adiff.D1{N,T},F::AbstractArray{adiff.D1{P,T}}) where {N,P,T}
  val  = ϕ.v
  grad = adiff.Grad(zeros(T,P))
  for ii=1:N
    grad += ϕ.g[ii]*F[ii].g
  end
  adiff.D1(val, grad)
end

# ============================================================================
# Invariants of C
# ============================================================================

@inline getI₁(F::Union{SMatrix{3,3}, SVector{9}}) = (F[1]F[1]+F[2]F[2]+F[3]F[3]+
                                                     F[4]F[4]+F[5]F[5]+F[6]F[6]+
                                                     F[7]F[7]+F[8]F[8]+F[9]F[9])
@inline getI₂(F::Union{SMatrix{3,3}, SVector{9}}) = let C = F'F
    # I₂=C₁₁C₂₂+C₁₁C₃₃+C₂₂C₃₃−C₁₂²−C₁₃²−C₂₃²
    C[1]C[5]+C[1]C[9]+C[5]C[9]-C[2]C[4]-C[3]C[7]-C[6]C[8]
end
@inline getĪ₁(F::Union{SMatrix{3,3}, SVector{9}}) = getI₁(F)/detJ(F)^(2/3)
@inline getĪ₂(F::Union{SMatrix{3,3}, SVector{9}}) = getI₂(F)/detJ(F)^(4/3)

function getĪ₁(elem::CElem{D,P} where D, u::AbstractArray{T}) where {P,T}
  Ī₁ = zero(T)
  @inbounds for ii=1:P
    Ī₁ += elem.wgt[ii]getĪ₁(getF(elem, u, ii))
  end
  return Ī₁/elem.V
end
getĪ₁(elems::Array{<:Elements.CElem}, u::AbstractArray) = [getĪ₁(elem, u[:,elem.nodes]) for elem in elems]

function getĪ₂(elem::CElem{D,P} where D, u::AbstractArray{T}) where {P,T}
  Ī₂ = zero(T)
  @inbounds for ii=1:P
    Ī₂ += elem.wgt[ii]getĪ₂(getF(elem, u, ii))
  end
  return Ī₂/elem.V
end
getĪ₂(elems::Array{<:Elements.CElem}, u::AbstractArray) = [getĪ₂(elem, u[:,elem.nodes]) for elem in elems]

"""
    el2nodes(elems, J)

Interpolate element-averaged J to nodes.
"""
function el2nodes(elems::Array{<:AbstractElement}, J::Array{T}) where T
  @assert length(elems)==length(J) "elems and J must have the same length "
  nNodes = 0
  for elem in elems
    nNodes = max(nNodes, elem.nodes...)
  end

  # Accumulate J values at each node
  accum = [T[] for _ in 1:nNodes]

  for (elem, J) in zip(elems, J)
    for node in elem.nodes
      push!(accum[node], J)
    end
  end

  # Average
  return [sum(vals) / length(vals) for vals in accum]
end


# ===========================================================================
# elements.toolkit.axisym.jl
# ---------------------------------------------------------------------------
# getF  dispatch for CASElem
#
# The axisymmetric deformation gradient is 3×3 in cylindrical coordinates
# (r, θ, z).  With the axis of symmetry along z and u = [u_r, u_z]:
#
#   F = | ∂u_r/∂r + 1     ∂u_r/∂z       0           |
#       | ∂u_z/∂r         ∂u_z/∂z + 1   0           |
#       | 0               0             u_r/r_GP + 1 |
#
# where  u_r/r_GP  =  (Σ_a N_a u_r^a) / r_GP.
#
# Column-major storage of F (as used throughout AD4SM):
#   index 1..9 = F[1,1], F[2,1], F[3,1], F[1,2], F[2,2], F[3,2],
#                F[1,3], F[2,3], F[3,3]
# ===========================================================================

"""
    getF(elem::CASElem, u, ii)

Return the 3×3 axisymmetric deformation gradient at Gauss point `ii`.

`u` is a (2 × N_nodes) array with rows [u_r; u_z].
"""
@inline function getF(elem::CASElem{P,M,T,N}, u::AbstractArray{D}, ii::Integer) where {P,M,T,N,D}
    ur = SVector{N}(u[1,:])          # radial displacements
    uz = SVector{N}(u[2,:])          # axial  displacements
    Nr = elem.∇N[1][ii]              # ∂N_a/∂r
    Nz = elem.∇N[2][ii]              # ∂N_a/∂z
    N0 = elem.N0[ii]                 # N_a values at GP
    r  = elem.r_GP[ii]               # reference radial coordinate

    # hoop stretch:  (r + u_r)/r = 1 + Σ_a N_a u_r^a / r
    Fθθ = one(D) + (N0 ⋅ ur) / r

    return SMatrix{3,3,D}(
        Nr⋅ur + 1,  Nr⋅uz,      zero(D),
        Nz⋅ur,      Nz⋅uz + 1,  zero(D),
        zero(D),    zero(D),    Fθθ
    )
end

# Volume-averaged F (used by getσ and diagnostics)
function getF(elem::CASElem{P}, u::AbstractArray{T}) where {P,T}
    nN  = length(elem.nodes)
    u   = SMatrix{2,nN,T}(u[1:2,:])
    Favg = @MMatrix zeros(T, 3, 3)
    @inbounds for ii in 1:P
        Favg .+= elem.wgt[ii] .* getF(elem, u, ii)
    end
    SMatrix{3,3,T}(Favg / elem.V)
end
