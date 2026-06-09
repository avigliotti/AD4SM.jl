export makeϕrKt
# 
#
include("./elasticelements.1stord.jl")
include("./elasticelements.2ndord.jl")
# 
# elastic energy evaluation functions for elements
# General CEElem energy integrator (works with CEElem/CPElem)
function getϕ(elem::CEElem{<:Any,P}, u::Array{D}) where {P,D}
  ϕ = zero(D)
  for ii=1:P
    Fii = getF(elem, u, ii)
    ϕ  += elem.wgt[ii]getϕ(Fii, elem.mat)
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
"""
    getϕ(elem::C3DE{P}, u0::Array{D}) where D<:adiff.D2

Optimized 3D mechanical free-energy evaluation using local 3×3 kinematics at
Gauss-point level and the `×` operator for chain-rule propagation back to the
nodal DOFs.
"""
function getϕ(elem::C3DE{P}, u0::Array{D}) where {P,D<:adiff.D2}

  u0 = adiff.D1.(u0)
  ϕ  = zero(D)
  @inline for ii=1:P
    F    = getF(elem, u0, ii)
    valF = adiff.val.(F)
    δϕ   = getϕ(adiff.D2(valF), elem.mat)
    ϕ   += elem.wgt[ii] * (δϕ × F)
  end
  ϕ
end
#
#
# functions for evaluating the residual and the tangent stiffness matrix over
# an array of elements
#
function makeϕrKt(elems::AbstractVector{<:CEElem}, u::AbstractMatrix{T}) where T
  nElems = length(elems)
  @assert nElems > 0 "makeϕrKt: `elems` is empty"  
  #=
  et = eltype(elems)
  D  = et.parameters[1]
  L  = et.parameters[5]
  N  = D*L 
  =# 
  N  = length(u[:,elems[1].nodes])
  M  = (N+1)N÷2

  Φ = Vector{adiff.D2{N,M,T}}(undef, nElems)
  Threads.@threads for ii=1:nElems
    Φ[ii] = getϕ(elems[ii], adiff.D2(u[:,elems[ii].nodes]))
  end

  makeϕrKt(Φ, elems, u)
end
#
#
# function getδϕ(elem::C3DE{P}, u0::Array{T})  where {P,T}  
# evaluates the strain energy density as a dual D2 number 
#
getδϕ(elem::AbstractElement, u::Array{<:Number}) = getϕ(elem, adiff.D2(u))

function getδϕ(elem::C3DE{P}, u0::Array{T})  where {P,T}
  #
  # This implementation computes the D2 dual for the element internal energy.
  # It builds the sensitivity of F with respect to nodal DOFs (δF) using the
  # element's ∇N data (assumed stored as ∇N[1][ii], ∇N[2][ii], ∇N[3][ii])
  # where each ∇N[*][ii] is a static-vector (SVector) or array with length nNodes.
  #

  u, v, w = u0[1:3:end], u0[2:3:end], u0[3:3:end]
  nnode   = length(u)             # number of nodes
  Ndofs   = 3 * nnode             # total number of nodal displacement DOFs
  wgt     = elem.wgt
  val     = zero(T)
  grad    = zeros(T, Ndofs)
  hess    = zeros(T, (Ndofs+1)*Ndofs ÷ 2)  # triangular storage
  δF      = zeros(T, Ndofs, 9)

  @inbounds for ii=1:P
    # get shape-function derivative arrays at Gauss point ii
    Nx = elem.∇N[1][ii]
    Ny = elem.∇N[2][ii]
    Nz = elem.∇N[3][ii]

    # Build δF: derivative of each F component wrt each nodal DOF
    # Column ordering of F components: (F11,F12,F13,F21,F22,F23,F31,F32,F33)
    # For node a, nodal DOFs indices (ux,uy,uz) = (3*(a-1)+1,...+3)
    for a=1:nnode
      idx = 3*(a-1)
      nx = Nx[a]
      ny = Ny[a]
      nz = Nz[a]

      δF[idx+1, 1] = nx   # dF11/d(ux_a)
      δF[idx+2, 2] = nx   # dF12/d(uy_a) ??? (kept same mapping as original code)
      δF[idx+3, 3] = nx   # dF13/d(uz_a)

      δF[idx+1, 4] = ny   # dF21/d(ux_a)
      δF[idx+2, 5] = ny   # dF22/d(uy_a)
      δF[idx+3, 6] = ny   # dF23/d(uz_a)

      δF[idx+1, 7] = nz   # dF31/d(ux_a)
      δF[idx+2, 8] = nz   # dF32/d(uy_a)
      δF[idx+3, 9] = nz   # dF33/d(uz_a)
    end

    # Evaluate F at this Gauss point
    F = SMatrix{3,3,T}(
                       (Nx⋅u + 1) , (Nx⋅v) , (Nx⋅w),
                       (Ny⋅u)      , (Ny⋅v + 1) , (Ny⋅w),
                       (Nz⋅u)      , (Nz⋅v) , (Nz⋅w + 1)
                      )

    # Evaluate constitutive D2 energy for F (material returns adiff.D2)
    ϕ = getϕ(adiff.D2(adiff.val.(F)), elem.mat)::adiff.D2{9, 45, T}

    # accumulate energy, gradient and (triangular) Hessian using δF mapping
    val += wgt[ii] * ϕ.v

    # Gradient: grad[i] += wgt * sum_j ϕ.g[j] * δF[i,j]
    @inbounds for j = 1:9
      coeff = wgt[ii] * ϕ.g[j]
      for i1 = 1:Ndofs
        grad[i1] += coeff * δF[i1, j]
      end
    end

    # Hessian: hess[index(i1,i2)] += wgt * sum_{j,k} ϕ.h[j,k] * δF[i1,j]*δF[i2,k]
    @inbounds for j = 1:9
      for k = 1:j
        hjk = wgt[ii] * ϕ.h[j,k]
        if hjk == zero(hjk)
          continue
        end
        for i1 = 1:Ndofs
          c1 = δF[i1, j]
          if c1 == zero(c1)
            continue
          end
          for i2 = 1:i1
            # triangular index mapping (i2 <= i1)
            idx_tri = (i1-1)*i1 ÷ 2 + i2
            hess[idx_tri] += hjk * c1 * δF[i2, k]
          end
        end
      end
    end
  end

  adiff.D2(val, adiff.Grad(grad), adiff.Grad(hess))
end

function getδϕ(elems::Vector{<:CEElem}, u::Array{T,2}) where T
  nElems = length(elems)
  N      = length(u[:,elems[1].nodes])
  M      = (N+1)N÷2

  Φ = Vector{adiff.D2{N,M,T}}(undef, nElems)
  Threads.@threads for ii=1:nElems
    Φ[ii] = getδϕ(elems[ii], u[:,elems[ii].nodes])
  end
  Φ
end


# ===========================================================================
# elasticelements.axisym.jl
# ---------------------------------------------------------------------------
# Energy, residual and tangent stiffness for CASE elements.
#
# KEY KINEMATIC FACTS (recap):
#   DOFs per element : 2*N  (u_r^1..u_r^N, u_z^1..u_z^N  interleaved as
#                            [ur1, uz1, ur2, uz2, ...] matching the
#                            global 2-row DOF layout u[1,:], u[2,:])
#   F components     : 9   (3×3, column-major)
#
# Non-zero sensitivities  δF[dof_i, F_col]  for node a:
#   dof layout:  ur^a → row 2(a-1)+1,   uz^a → row 2(a-1)+2
#
#   F[1,1] = ∂u_r/∂r + 1   → col 1   ∂/∂ur^a = Nr[a]
#   F[2,1] = ∂u_z/∂r       → col 2   ∂/∂uz^a = Nr[a]
#   F[1,2] = ∂u_r/∂z       → col 4   ∂/∂ur^a = Nz[a]
#   F[2,2] = ∂u_z/∂z + 1   → col 5   ∂/∂uz^a = Nz[a]
#   F[3,3] = u_r/r + 1     → col 9   ∂/∂ur^a = N0[a]/r
#   (all other columns are zero for axisymmetric problems)
#
# Note: column index follows Julia column-major SMatrix{3,3} ordering:
#   col 1 = F[:,1] = (F11,F21,F31)   col 4 = F[:,2] = (F12,F22,F32)  etc.
# ===========================================================================

# ---------------------------------------------------------------------------
# getϕ  (plain real — used for debugging / energy monitoring)
# ---------------------------------------------------------------------------
"""
    getϕ(elem::CASE, u)

Evaluate the total strain energy of an axisymmetric element given the
nodal displacement array `u` (2 × N_nodes, rows = [u_r; u_z]).
"""
function getϕ(elem::CASE{P,M,T,N,O}, u::AbstractArray{D}) where {P,M,T,N,O,D}
  ϕ = zero(D)
  @inbounds for ii in 1:P
    F   = getF(elem, u, ii)
    ϕ  += elem.wgt[ii] * getϕ(F, elem.mat)
  end
  return ϕ
end

"""
    getϕ(elem::CASE{P,M,T,N,O}, u0::AbstractArray{D}) where D<:adiff.D2

Optimized axisymmetric free-energy evaluation using local 3×3 kinematics at
Gauss-point level and the `×` operator for chain-rule propagation back to the
2N nodal DOFs.
"""
function getϕ(elem::CASE{P,M,T,N,O}, u0::AbstractArray{D}) where {P,M,T,N,O,D<:adiff.D2}
  u0 = adiff.D1.(u0)
  ϕ  = zero(D)
  @inbounds for ii in 1:P
    F    = getF(elem, u0, ii)
    valF = adiff.val.(F)
    δϕ   = getϕ(adiff.D2(valF), elem.mat)
    ϕ   += elem.wgt[ii] * (δϕ × F)
  end
  return ϕ
end

# ---------------------------------------------------------------------------
# getδϕ  — D2 dual: energy + gradient (residual) + Hessian (stiffness)
# ---------------------------------------------------------------------------
"""
    getδϕ(elem::CASE{P,M,T,N}, u0)

Compute the element strain energy as an `adiff.D2` dual number with
respect to the `2N` nodal DOFs `u0` (laid out as [ur1,uz1,ur2,uz2,...]).

Returns `adiff.D2{2N, N*(2N+1), T}`.
"""
function getδϕ(elem::CASE{P_,M,T_,Nn,O}, u0::AbstractArray{T}) where {P_,M,T_,Nn,O,T}

  P     = length(elem.wgt)
  Ndofs = 2 * Nn                         # 2 DOFs per node (ur, uz)
  val   = zero(T)
  grad  = zeros(T, Ndofs)
  hess  = zeros(T, (Ndofs+1)*Ndofs ÷ 2)

  # δF layout:  δF[dof_i, F_col],  F_col ∈ 1..9 (column-major)
  δF = zeros(T, Ndofs, 9)

  @inbounds for ii in 1:P

    Nr  = elem.∇N[1][ii]         # ∂N/∂r  at GP ii
    Nz  = elem.∇N[2][ii]         # ∂N/∂z  at GP ii
    N0  = elem.N0[ii]            # N_a    at GP ii
    r   = elem.r_GP[ii]          # reference radial coord at GP ii
    w   = elem.wgt[ii]

    # ----------------------------------------------------------------
    # Fill δF for this Gauss point
    # ----------------------------------------------------------------
    fill!(δF, zero(T))
    @inbounds for a in 1:Nn
      ur_idx = 2*(a-1) + 1     # DOF index for u_r^a
      uz_idx = 2*(a-1) + 2     # DOF index for u_z^a

      δF[ur_idx, 1] = Nr[a]        # ∂F11/∂ur^a
      δF[uz_idx, 2] = Nr[a]        # ∂F21/∂uz^a
      # col 3 (F31) = 0
      δF[ur_idx, 4] = Nz[a]        # ∂F12/∂ur^a
      δF[uz_idx, 5] = Nz[a]        # ∂F22/∂uz^a
      # col 6 (F32) = 0
      # col 7 (F13) = 0
      # col 8 (F23) = 0
      δF[ur_idx, 9] = N0[a] / r    # ∂F33/∂ur^a  (hoop term)
    end

    # ----------------------------------------------------------------
    # Evaluate F at GP from the plain (non-dual) displacements
    # ----------------------------------------------------------------
    ur = SVector{Nn,T}(u0[1:2:end])
    uz = SVector{Nn,T}(u0[2:2:end])
    Fθθ = one(T) + (N0 ⋅ ur) / r
    F_val = SMatrix{3,3,T}(
                           Nr⋅ur + 1,  Nr⋅uz,  zero(T),
                           Nz⋅ur,      Nz⋅uz+1, zero(T),
                           zero(T),    zero(T), Fθθ
                          )

    # ----------------------------------------------------------------
    # Constitutive dual  ϕ(F) — material is agnostic of element type
    # ----------------------------------------------------------------
    ϕ = getϕ(adiff.D2(adiff.val.(F_val)), elem.mat)::adiff.D2{9,45,T}

    # ----------------------------------------------------------------
    # Accumulate energy
    # ----------------------------------------------------------------
    val += w * ϕ.v

    # ----------------------------------------------------------------
    # Accumulate gradient:  r_i += w * Σ_j (∂ϕ/∂F_j) δF[i,j]
    # ----------------------------------------------------------------
    @inbounds for j in 1:9
      coeff = w * ϕ.g[j]
      iszero(coeff) && continue
      for i in 1:Ndofs
        grad[i] += coeff * δF[i, j]
      end
    end

    # ----------------------------------------------------------------
    # Accumulate Hessian (triangular storage, i2 ≤ i1):
    #   K[i1,i2] += w * Σ_{j,k} (∂²ϕ/∂F_j∂F_k) δF[i1,j] δF[i2,k]
    # ----------------------------------------------------------------
    @inbounds for j in 1:9, k in 1:j
      hjk = w * ϕ.h[j,k]
      iszero(hjk) && continue
      for i1 in 1:Ndofs
        c1 = δF[i1, j]
        iszero(c1) && continue
        for i2 in 1:i1
          idx_tri = (i1-1)*i1 ÷ 2 + i2
          hess[idx_tri] += hjk * c1 * δF[i2, k]
        end
      end
    end

  end  # Gauss loop

  return adiff.D2(val, adiff.Grad(grad), adiff.Grad(hess))
end


# ---------------------------------------------------------------------------
# makeϕrKt  — assemble global energy, residual and stiffness
# ---------------------------------------------------------------------------
"""
    makeϕrKt(elems::AbstractVector{<:CASE}, u)

Assemble the global strain energy `ϕ`, residual vector `r`, and tangent
stiffness matrix `Kt` for an array of axisymmetric elements.

`u` must be a (2 × n_nodes) displacement array (rows = [u_r; u_z]).

Returns `(ϕ, r, Kt)` in the same format as the CEElem version.
"""
function makeϕrKt(elems::AbstractVector{<:CASE}, u::AbstractMatrix{T}) where T

  nElems = length(elems)
  @assert nElems > 0 "makeϕrKt: `elems` is empty"

  # Element DOF count — same for every element in a homogeneous array
  elem1  = elems[1]
  Nn     = length(elem1.nodes)
  Ndofs  = 2 * Nn
  Mdofs  = (Ndofs + 1) * Ndofs ÷ 2

  Φ = Vector{adiff.D2{Ndofs, Mdofs, T}}(undef, nElems)

  Threads.@threads for ii in 1:nElems
    Φ[ii] = getϕ(elems[ii], adiff.D2(u[:, elems[ii].nodes]))
  end

  # Reuse the existing sparse-assembly utility from elements.toolkit.jl
  makeϕrKt(Φ, elems, u)
end


# ---------------------------------------------------------------------------
# Convenience: getσ for post-processing
# ---------------------------------------------------------------------------
"""
    getσ(elem::CASE, u)

Return the volume-averaged Cauchy stress tensor (3×3) for the element.
"""
function getσ(elem::CASE{P_,M,T_,Nn}, u::AbstractArray{T}) where {P_,M,T_,Nn,T}
  P = length(elem.wgt)
  σ = @MMatrix zeros(T, 3, 3)
  u_s = SMatrix{2,Nn,T}(u[1:2,:])
  @inbounds for ii in 1:P
    F   = getF(elem, u_s, ii)
    δϕ  = getϕ(adiff.D1(F), elem.mat)
    Pij = reshape(adiff.grad(δϕ), 3, 3)
    J   = detJ(F)
    σ  .+= elem.wgt[ii] * (1/J) .* (Pij * F')
  end
  return SMatrix{3,3,T}(σ / elem.V)
end
