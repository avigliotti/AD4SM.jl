export makeϕrKt
# 
# constructors
#
function Rod(nodes, p0, A; mat=Materials.NO_MAT) 

  r0  = p0[2]-p0[1] 
  l0  = norm(r0)
  Rod(nodes, r0, l0, A, mat)
end
function Beam(nodes, p0, t, w; mat=Materials.NO_MAT, Nx = 5, Ny = 3)

  lgwx = lgwt(Nx)
  lgwy = lgwt(Ny, a=-0.5, b=0.5)

  d0  = p0[2]-p0[1] 
  L   = norm(d0)
  r0  = d0/L

  Beam(nodes, r0, L, t, w, lgwx, lgwy, mat)
end
#
# continuous elements
#
# ===========================================================================
# CONTINUOUS (MECHANICAL) ELEMENTS
# ===========================================================================

function Line(nodes::Vector{<:Integer}, 
              p0::Vector{<:AbstractVector{T}};
              mat::M=Materials.NO_MAT,
              bReduced::Bool=false) where {T<:Number, M<:Material}
  
  x1, x2 = p0[1], p0[2]
  L      = abs(x2 - x1)
  
  Nx  = [(x2-x1)/L] 
  wgt = [one(T)]       
  A   = L    

  C1DE(nodes, tuple(Nx...), tuple(wgt...), A, mat) 
end

function Tria03(nodes::Vector{<:Integer}, 
                p0::Vector{<:AbstractVector{T}}; 
                mat::M=Materials.NO_MAT,
                bReduced::Bool=false) where {T<:Number, M<:Material}

  N(ξ,η) = SVector(1-ξ-η, ξ, η)
  GPs = ((SVector{2,T}(1/3, 1/3), T(0.5)),)
  
  nGP = length(GPs)
  nN  = length(nodes)

  ∇N,wgt,V = _calculate_mech_fields_2d(N, GPs, nodes, p0)

  C2DE{nGP,M,T,nN,1}(nodes, ∇N, wgt, V, mat) 
end

function Quad04(nodes::Vector{<:Integer}, 
                p0::Vector{<:AbstractVector{T}};
                mat::M=Materials.NO_MAT, 
                bReduced::Bool=false) where {T<:Number, M<:Material}

  function N(ξ, η)
      omx, opx = 1-ξ, 1+ξ
      ome, ope = 1-η, 1+η
      SVector(omx*ome, opx*ome, opx*ope, omx*ope) .* T(0.25)
  end

  GPs = if bReduced
      ((SVector{2,T}(0, 0), T(4.0)),)
  else
      g = T(1/√3) 
      w = one(T)
      ((SVector{2,T}(-g, -g), w), (SVector{2,T}( g, -g), w),
       (SVector{2,T}( g,  g), w), (SVector{2,T}(-g,  g), w))
  end

  nGP = length(GPs)
  nN  = length(nodes)

  ∇N,wgt,V = _calculate_mech_fields_2d(N, GPs, nodes, p0)

  C2DE{nGP,M,T,nN,1}(nodes, ∇N, wgt, V, mat) 
end

function Tet04(nodes::Vector{<:Integer}, 
               p0::Vector{<:AbstractVector{T}};
               mat::M=Materials.NO_MAT,
               bReduced::Bool=false) where {T<:Number, M<:Material}

  N(ξ,η,ζ) = SVector(1-ξ-η-ζ, ξ, η, ζ)

  GPs = if bReduced
    ((SVector{3,T}(0.25, 0.25, 0.25), T(1/6)),)
  else 
    a, b = T(0.5854101966249685), T(0.1381966011250105)
    w    = T(1/24) 
    ((SVector{3,T}(a,b,b),w), (SVector{3,T}(b,a,b),w),
     (SVector{3,T}(b,b,a),w), (SVector{3,T}(b,b,b),w))
  end

  nGP = length(GPs)
  nN  = length(nodes)

  ∇N,wgt,V = _calculate_mech_fields_3d(N, GPs, nodes, p0)

  C3DE{nGP,M,T,nN,1}(nodes, ∇N, wgt, V, mat) 
end

function Hex08(nodes::Vector{<:Integer}, 
               p0::Vector{<:AbstractVector{T}};
               mat::M=Materials.NO_MAT,
               bReduced::Bool=false) where {T<:Number, M<:Material}

  function N(ξ, η, ζ)
      omx, opx = 1-ξ, 1+ξ
      ome, ope = 1-η, 1+η
      omz, opz = 1-ζ, 1+ζ
      SVector(omx*ome*omz, opx*ome*omz, opx*ope*omz, omx*ope*omz,
              omx*ome*opz, opx*ome*opz, opx*ope*opz, omx*ope*opz) .* T(0.125)
  end

  GPs = if bReduced
      ((SVector{3,T}(0, 0, 0), T(8.0)),)
  else
      g = T(1/√3)
      w = one(T)
      pts = SVector{3,T}[]
      for k in (-g,g), j in (-g,g), i in (-g,g)
          push!(pts, SVector{3,T}(i,j,k))
      end
      Tuple((p, w) for p in pts)
  end

  nGP = length(GPs)
  nN  = length(nodes)

  ∇N,wgt,V = _calculate_mech_fields_3d(N, GPs, nodes, p0)

  C3DE{nGP,M,T,nN,1}(nodes, ∇N, wgt, V, mat) 
end

function Wdg06(nodes::Vector{<:Integer}, 
                p0::Vector{<:AbstractVector{T}};
               mat::M=Materials.NO_MAT,
               bReduced::Bool=false) where {T<:Number, M<:Material}

  N(ξ,η,ζ) = SVector((1-ζ)*(1-ξ-η), (1-ζ)*ξ, (1-ζ)*η,
                     (1+ζ)*(1-ξ-η), (1+ζ)*ξ, (1+ζ)*η) .* T(0.5)

  GPs = if bReduced
      ((SVector{3,T}(1/3, 1/3, 0), T(1.0)),)
  else
      r23, r16, sq3 = T(2/3), T(1/6), T(1/√3)
      w = T(1/3)
      ((SVector{3,T}(r23, r16,  sq3), w), (SVector{3,T}(r23, r16, -sq3), w),
       (SVector{3,T}(r16, r23,  sq3), w), (SVector{3,T}(r16, r23, -sq3), w),
       (SVector{3,T}(r16, r16,  sq3), w), (SVector{3,T}(r16, r16, -sq3), w))
  end

  nGP = length(GPs)
  nN  = length(nodes)

  ∇N,wgt,V = _calculate_mech_fields_3d(N, GPs, nodes, p0)

  C3DE{nGP,M,T,nN,1}(nodes, ∇N, wgt, V, mat) 
end

const Quad = Quad04       # backward compatilbilty, will be removed
const Tria = Tria03       # backward compatilbilty, will be removed

# 2D Mechanical Fields (Used by Tria03, Quad04)
function _calculate_mech_fields_2d(N::F, GPs, nodes::Vector, p0::Vector{<:AbstractVector{T}}) where {F<:Function, T<:Number}
    nGP = length(GPs)
    nN  = length(nodes)

    Nx  = Vector{Vector{T}}(undef, nGP)
    Ny  = Vector{Vector{T}}(undef, nGP)
    wgt = Vector{T}(undef, nGP)
    V   = zero(T)

    @inbounds for (ii, (coords, wii)) in enumerate(GPs)
        N_dual = N(adiff.D1(coords)...)
        # Interpolate physical coordinates p = {x, y}
        p      = sum(N_dual[k] * p0[k] for k in 1:nN)
        # Transposed Jacobian Jᵀ (2x2)
        Jᵀ     = SMatrix{2,2,T}(p[k].g[j] for j in 1:2, k in 1:2)
        # Map gradients: J⁻ᵀ ∇_ξ N
        grads  = Jᵀ \ hcat(adiff.grad.(N_dual)...)

        Nx[ii]  = grads[1, :]
        Ny[ii]  = grads[2, :]
        # FIX: Use standard `det` function
        wgt[ii] = abs(det(Jᵀ)) * wii 
        V      += wgt[ii]
    end
    
    ∇N = tuple(tuple(Nx...), tuple(Ny...))
    # Return ∇N (tuple of tuples), wgt (tuple), V (scalar)
    return ∇N, tuple(wgt...), V
end

# 3D Mechanical Fields (Used by Tet04, Hex08, Wdg06)
function _calculate_mech_fields_3d(N::F, GPs, nodes::Vector, p0::Vector{<:AbstractVector{T}}) where {F<:Function, T<:Number}
    nGP = length(GPs)
    nN  = length(nodes)

    Nx  = Vector{Vector{T}}(undef, nGP)
    Ny  = Vector{Vector{T}}(undef, nGP)
    Nz  = Vector{Vector{T}}(undef, nGP)
    wgt = Vector{T}(undef, nGP)
    V   = zero(T)

    @inbounds for (ii, (coords, wii)) in enumerate(GPs)
        N_dual = N(adiff.D1(coords)...)
        p      = sum(N_dual[k] * p0[k] for k in 1:nN)
        # Transposed Jacobian Jᵀ (3x3)
        Jᵀ     = SMatrix{3,3,T}(p[k].g[j] for j in 1:3, k in 1:3)
        grads  = Jᵀ \ hcat(adiff.grad.(N_dual)...)

        Nx[ii]  = grads[1, :]
        Ny[ii]  = grads[2, :]
        Nz[ii]  = grads[3, :]
        # FIX: Use standard `det` function
        wgt[ii] = abs(det(Jᵀ)) * wii
        V      += wgt[ii]
    end

    ∇N = tuple(tuple(Nx...), tuple(Ny...), tuple(Nz...))
    return ∇N, tuple(wgt...), V
end

# ===========================================================================
# First-order axisymmetric element constructors:  ASTria  and  ASQuad
#
# Both elements live in the meridional (r,z) plane.
# Convention:  p0[a] = [r_a, z_a]   (first coordinate is the radial one)
#
# The integration weight already absorbs the 2π factor:
#   wgt[ii] = det(J_ii) * w_ref_ii * 2π * r_GP_ii
# so that  ∫_Ω (·) dV  =  ∑_ii wgt[ii] * (·)_ii
# ===========================================================================

# ---------------------------------------------------------------------------
# Helper: build CAS fields from a shape-function N(ξ,η), a Gauss-point tuple
#         and the nodal coordinates p0.
# Returns (N0, Nr, Nz, r_GP, wgt, V) — all as plain Vectors, one entry per GP.
# ---------------------------------------------------------------------------
function _calculate_as_fields_as(N_fun, GPs, nodes, p0::Vector{<:AbstractVector{T}}) where T
    nGP = length(GPs)
    nN  = length(nodes)

    N0_vec  = Vector{Vector{T}}(undef, nGP)
    Nr_vec  = Vector{Vector{T}}(undef, nGP)
    Nz_vec  = Vector{Vector{T}}(undef, nGP)
    r_vec   = Vector{T}(undef, nGP)
    wgt_vec = Vector{T}(undef, nGP)
    V       = zero(T)

    @inbounds for (ii, (coords, wii)) in enumerate(GPs)
        # Evaluate shape functions + derivatives via AD
        N_dual = N_fun(adiff.D1(coords)...)

        # Physical coordinates at GP:  p = [r, z]
        p = sum(N_dual[k] * p0[k] for k in 1:nN)

        # Transposed Jacobian  Jᵀ (2×2),  Jᵀ[j,k] = ∂p_k/∂ξ_j
        Jᵀ = SMatrix{2,2,T}(p[k].g[j] for j in 1:2, k in 1:2)

        # Physical gradients:  ∇_x N  =  J^{-T} ∇_ξ N
        grads = Jᵀ \ hcat(adiff.grad.(N_dual)...)

        N0_vec[ii]  = adiff.val.(N_dual)
        Nr_vec[ii]  = grads[1, :]          # ∂N_a/∂r
        Nz_vec[ii]  = grads[2, :]          # ∂N_a/∂z

        # Radial coordinate at GP  r = Σ N_a r_a
        r_gp        = p[1].v
        r_vec[ii]   = r_gp

        # Weight includes 2π r factor for the volume integral
        wgt_vec[ii] = abs(det(Jᵀ)) * wii * 2T(π) * r_gp
        V          += wgt_vec[ii]
    end

    return N0_vec, Nr_vec, Nz_vec, r_vec, wgt_vec, V
end


# ---------------------------------------------------------------------------
# ASTria  —  3-node linear triangle, 1-point centroid rule
# ---------------------------------------------------------------------------
"""
    ASTria(nodes, p0; mat=Materials.NO_MAT)

3-node linear axisymmetric triangular element (CST in the meridional plane).
Uses 1-point centroid quadrature.

Arguments:
- `nodes` : length-3 integer vector of nodal IDs
- `p0`    : length-3 vector of [r, z] reference nodal coordinates
- `mat`   : material model (must accept a 3×3 F)
"""
function ASTria(nodes::Vector{<:Integer},
                p0::Vector{<:AbstractVector{T}};
                mat::M = Materials.NO_MAT) where {T<:Number, M<:Material}

    N_fun(ξ, η) = SVector(1-ξ-η, ξ, η)

    # 1-point centroid rule for triangles  (weight = area in reference)
    GPs = ((SVector{2,T}(T(1)/3, T(1)/3), T(1)/2),)

    N0, Nr, Nz, r_GP, wgt, V = _calculate_as_fields_as(N_fun, GPs, nodes, p0)

    CASE(nodes, tuple(N0...), tuple(Nr...), tuple(Nz...),
         tuple(r_GP...), tuple(wgt...), V, mat, 1)
end


# ---------------------------------------------------------------------------
# ASQuad  —  4-node bilinear quadrilateral, 2×2 Gauss rule
# ---------------------------------------------------------------------------
"""
    ASQuad(nodes, p0; mat=Materials.NO_MAT, bReduced=false)

4-node bilinear axisymmetric quadrilateral element.
Full integration uses a 2×2 Gauss rule; reduced integration uses 1 central point.

Arguments:
- `nodes`    : length-4 integer vector of nodal IDs
- `p0`       : length-4 vector of [r, z] reference nodal coordinates
                 (counter-clockwise ordering, r ≥ 0)
- `mat`      : material model (must accept a 3×3 F)
- `bReduced` : if `true`, use 1-point central quadrature (hourglass-prone)
"""
function ASQuad(nodes::Vector{<:Integer},
                p0::Vector{<:AbstractVector{T}};
                mat::M    = Materials.NO_MAT,
                bReduced::Bool = false) where {T<:Number, M<:Material}

    function N_fun(ξ, η)
        omx, opx = 1-ξ, 1+ξ
        ome, ope = 1-η, 1+η
        SVector(omx*ome, opx*ome, opx*ope, omx*ope) .* T(0.25)
    end

    GPs = if bReduced
        ((SVector{2,T}(0, 0), T(4)),)
    else
        g = T(1/√3)
        w = one(T)
        ((SVector{2,T}(-g,-g), w), (SVector{2,T}( g,-g), w),
         (SVector{2,T}( g, g), w), (SVector{2,T}(-g, g), w))
    end

    N0, Nr, Nz, r_GP, wgt, V = _calculate_as_fields_as(N_fun, GPs, nodes, p0)

    CASE(nodes, tuple(N0...), tuple(Nr...), tuple(Nz...),
         tuple(r_GP...), tuple(wgt...), V, mat, 1)
end
# 
# elastic energy evaluation functions for elements
# General CEElem energy integrator (works with CEElem/CPElem)
function getϕ(elem::CEElem{<:Any,P}, u::Array{D}) where {P,D}
  ϕ = zero(D)
  @inbounds for ii=1:P
    F   = getF(elem, u, ii)
    ϕ  += elem.wgt[ii]getϕ(F, elem.mat)
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
# getδϕ(elem::AbstractElement, u::Array{<:Number}) = getϕ(elem, adiff.D2(u))

function getδϕ(elem::C3DE{P,M,T,N} where {M}, u0::Array{T})  where {P,N,T}
  #
  # This implementation computes the D2 dual for the element internal energy.
  # It builds the sensitivity of F with respect to nodal DOFs (δF) using the
  # element's ∇N data (assumed stored as ∇N[1][ii], ∇N[2][ii], ∇N[3][ii])
  # where each ∇N[*][ii] is a static-vector (SVector) or array with length nNodes.
  #

  u, v, w = SVector{N}(u0[1:3:end]), 
            SVector{N}(u0[2:3:end]), 
            SVector{N}(u0[3:3:end])
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
    ϕ = getϕ(adiff.D2(F), elem.mat)::adiff.D2{9, 45, T}

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

# ---------------------------------------------------------------------------
# getδϕ  — D2 dual: energy + gradient (residual) + Hessian (stiffness)
# ---------------------------------------------------------------------------
"""
    getδϕ(elem::CASE{P,M,T,N}, u0)

Compute the element strain energy as an `adiff.D2` dual number with
respect to the `2N` nodal DOFs `u0` (laid out as [ur1,uz1,ur2,uz2,...]).

Returns `adiff.D2{2N, N*(2N+1), T}`.
"""
function getδϕ(elem::CASE{P_,M,T_,Nn}, u0::AbstractArray{T}) where {P_,M,T_,Nn,T}

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
    N0  = elem.N[ii]            # N_a    at GP ii
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
    J   = det(F)
    σ  .+= elem.wgt[ii] * (1/J) .* (Pij * F')
  end
  return SMatrix{3,3,T}(σ / elem.V)
end
