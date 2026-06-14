# constructors
#
function Rod(nodes, p0, A; mat=Materials.NoMat()) 

  r0  = p0[2]-p0[1] 
  l0  = norm(r0)
  Rod(nodes, r0, l0, A, mat)
end
function Beam(nodes, p0, t, w; mat=Materials.NoMat(), Nx = 5, Ny = 3)

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
              mat::M=Materials.NoMat(),
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
                mat::M=Materials.NoMat(),
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
                mat::M=Materials.NoMat(), 
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
               mat::M=Materials.NoMat(),
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
               mat::M=Materials.NoMat(),
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
               mat::M=Materials.NoMat(),
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
        wgt[ii] = det(Jᵀ) * wii 
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
        wgt[ii] = det(Jᵀ) * wii
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
        wgt_vec[ii] = det(Jᵀ) * wii * 2T(π) * r_gp
        V          += wgt_vec[ii]
    end

    return N0_vec, Nr_vec, Nz_vec, r_vec, wgt_vec, V
end


# ---------------------------------------------------------------------------
# ASTria  —  3-node linear triangle, 1-point centroid rule
# ---------------------------------------------------------------------------
"""
    ASTria(nodes, p0; mat=Materials.NoMat())

3-node linear axisymmetric triangular element (CST in the meridional plane).
Uses 1-point centroid quadrature.

Arguments:
- `nodes` : length-3 integer vector of nodal IDs
- `p0`    : length-3 vector of [r, z] reference nodal coordinates
- `mat`   : material model (must accept a 3×3 F)
"""
function ASTria(nodes::Vector{<:Integer},
                p0::Vector{<:AbstractVector{T}};
                mat::M = Materials.NoMat()) where {T<:Number, M<:Material}

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
    ASQuad(nodes, p0; mat=Materials.NoMat(), bReduced=false)

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
                mat::M    = Materials.NoMat(),
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
