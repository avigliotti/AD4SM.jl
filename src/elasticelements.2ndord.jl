# ===========================================================================
# elasticelements.2ndord.jl
# ---------------------------------------------------------------------------
# Second-order (Quadratic) Mechanical Elements
# ===========================================================================

#=
# Theoretical Background: Second-Order Mechanical Elements

## 1. Isoparametric Mapping & Jacobian
The mapping from reference coordinates ($\boldsymbol{\xi}$) to physical coordinates ($\mathbf{x}$) is:
$$ \mathbf{x}(\boldsymbol{\xi}) = \sum_{a} N_a(\boldsymbol{\xi}) \mathbf{x}_a $$
The Jacobian matrix of this transformation is $\mathbf{J} = \partial \mathbf{x} / \partial \boldsymbol{\xi}$.
Spatial gradients of shape functions are computed via the chain rule:
$$ \nabla_\mathbf{x} N = \mathbf{J}^{-T} \nabla_{\boldsymbol{\xi}} N $$
In the code, this is implemented as `J' \ ∇N_ref`.

## 2. Volumetric Locking & Reduced Integration
Full integration (e.g., $3\times3$ for Quad09) imposes strict volumetric constraints that can cause "locking" (stiff behavior/checkerboard pressure).
Reduced integration (e.g., $2\times2$) relaxes these constraints, providing accurate results for nearly incompressible materials or constrained boundaries.

## 3. Element Types
- **Lagrange (Quad09, Hex27)**: Full tensor product. High accuracy but prone to locking without reduced integration.
- **Serendipity (Quad08, Hex20)**: Boundary nodes only. More efficient, less prone to locking, but can suffer from hourglassing if under-integrated.
- **Simplices (Tria06, Tet10)**: Standard quadratic elements.
=#

# ---------------------------------------------------------------------------
# Shared Helper: Gauss Quadrature Rules
# ---------------------------------------------------------------------------
function get_gauss_rule(::Type{Val{:line}}, bReduced::Bool)
    if bReduced # 2 points (Order 3 precision)
        val = 0.577350269189626
        return ((-val, 1.0), (val, 1.0))
    else # 3 points (Order 5 precision)
        v1, w1 = 0.774596669241483, 0.555555555555556
        v2, w2 = 0.0, 0.888888888888889
        return ((-v1, w1), (v2, w2), (v1, w1))
    end
end

# ---------------------------------------------------------------------------
# 2D Elements
# ---------------------------------------------------------------------------

"""
    Tria06(nodes, p0; mat=Materials.Hooke(), bReduced=false)

Constructs a 6-node quadratic triangular element.
**Quadrature**: Uses a 3-point rule (Strang) by default. `bReduced` currently maps to the same rule as it is optimal for T6 stiffness.
"""
function Tria06(nodes::Vector{<:Integer}, 
                p0::Vector{Vector{T}};
                mat=Materials.Hooke(),
                bReduced::Bool=false) where T<:Number

    # Shape functions: L1(2L1-1), etc.
    function N(ξ, η) 
        λ = (1.0 - ξ - η, ξ, η) # Barycentric coordinates
        SVector(
            λ[1]*(2λ[1]-1), λ[2]*(2λ[2]-1), λ[3]*(2λ[3]-1), # Corners
            4λ[1]λ[2],      4λ[2]λ[3],      4λ[3]λ[1]       # Midsides
        )
    end

    # 3-point rule (Degree 2) - Midpoints of edges
    # This is the standard efficient rule for quadratic triangles.
    # We ignore bReduced as 1-point integration is unstable for quadratic stiffness.
    w = 1.0/3.0
    GPs = ((SVector(1.0/6.0, 1.0/6.0), w),
           (SVector(2.0/3.0, 1.0/6.0), w),
           (SVector(1.0/6.0, 2.0/3.0), w),)

    nGP = length(GPs)
    Nx  = Matrix{Vector{T}}(undef, 1, nGP)
    Ny  = Matrix{Vector{T}}(undef, 1, nGP)
    wgt = Matrix{T}(undef, 1, nGP)
    A   = zero(T)

    @inbounds for (ii, (Pii, wii)) in enumerate(GPs)
        # AD: Differentiate shape functions wrt reference coords (ξ, η)
        N_dual = N(adiff.D1(Pii)...) 

        # Map to physical space
        # p is a vector of D1 duals representing (x, y)
        p = sum(N_dual[a] * p0[a] for a in 1:6) 

        # Jacobian J_ij = dx_i / dξ_j
        # transposed Jacobian
        Jᵀ = SMatrix{2,2,T}(p[ii].g[jj] for jj in 1:2, ii in 1:2)
        det_J = detJ(Jᵀ)

        # Spatial Gradient: ∇x N = J^(-T) * ∇ξ N
        # We use J' \ ... to solve J^T * ∇x N = ∇ξ N
        grads_ref = hcat(adiff.grad.(N_dual)...) # Columns are ∇ξ N_a
        grads_phys = Jᵀ \ grads_ref

        Nx[1, ii]  = grads_phys[1, :]
        Ny[1, ii]  = grads_phys[2, :]
        
        # Area weight: 0.5 * detJ * w_gauss (Triangle Area factor is 0.5)
        wgt[1, ii] = 0.5 * abs(det_J) * wii
        A += wgt[1, ii]
    end

    C2DE(nodes, tuple(vec(Nx)...), tuple(vec(Ny)...), tuple(vec(wgt)...), A, mat, 2)
end

"""
    Quad09(nodes, p0; mat=Materials.Hooke(), bReduced=false)

Constructs a 9-node quadratic Lagrangian quadrilateral.
**Quadrature**: 
- Full (`bReduced=false`): 3x3 Gauss (Exact, prone to locking).
- Reduced (`bReduced=true`): 2x2 Gauss (Recommended to avoid locking).
"""
function Quad09(nodes::Vector{<:Integer}, 
                p0::Vector{Vector{T}};
                mat=Materials.Hooke(),
                bReduced::Bool=false) where T<:Number

    # 1D Lagrange polynomials
    poly(ξ) = SVector(0.5*ξ*(ξ-1), (1-ξ^2), 0.5*ξ*(ξ+1))

    # Tensor product N(ξ,η)
    function N(ξ, η)
        Nξ, Nη = poly(ξ), poly(η)
        # Ordering: Corners(1-4), Midsides(5-8), Center(9)
        SVector(
            Nξ[1]*Nη[1], Nξ[3]*Nη[1], Nξ[3]*Nη[3], Nξ[1]*Nη[3],
            Nξ[2]*Nη[1], Nξ[3]*Nη[2], Nξ[2]*Nη[3], Nξ[1]*Nη[2],
            Nξ[2]*Nη[2]
        )
    end

    GP = get_gauss_rule(Val{:line}, bReduced)
    nGP_1d = length(GP)
    nGP = nGP_1d^2
    
    Nx  = Matrix{Vector{T}}(undef, nGP_1d, nGP_1d)
    Ny  = Matrix{Vector{T}}(undef, nGP_1d, nGP_1d)
    wgt = Matrix{T}(undef, nGP_1d, nGP_1d)
    A   = zero(T)

    @inbounds for (jj, (η, wη)) in enumerate(GP), 
      (ii, (ξ, wξ)) in enumerate(GP)

      N_dual = N(adiff.D1([ξ,η])...)
      p = sum(N_dual[a] * p0[a] for a in 1:9)

      # transposed Jacobian
      Jᵀ = SMatrix{2,2,T}(p[ii].g[jj] for jj in 1:2, ii in 1:2)

      grads_ref  = hcat(adiff.grad.(N_dual)...)
      grads_phys = Jᵀ \ grads_ref

      Nx[ii, jj]  = grads_phys[1, :]
      Ny[ii, jj]  = grads_phys[2, :]

      val_w = detJ(Jᵀ) * wξ * wη
      wgt[ii, jj] = val_w
      A += val_w
    end

    C2DE(nodes, tuple(vec(Nx)...), tuple(vec(Ny)...), tuple(vec(wgt)...), A, mat, 2)
end

"""
    Quad08(nodes, p0; mat=Materials.Hooke(), bReduced=false)

Constructs an 8-node Serendipity quadratic quadrilateral.
**Quadrature**: Defaults to 3x3, can reduce to 2x2.
"""
function Quad08(nodes::Vector{<:Integer}, 
                p0::Vector{Vector{T}};
                mat=Materials.Hooke(),
                bReduced::Bool=false) where T<:Number

    GP = get_gauss_rule(Val{:line}, bReduced)
    nGP_1d = length(GP)
    
    # Nodal coordinates in parent space for Serendipity Q8
    # Corners(1-4), Midsides(5-8)
    ξ_n = SVector(-1,  1,  1, -1,  0,  1,  0, -1)
    η_n = SVector(-1, -1,  1,  1, -1,  0,  1,  0)

    Nx  = Matrix{Vector{T}}(undef, nGP_1d, nGP_1d)
    Ny  = Matrix{Vector{T}}(undef, nGP_1d, nGP_1d)
    wgt = Matrix{T}(undef, nGP_1d, nGP_1d)
    A   = zero(T)

    @inbounds for (jj, (η, wη)) in enumerate(GP), (ii, (ξ, wξ)) in enumerate(GP)

      dξ, dη = adiff.D1(ξ), adiff.D1(η)

      # Serendipity Shape Functions
      # Corners: 0.25(1+ξξ_i)(1+ηη_i)(ξξ_i + ηη_i - 1)
      # Midsides: 0.5(1-ξ^2)(1+ηη_i) or 0.5(1+ξξ_i)(1-η^2)
      Nd = MVector{8, typeof(dξ)}(undef)

      for k in 1:4 
        Nd[k] = 0.25 * (1 + dξ*ξ_n[k]) * (1 + dη*η_n[k]) * (dξ*ξ_n[k] + dη*η_n[k] - 1)
      end
      for k in 5:8 
        if ξ_n[k] == 0
          Nd[k] = 0.5 * (1 - dξ^2) * (1 + dη*η_n[k])
        else
          Nd[k] = 0.5 * (1 + dξ*ξ_n[k]) * (1 - dη^2)
        end
      end

      p = sum(Nd[k] * p0[k] for k in 1:8)
      # transposed Jacobian
      Jᵀ = SMatrix{2,2,T}(p[ii].g[jj] for jj in 1:2, ii in 1:2)

      grads_ref  = hcat(adiff.grad.(Nd)...)
      grads_phys = Jᵀ \ grads_ref

      val_w = detJ(Jᵀ) * wξ * wη
      wgt[ii, jj] = val_w
      Nx[ii, jj]  = grads_phys[1,:]
      Ny[ii, jj]  = grads_phys[2,:]
      A += val_w
    end

    C2DE(nodes, tuple(vec(Nx)...), tuple(vec(Ny)...), tuple(vec(wgt)...), A, mat, 2)
end

# ---------------------------------------------------------------------------
# 3D Elements
# ---------------------------------------------------------------------------

"""
    Tet10(nodes, p0; mat=Materials.Hooke(), bReduced=false)

Constructs a 10-node quadratic tetrahedron.
**Quadrature**: 4-point rule (Keast, degree 2) by default.
"""
function Tet10(nodes::Vector{<:Integer}, 
               p0::Vector{Vector{T}};
               mat=Materials.Hooke(),
               bReduced::Bool=false) where T<:Number

    # Barycentric shape functions
    function N(ξ, η, ζ)
      λ = (1.0 - ξ - η - ζ, ξ, η, ζ)
      SVector(λ[1]*(2λ[1]-1),
              λ[2]*(2λ[2]-1), 
              λ[3]*(2λ[3]-1), 
              λ[4]*(2λ[4]-1), 
              4λ[1]λ[2], 
              4λ[2]λ[3], 
              4λ[3]λ[1], 
              4λ[1]λ[4], 
              4λ[2]λ[4], 
              4λ[3]λ[4])
    end

    # Quadrature (Keast 4-point rule)
    a = 0.5854101966249685
    b = 0.1381966011250105
    w = 1.0/24.0 
    GPs = (
        (SVector(a, b, b), w),
        (SVector(b, a, b), w),
        (SVector(b, b, a), w),
        (SVector(b, b, b), w)
    )

    nGP = length(GPs)
    Nx  = Vector{Vector{T}}(undef, nGP)
    Ny  = Vector{Vector{T}}(undef, nGP)
    Nz  = Vector{Vector{T}}(undef, nGP)
    wgt = Vector{T}(undef, nGP)
    V   = zero(T)

    @inbounds for (ii, (coords, wii)) in enumerate(GPs)
        N_dual = N(adiff.D1(coords)...)
        p = sum(N_dual[a] * p0[a] for a in 1:10)
        
        Jᵀ = SMatrix{3,3,T}(p[ii].g[jj] for jj in 1:3, ii in 1:3)
        grads_ref  = hcat(adiff.grad.(N_dual)...)
        grads_phys = Jᵀ \ grads_ref # J^(-T)

        Nx[ii] = grads_phys[1, :]
        Ny[ii] = grads_phys[2, :]
        Nz[ii] = grads_phys[3, :]
        
        # Volume weight (Tet volume is 1/6 detJ? Keast weights sum to 1/6)
        # We assume standard detJ scaling.
        wgt[ii] = abs(detJ(Jᵀ)) * wii 
        V += wgt[ii]
    end

    C3DE(nodes, tuple(Nx...), tuple(Ny...), tuple(Nz...), tuple(wgt...), V, mat, 2)
end

"""
    Hex27(nodes, p0; mat=Materials.Hooke(), bReduced=false)

Constructs a 27-node quadratic Lagrange hexahedron.
**Quadrature**:
- Full: 3x3x3 Gauss.
- Reduced: 2x2x2 Gauss.
"""
function Hex27(nodes::Vector{<:Integer}, 
               p0::Vector{Vector{T}};
               mat=Materials.Hooke(),
               bReduced::Bool=false) where T<:Number

    poly(ξ) = SVector(0.5*ξ*(ξ-1), (1-ξ^2), 0.5*ξ*(ξ+1))

    function N(ξ, η, ζ)
        Nx, Ny, Nz = poly(ξ), poly(η), poly(ζ)
        # Efficient tensor product generation using broadcasting/reshape might be cleaner,
        # but explicit loops ensure control over ordering (Lexicographical X-Y-Z)
        vals = MVector{27, typeof(Nx[1])}(undef)
        idx = 1
        for k in 1:3, j in 1:3, i in 1:3
            vals[idx] = Nx[i] * Ny[j] * Nz[k]
            idx += 1
        end
        return SVector(vals)
    end

    GP = get_gauss_rule(Val{:line}, bReduced)
    nGP_1d = length(GP)
    
    Nx  = Array{Vector{T}, 3}(undef, nGP_1d, nGP_1d, nGP_1d)
    Ny  = Array{Vector{T}, 3}(undef, nGP_1d, nGP_1d, nGP_1d)
    Nz  = Array{Vector{T}, 3}(undef, nGP_1d, nGP_1d, nGP_1d)
    wgt = Array{T, 3}(undef, nGP_1d, nGP_1d, nGP_1d)
    V   = zero(T)

    @inbounds for 
      (k, (ζ, wζ)) in enumerate(GP), 
      (j, (η, wη)) in enumerate(GP), 
      (i, (ξ, wξ)) in enumerate(GP)
        
      N_dual = N(adiff.D1([ξ,η,ζ])...)
        p = sum(N_dual[a] * p0[a] for a in 1:27)
        
        Jᵀ = SMatrix{3,3,T}(p[ii].g[jj] for jj in 1:3, ii in 1:3)
        grads_ref  = hcat(adiff.grad.(N_dual)...)
        grads_phys = Jᵀ \ grads_ref # J^(-T)

        Nx[i,j,k] = grads_phys[1, :]
        Ny[i,j,k] = grads_phys[2, :]
        Nz[i,j,k] = grads_phys[3, :]
        
        val_w = detJ(Jᵀ) * wξ * wη * wζ
        wgt[i, j, k] = val_w
        V += val_w
    end

    C3DE(nodes, tuple(vec(Nx)...), tuple(vec(Ny)...), tuple(vec(Nz)...), tuple(vec(wgt)...), V, mat, 2)
end
