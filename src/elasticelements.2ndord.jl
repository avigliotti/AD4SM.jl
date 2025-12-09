# ===========================================================================
# elasticelements.2ndord.jl
# ---------------------------------------------------------------------------
# Second-order (Quadratic) Mechanical Elements
# ===========================================================================

#=
# Theoretical Background: Second-Order Mechanical Elements

Second-order elements use quadratic shape functions to interpolate the geometry
and displacement fields. This allows them to capture linear strain gradients
(unlike first-order elements which have constant strain) and curve boundaries.

## 1. Shape Functions
These elements are "Isoparametric", meaning the same shape functions $N(\xi)$
interpolate both the geometry $X$ and the displacement $u$:
$$ X(\xi) = \sum_{a=1}^{N_{nodes}} N_a(\xi) X_a $$
$$ u(\xi) = \sum_{a=1}^{N_{nodes}} N_a(\xi) u_a $$

### Lagrange vs. Serendipity
- **Lagrange** elements (Quad09, Hex27) use a full tensor product of 1D
  polynomials. They include internal nodes (bubble functions).
- **Serendipity** elements (Quad8, Hex20) only have nodes on the boundary.
  
This implementation focuses on **Lagrange** (Quad09, Hex27) and standard
Simplex (Tria06, Tet10) elements for robustness.

## 2. Integration (Quadrature)
Quadratic elements require higher-order Gaussian quadrature to integrate the
stiffness matrix exactly (or sufficiently accurately) because the integrand
($B^T D B$) has a higher polynomial degree.
- **Quad09**: Typically $3 \times 3$ Gauss points.
- **Hex27**: Typically $3 \times 3 \times 3$ Gauss points.
- **Tria06**: typically 3 or 4 points (Strang rules).
- **Tet10**: typically 4 or 5 points.

## 3. Implementation with Automatic Differentiation
Instead of hardcoding the gradient matrix $B$, we define the shape functions $N(\xi)$
and use Dual numbers (via `adiff`) to compute the spatial derivatives:
$$ \nabla_X N = J^{-1} \nabla_\xi N $$
where $J = \frac{\partial X}{\partial \xi}$ is the Jacobian computed automatically.
=#

# ---------------------------------------------------------------------------
# 2D Elements
# ---------------------------------------------------------------------------

"""
    Tria06(nodes, p0; mat=Materials.Hooke())

Constructs a 6-node quadratic triangular element.

**Nodes**: 3 corners followed by 3 mid-side nodes.
**Quadrature**: 3-point rule (degree 2).
"""
function Tria06(nodes::Vector{<:Integer}, 
               p0::Vector{Vector{T}};
               mat=Materials.Hooke()) where T<:Number

  # Quadratic Shape Functions for Triangle (L1, L2, L3 are barycentric coords)
  # N1 = L1(2L1 - 1)
  # N4 = 4L1L2 ...
  # Mapping from xi, eta to Barycentric: L1 = 1-xi-eta, L2 = xi, L3 = eta
  function N(ξ, η) 
      λ1, λ2, λ3 = 1.0 - ξ - η, ξ, η
      [
       λ1 * (2λ1 - 1),  # Node 1
       λ2 * (2λ2 - 1),  # Node 2
       λ3 * (2λ3 - 1),  # Node 3
       4 * λ1 * λ2,     # Node 4 (Edge 1-2)
       4 * λ2 * λ3,     # Node 5 (Edge 2-3)
       4 * λ3 * λ1      # Node 6 (Edge 3-1)
      ]
  end

  # 3-point Quadrature (Order 2) - Midpoints of edges
  # Points (1/6, 1/6), (2/3, 1/6), (1/6, 2/3) with weight 1/3
  GPs = [([1/6, 1/6], 1/3), ([2/3, 1/6], 1/3), ([1/6, 2/3], 1/3)]
  
  nGP = length(GPs)
  Nx  = Matrix{Vector{T}}(undef, 1, nGP) # 1 row, nGP cols for compatibility
  Ny  = Matrix{Vector{T}}(undef, 1, nGP)
  wgt = Matrix{T}(undef, 1, nGP)
  A   = zero(T)

  for (ii, (Pii, wii)) in enumerate(GPs)
      # 1. Evaluate shape functions N at Gauss point using AD for derivatives
      N_dual = N(adiff.D1(Pii)...)
      
      # 2. Map geometry: X = sum(N_a * X_a)
      p = sum(N_dual[a] * p0[a] for a in 1:6)
      
      # 3. Compute Jacobian J = dX/dξ
      J = SMatrix{2,2}(p[i].g[j] for i in 1:2, j in 1:2)
      det_J = detJ(J)
      
      # 4. Compute spatial gradients: ∇N = J^(-T) * ∇_ξ N
      # Nxy has columns [dN/dx, dN/dy]
      Nxy = J \ hcat(adiff.grad.(N_dual)...)

      Nx[1, ii]  = Nxy[1, :]
      Ny[1, ii]  = Nxy[2, :]
      wgt[1, ii] = 0.5 * abs(det_J) * wii # 0.5 factor for triangle area
      A += wgt[1, ii]
  end

  # Flatten tuples for C2DE constructor
  C2DE(nodes, tuple(Nx[1,:]...), tuple(Ny[1,:]...), tuple(wgt[1,:]...), A, mat, 2)
end

"""
    Quad09(nodes, p0; mat=Materials.Hooke())

Constructs a 9-node quadratic Lagrangian quadrilateral element.

**Nodes**: 4 corners, 4 mid-sides, 1 center.
**Quadrature**: 3x3 Gauss-Legendre.
"""
function Quad09(nodes::Vector{<:Integer}, 
               p0::Vector{Vector{T}};
               mat=Materials.Hooke()) where T<:Number

  # 1D Quadratic Lagrange polynomials
  # L1 = ξ(ξ-1)/2, L2 = (1-ξ^2), L3 = ξ(ξ+1)/2
  poly(ξ) = [0.5*ξ*(ξ-1), (1-ξ^2), 0.5*ξ*(ξ+1)]

  # Tensor product shape functions
  function N(ξ, η)
    N1, N2 = poly(ξ), poly(η)
    [N1[1]N2[1], N1[3]N2[1], N1[3]N2[3], N1[1]N2[3],
     N1[2]N2[1], N1[3]N2[2], N1[2]N2[3], N1[1]N2[2],
     N1[2]N2[2]]
  end

  # 3-point Gauss rule (coords, weights)
  g_coords = [-0.774596669241483, 0.0, 0.774596669241483] # sqrt(3/5)
  g_weights = [0.555555555555556, 0.888888888888889, 0.555555555555556] # 5/9, 8/9, 5/9
  
  # val = 0.577350269189626 # sqrt(1/3)
  # g_coords  = [-val, val]
  # g_weights = [1.0, 1.0]

  GP  = zip(g_coords, g_weights)
  nGP = length(g_weights)  
  Nx  = Matrix{Vector{T}}(undef, nGP, nGP)
  Ny  = Matrix{Vector{T}}(undef, nGP, nGP)
  wgt = Matrix{T}(undef, nGP, nGP)
  A   = zero(T)

  for (ii, (ξ, wξ)) in enumerate(GP), (jj, (η, wη)) in enumerate(GP)
      N_dual = N(adiff.D1([ξ, η])...)
      
      p = sum(N_dual[a] * p0[a] for a in 1:9)
      J = SMatrix{2,2}(p[ii].g[jj] for jj in 1:2, ii in 1:2)
      
      Nxy = J \ hcat(adiff.grad.(N_dual)...)

      Nx[ii, jj]  = Nxy[1, :]
      Ny[ii, jj]  = Nxy[2, :]
      wgt[ii, jj] = detJ(J) * wξ * wη
      A          += wgt[ii, jj]
  end

  C2DE(nodes, tuple(vec(Nx)...), tuple(vec(Ny)...), 
       tuple(vec(wgt)...), A, mat, 2)
end

"""
    Quad08(nodes, p0; mat=Materials.Hooke())

Constructs an 8-node Serendipity quadratic quadrilateral.
Nodes 1-4: Corners (CCW). Nodes 5-8: Midsides.
Uses 3x3 Gauss-Legendre quadrature.
"""
function Quad08(nodes::Vector{<:Integer}, 
                p0::Vector{Vector{T}};
                mat=Materials.Hooke()) where T<:Number

    # 3x3 Gauss Rule
    gp_loc = 0.774596669241483 # sqrt(3/5)
    gp_w1  = 0.555555555555556 # 5/9
    gp_w2  = 0.888888888888889 # 8/9
    
    GPs = [(-gp_loc, gp_w1), (0.0, gp_w2), (gp_loc, gp_w1)]
    
    # Nodal coordinates in parent space
    # 1(-1,-1), 2(1,-1), 3(1,1), 4(-1,1)
    # 5(0,-1), 6(1,0), 7(0,1), 8(-1,0)
    node_xi  = [-1,  1,  1, -1,  0,  1,  0, -1]
    node_eta = [-1, -1,  1,  1, -1,  0,  1,  0]

    nGP = 9 
    Nx  = Array{Array{T,1},2}(undef, 3, 3)
    Ny  = Array{Array{T,1},2}(undef, 3, 3)
    wgt = Array{T,2}(undef, 3, 3)
    A   = zero(T)

    for (ii, (xi, w_xi)) in enumerate(GPs), (jj, (eta, w_eta)) in enumerate(GPs)
        
        d_xi  = adiff.D1(xi)
        d_eta = adiff.D1(eta)
        
        Nd = Vector{typeof(d_xi)}(undef, 8)
        
        # Corner Nodes
        for k in 1:4 
            Nd[k] = 0.25 * (1 + d_xi*node_xi[k]) * (1 + d_eta*node_eta[k]) * (d_xi*node_xi[k] + d_eta*node_eta[k] - 1)
        end
        # Midside Nodes
        for k in 5:8 
            if node_xi[k] == 0
                Nd[k] = 0.5 * (1 - d_xi^2) * (1 + d_eta*node_eta[k])
            else
                Nd[k] = 0.5 * (1 + d_xi*node_xi[k]) * (1 - d_eta^2)
            end
        end

        p_dual = sum(Nd[k] * p0[k] for k in 1:8)
        J = [p_dual[r].g[c] for c in 1:2, r in 1:2]
        
        grads_L = hcat(adiff.grad.(Nd)...)
        grads_x = J \ grads_L
        
        val_w = detJ(SMatrix{2,2}(J)) * w_xi * w_eta
        wgt[ii, jj] = val_w
        Nx[ii, jj]  = grads_x[1,:]
        Ny[ii, jj]  = grads_x[2,:]
        A += val_w
    end

    C2DE(nodes, tuple(vec(Nx)...), tuple(vec(Ny)...), tuple(vec(wgt)...), A, mat ,2)
end

# ---------------------------------------------------------------------------
# 3D Elements
# ---------------------------------------------------------------------------

"""
    Tet10(nodes, p0; mat=Materials.Hooke())

Constructs a 10-node quadratic tetrahedron.

**Nodes**: 4 corners, 6 mid-edge nodes.
**Quadrature**: 4-point rule (Keast rule, precision 2/3).
"""
function Tet10(nodes::Vector{<:Integer}, 
               p0::Vector{Vector{T}};
               mat=Materials.Hooke()) where T<:Number

  # Barycentric shape functions for Tet10
  # λ1 = 1-ξ-η-ζ, λ2=ξ, λ3=η, λ4=ζ
  function N(ξ, η, ζ)
      λ = [1.0 - ξ - η - ζ, ξ, η, ζ]
      [
       λ[1]*(2λ[1]-1), λ[2]*(2λ[2]-1), λ[3]*(2λ[3]-1), λ[4]*(2λ[4]-1), # Corners
       4λ[1]λ[2], 4λ[2]λ[3], 4λ[3]λ[1], # Base edges
       4λ[1]λ[4], 4λ[2]λ[4], 4λ[3]λ[4]  # Vertical edges
      ]
  end

  # Quadrature (Standard 4-point rule for Tet)
  a = 0.5854101966249685
  b = 0.1381966011250105
  w = 1.0/24.0 # Volume is 1/6, sum of weights = 1/6. (1/24 * 4 = 1/6)
  GPs = [
      ([a, b, b], w),
      ([b, a, b], w),
      ([b, b, a], w),
      ([b, b, b], w)
  ]

  nGP = length(GPs)
  Nx  = Array{Vector{T}}(undef, nGP)
  Ny  = Array{Vector{T}}(undef, nGP)
  Nz  = Array{Vector{T}}(undef, nGP)
  wgt = Vector{T}(undef, nGP)
  V   = zero(T)

  for (ii, (coords, wii)) in enumerate(GPs)
      N_dual = N(adiff.D1(coords)...)
      
      p = sum(N_dual[a] * p0[a] for a in 1:10)
      J = SMatrix{3,3}(p[i].g[j] for i in 1:3, j in 1:3)
      
      Nxyz = J \ hcat(adiff.grad.(N_dual)...)

      Nx[ii] = Nxyz[1, :]
      Ny[ii] = Nxyz[2, :]
      Nz[ii] = Nxyz[3, :]
      wgt[ii] = abs(detJ(J)) * wii # J includes the factor 6 implicitly via integration? 
      # Note: Standard Tet volume is 1/6 det(J_linear). Here J varies. 
      # We use standard Gaussian weights summing to 1/6 * detJ roughly.
      
      V += wgt[ii]
  end

  C3DE(nodes, tuple(Nx...), tuple(Ny...), tuple(Nz...), tuple(wgt...), V, mat, 2)
end

"""
    Hex27(nodes, p0; mat=Materials.Hooke())

Constructs a 27-node quadratic Lagrange hexahedron.
**Quadrature**: 3x3x3 Gauss-Legendre.
"""
function Hex27(nodes::Vector{<:Integer}, 
               p0::Vector{Vector{T}};
               mat=Materials.Hooke()) where T<:Number

  poly(ξ) = [0.5*ξ*(ξ-1), (1-ξ^2), 0.5*ξ*(ξ+1)]

  # Tensor product 3D
  function N(ξ, η, ζ)
      Nx, Ny, Nz = poly(ξ), poly(η), poly(ζ)
      vals = Vector{T}(undef, 27)
      idx = 1
      # Lexicographical ordering usually expected for generic generation
      # Ensure input nodes match this order:
      # Iterate Z, then Y, then X (or matching your mesh generator)
      for k in 1:3, j in 1:3, i in 1:3
          vals[idx] = Nx[i] * Ny[j] * Nz[k]
          idx += 1
      end
      return vals
  end

  g_coords = [-0.774596669241483, 0.0, 0.774596669241483]
  g_weights = [0.555555555555556, 0.888888888888889, 0.555555555555556]
  GP = zip(g_coords, g_weights)
  
  nGP = 3
  Nx  = Array{Vector{T}, 3}(undef, nGP, nGP, nGP)
  Ny  = Array{Vector{T}, 3}(undef, nGP, nGP, nGP)
  Nz  = Array{Vector{T}, 3}(undef, nGP, nGP, nGP)
  wgt = Array{T, 3}(undef, nGP, nGP, nGP)
  V   = zero(T)

  for (k, (ζ, wζ)) in enumerate(GP), (j, (η, wη)) in enumerate(GP), (i, (ξ, wξ)) in enumerate(GP)
      N_dual = N(adiff.D1([ξ, η, ζ])...)
      
      p = sum(N_dual[a] * p0[a] for a in 1:27)
      J = SMatrix{3,3}(p[r].g[c] for r in 1:3, c in 1:3)
      
      Nxyz = J \ hcat(adiff.grad.(N_dual)...)

      Nx[i, j, k] = Nxyz[1, :]
      Ny[i, j, k] = Nxyz[2, :]
      Nz[i, j, k] = Nxyz[3, :]
      wgt[i, j, k] = detJ(J) * wξ * wη * wζ
      V += wgt[i, j, k]
  end

  C3DE(nodes, 
       tuple(vec(Nx)...), 
       tuple(vec(Ny)...), 
       tuple(vec(Nz)...), 
       tuple(vec(wgt)...), V, mat, 2)
end

#=
"""
    Hex20(nodes, p0; mat=Materials.Hooke())

Constructs a 20-node Serendipity quadratic hexahedron.
Uses 3x3x3 Gauss-Legendre quadrature.
"""
function Hex20(nodes::Vector{<:Integer}, 
               p0::Vector{Vector{T}};
               mat=Materials.Hooke()) where T<:Number

    gp_loc = 0.774596669241483
    gp_w1  = 0.555555555555556
    gp_w2  = 0.888888888888889
    GPs = [(-gp_loc, gp_w1), (0.0, gp_w2), (gp_loc, gp_w1)]

    Nx  = Array{Array{T,1},3}(undef, 3, 3, 3)
    Ny  = Array{Array{T,1},3}(undef, 3, 3, 3)
    Nz  = Array{Array{T,1},3}(undef, 3, 3, 3)
    wgt = Array{T,3}(undef, 3, 3, 3)
    V   = zero(T)

    # Standard 20-node definition
    # Corners 1-8
    ξ_n = [-1, 1, 1, -1, -1, 1, 1, -1]
    η_n = [-1, -1, 1, 1, -1, -1, 1, 1]
    ζ_n = [-1, -1, -1, -1, 1, 1, 1, 1]
    # Midsides 9-20 (Edges)
    append!(ξ_n, [0, 1, 0, -1, 0, 1, 0, -1, -1, 1, 1, -1])
    append!(η_n, [-1, 0, 1, 0, -1, 0, 1, 0, -1, -1, 1, 1])
    append!(ζ_n, [-1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0])

    for (ii, (ξ, wξ)) in enumerate(GPs), (jj, (η, wη)) in enumerate(GPs), (kk, (ζ, wζ)) in enumerate(GPs)
        
        dξ, dη, dζ = adiff.D1([ξ, η, ζ])
        Nd = Vector{typeof(dξ)}(undef, 20)

        for k in 1:8 # Corners
             Nd[k] = 0.125 * (1 + dξ*ξ_n[k]) * (1 + dη*η_n[k]) * (1 + dζ*ζ_n[k]) * (dξ*ξ_n[k] + dη*η_n[k] + dζ*ζ_n[k] - 2)
        end
        for k in 9:20 # Midsides
            if ξ_n[k] == 0
                Nd[k] = 0.25 * (1 - dξ^2) * (1 + dη*η_n[k]) * (1 + dζ*ζ_n[k])
            elseif η_n[k] == 0
                Nd[k] = 0.25 * (1 + dξ*ξ_n[k]) * (1 - dη^2) * (1 + dζ*ζ_n[k])
            else
                Nd[k] = 0.25 * (1 + dξ*ξ_n[k]) * (1 + dη*η_n[k]) * (1 - dζ^2)
            end
        end

        p_dual = sum(Nd[k] * p0[k] for k in 1:20)
        J = [p_dual[r].g[c] for c in 1:3, r in 1:3]
        
        grads_L = hcat(adiff.grad.(Nd)...)
        grads_x = J \ grads_L
        
        w_val = detJ(SMatrix{3,3}(J)) * wξ * wη * wζ
        wgt[ii,jj,kk] = w_val
        Nx[ii,jj,kk]  = grads_x[1,:]
        Ny[ii,jj,kk]  = grads_x[2,:]
        Nz[ii,jj,kk]  = grads_x[3,:]
        V += w_val
    end
    
    C3DE(nodes, 
         tuple(vec(Nx)...), 
         tuple(vec(Ny)...), 
         tuple(vec(Nz)...), tuple(vec(wgt)...), V, mat, 2)
end
=#
# ---------------------------------------------------------------------------
# Energy Evaluation (Fallback Check)
# ---------------------------------------------------------------------------
# The standard `getϕ(elem::CEElem, u)` defined in `elasticelements.jl` [cite: 167]
# iterates over all quadrature points stored in `elem.wgt` and `elem.∇N`.
# Since the constructors above correctly populate these fields with the 
# high-order quadrature data, the existing functions will correctly evaluate 
# the integral for 2nd order elements without modification.
#
# No specialized `getϕ` is required here.
