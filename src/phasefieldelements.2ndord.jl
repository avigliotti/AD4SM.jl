# ===========================================================================
# phasefieldelements.2ndord.jl
# ---------------------------------------------------------------------------
# Second-order (Quadratic) Phase-Field Elements
# ===========================================================================

#=
# Theoretical Background: Phase Field Elements

Phase-field elements interpolate a scalar order parameter, $d \in [0, 1]$, where
$d=0$ represents intact material and $d=1$ represents fully broken material.

## Free Energy Functional
The functional typically integrated is:
$$ \Psi(d, \nabla d) = \int_\Omega \left[ \frac{G_c}{2l_0} \left( d^2 + l_0^2 |\nabla d|^2 \right) + (1-d)^2 \psi_{elas}^+ \right] dV $$

## Discretization
Similar to the mechanical elements, we use quadratic shape functions:
$$ d(\xi) = \sum_{a=1}^{N_{nodes}} N_a(\xi) d_a $$
$$ \nabla d(\xi) = \sum_{a=1}^{N_{nodes}} \nabla N_a(\xi) d_a $$

Because the energy depends on $d$ directly (not just derivatives), the `CPElem`
struct stores the shape function values `N` at Gauss points, in addition to
`∇N` and `wgt`.

## Higher Order Benefits
Using quadratic interpolation for the phase field:
1. Improves the resolution of the diffuse crack profile (the $\tanh$ profile).
2. Ensures better compatibility if the displacement field is also quadratic.
=#

using ..Materials:PhaseField

# ---------------------------------------------------------------------------
# 2D Phase Field Elements
# ---------------------------------------------------------------------------

"""
    Tria6P(nodes, p0; mat=Materials.Hooke())

Constructs a 6-node quadratic triangular phase-field element.
"""
function Tria6P(nodes::Vector{<:Integer}, 
                p0::Vector{Vector{T}};
                mat=Materials.Hooke()) where T<:Number

  # Shape functions (Same as Tria6)
  function N(ξ, η) 
      λ = [1.0 - ξ - η, ξ, η]
      [λ[1]*(2λ[1]-1), λ[2]*(2λ[2]-1), λ[3]*(2λ[3]-1), 
       4λ[1]λ[2], 4λ[2]λ[3], 4λ[3]λ[1]]
  end

  GPs = [([1/6, 1/6], 1/3), ([2/3, 1/6], 1/3), ([1/6, 2/3], 1/3)]
  nGP = length(GPs)
  
  N0  = Matrix{Vector{T}}(undef, 1, nGP) # Store shape values
  Nx  = Matrix{Vector{T}}(undef, 1, nGP)
  Ny  = Matrix{Vector{T}}(undef, 1, nGP)
  wgt = Matrix{T}(undef, 1, nGP)
  A   = zero(T)

  for (ii, (Pii, wii)) in enumerate(GPs)
      N_dual = N(adiff.D1(Pii)...)
      p = sum(N_dual[a] * p0[a] for a in 1:6)
      J = SMatrix{2,2}(p[i].g[j] for i in 1:2, j in 1:2)
      Nxy = J \ hcat(adiff.grad.(N_dual)...)

      N0[1, ii]  = adiff.val.(N_dual) # Extract values
      Nx[1, ii]  = Nxy[1, :]
      Ny[1, ii]  = Nxy[2, :]
      wgt[1, ii] = 0.5 * abs(detJ(J)) * wii
      A += wgt[1, ii]
  end

  C2DP(nodes, tuple(N0[1,:]...), tuple(Nx[1,:]...), 
       tuple(Ny[1,:]...), tuple(wgt[1,:]...), A, mat, 2)
end

"""
    Quad9P(nodes, p0; mat=Materials.Hooke())

Constructs a 9-node quadratic phase-field quadrilateral.
"""
function Quad9P(nodes::Vector{<:Integer}, 
                p0::Vector{Vector{T}};
                mat=Materials.Hooke()) where T<:Number

  poly(ξ) = [0.5*ξ*(ξ-1), (1-ξ^2), 0.5*ξ*(ξ+1)]
  function N(ξ, η)
      Nx, Ny = poly(ξ), poly(η)
      [Nx[1]*Ny[1], Nx[3]*Ny[1], Nx[3]*Ny[3], Nx[1]*Ny[3],
       Nx[2]*Ny[1], Nx[3]*Ny[2], Nx[2]*Ny[3], Nx[1]*Ny[2],
       Nx[2]*Ny[2]]
  end

  g_coords = [-0.774596669241483, 0.0, 0.774596669241483]
  g_weights = [0.555555555555556, 0.888888888888889, 0.555555555555556]
  GP = zip(g_coords, g_weights)
  
  nGP = 3
  N0  = Matrix{Vector{T}}(undef, nGP, nGP)
  Nx  = Matrix{Vector{T}}(undef, nGP, nGP)
  Ny  = Matrix{Vector{T}}(undef, nGP, nGP)
  wgt = Matrix{T}(undef, nGP, nGP)
  A   = zero(T)

  for (ii, (ξ, wξ)) in enumerate(GP), (jj, (η, wη)) in enumerate(GP)
      N_dual = N(adiff.D1([ξ, η])...)
      p = sum(N_dual[a] * p0[a] for a in 1:9)
      J = SMatrix{2,2}(p[i].g[j] for i in 1:2, j in 1:2)
      Nxy = J \ hcat(adiff.grad.(N_dual)...)

      N0[ii, jj]  = adiff.val.(N_dual)
      Nx[ii, jj]  = Nxy[1, :]
      Ny[ii, jj]  = Nxy[2, :]
      wgt[ii, jj] = detJ(J) * wξ * wη
      A += wgt[ii, jj]
  end

  C2DP(nodes, tuple(vec(N0)...), tuple(vec(Nx)...), 
       tuple(vec(Ny)...), tuple(vec(wgt)...), A, mat, 2)
end

# ---------------------------------------------------------------------------
# 3D Phase Field Elements
# ---------------------------------------------------------------------------

"""
    Tet10P(nodes, p0; mat=Materials.Hooke())

Constructs a 10-node quadratic phase-field tetrahedron.
"""
function Tet10P(nodes::Vector{<:Integer}, 
                p0::Vector{Vector{T}};
                mat=Materials.Hooke()) where T<:Number

  function N(ξ, η, ζ)
      λ = [1.0 - ξ - η - ζ, ξ, η, ζ]
      [λ[1]*(2λ[1]-1), λ[2]*(2λ[2]-1), λ[3]*(2λ[3]-1), λ[4]*(2λ[4]-1), 
       4λ[1]λ[2], 4λ[2]λ[3], 4λ[3]λ[1], 4λ[1]λ[4], 4λ[2]λ[4], 4λ[3]λ[4]]
  end

  a, b, w = 0.5854101966249685, 0.1381966011250105, 1.0/24.0
  GPs = [([a, b, b], w), ([b, a, b], w), ([b, b, a], w), ([b, b, b], w)]
  nGP = length(GPs)
  
  N0  = Array{Vector{T}}(undef, nGP)
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

      N0[ii]  = adiff.val.(N_dual)
      Nx[ii]  = Nxyz[1, :]
      Ny[ii]  = Nxyz[2, :]
      Nz[ii]  = Nxyz[3, :]
      wgt[ii] = abs(detJ(J)) * wii
      V += wgt[ii]
  end

  C3DP(nodes, tuple(N0...), tuple(Nx...), 
       tuple(Ny...), tuple(Nz...), tuple(wgt...), V, mat, 2)
end

"""
    Hex27P(nodes, p0; mat=Materials.Hooke())

Constructs a 27-node quadratic phase-field hexahedron.
"""
function Hex27P(nodes::Vector{<:Integer}, 
                p0::Vector{Vector{T}};
                mat=Materials.Hooke()) where T<:Number

  poly(ξ) = [0.5*ξ*(ξ-1), (1-ξ^2), 0.5*ξ*(ξ+1)]
  function N(ξ, η, ζ)
      Nx, Ny, Nz = poly(ξ), poly(η), poly(ζ)
      vals = Vector{T}(undef, 27)
      idx = 1
      for k in 1:3, j in 1:3, i in 1:3
          vals[idx] = Nx[i] * Ny[j] * Nz[k]
          idx += 1
      end
      return vals
  end

  gp_c = [-0.774596669241483, 0.0, 0.774596669241483]
  gp_w = [0.555555555555556, 0.888888888888889, 0.555555555555556]
  GP = zip(gp_c, gp_w)
  
  nGP = 3
  N0  = Array{Vector{T}, 3}(undef, nGP, nGP, nGP)
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

      N0[i, j, k]  = adiff.val.(N_dual)
      Nx[i, j, k]  = Nxyz[1, :]
      Ny[i, j, k]  = Nxyz[2, :]
      Nz[i, j, k]  = Nxyz[3, :]
      wgt[i, j, k] = detJ(J) * wξ * wη * wζ
      V += wgt[i, j, k]
  end

  C3DP(nodes, tuple(vec(N0)...), tuple(vec(Nx)...), 
       tuple(vec(Ny)...), tuple(vec(Nz)...), 
       tuple(vec(wgt)...), V, mat, 2)
end

# ---------------------------------------------------------------------------
# Integral Evaluation
# ---------------------------------------------------------------------------
# The generic integral evaluation function `getϕ(elem::CPElem, u0, d0)` 
# located in `phasefieldelements.jl` [cite: 29] relies on the `elem.wgt` 
# and shape function arrays `elem.N`.
#
# As `Quad9P`, `Hex27P`, etc. return standard `CPElem` structures populated 
# with the correct 3x3 (or 3x3x3) quadrature rules and quadratic shape functions,
# the existing evaluation functions will inherently perform the correct high-order
# integration without modification.
