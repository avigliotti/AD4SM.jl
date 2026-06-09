__precompile__()

module Elements

using ..AD4SM.adiff
import ..Materials.getϕ
using ..Materials

using StaticArrays, SparseArrays
using LinearAlgebra:I,det
import LinearAlgebra.×

export AbstractElement, AbstractContinuumElem, AbstractCElem
export CElem, CEElem, CPElem, C1DE, C2DE, C3DE, C1DP, C2DP, C3DP,
       C1D, C2D, C3D, LagrangePoly
export getϕ, getσ, getP, detJ, getF, ×, getV
export get∇n
export getI₁, getI₂, getĪ₁, getĪ₂


# ---------------------------------------------------------------------------
# Inline static dot product
# ---------------------------------------------------------------------------

@inline function dot(a::SVector{N,S}, b::SVector{N,T}) where {N,S,T}
  s = zero(T)
  @inbounds @simd for ii in 1:N
    s += a[ii] * b[ii]
  end
  return s
end

@inline function dot(a::SMatrix{N,M,T}, b::SMatrix{N,M,T}) where {N,M,T}
  s = zero(T)
  @inbounds  @simd for ii in 1:N*M
    s += a[ii] * b[ii]
  end
  return s
end

const ⋅ = dot
# ---------------------------------------------------------------------------
# Abstract type hierarchy
# ---------------------------------------------------------------------------

abstract type AbstractElement end
abstract type AbstractContinuumElem <: AbstractElement end
# Updated AbstractCElem to include Order parameter (O)
abstract type AbstractCElem{D,P,M,T,N,O} <: AbstractContinuumElem end

"""
AbstractASElem{P,M,T,N,O}

Abstract supertype for axisymmetric continuum elements.
The reference mesh lives in the meridional (r,z) plane (2 DOFs/node),
but the deformation gradient is always 3×3 (including the hoop stretch).

Type parameters mirror AbstractCElem but D is implicit (always 2 in reference):
- `P` : number of Gauss points
- `M` : material type
- `T` : numeric type
- `N` : number of element nodes
- `O` : element order  (1 = linear, 2 = quadratic, ...)
"""
abstract type AbstractASElem{P,M,T,N,O} <: AbstractContinuumElem end

# ---------------------------------------------------------------------------
# CEElem — Mechanical elements
# ---------------------------------------------------------------------------

"""
CEElem{D,P,M,T,N,O}

Generic D-dimensional continuum finite element for displacement-based
mechanical analysis.
Type parameters:
- `D` : spatial dimension (1, 2, or 3)
- `P` : number of quadrature points
- `M` : material model type
- `T` : numeric type
- `N` : number of element nodes
- `O` : Element order (1=Linear, 2=Quadratic, etc.)

Fields:
- `nodes` : vector of nodal IDs (any integer type)
- `∇N`    : NTuple{D,NTuple{P,SVector{N,T}}} — shape-function derivatives
- `wgt`   : NTuple{P,T} — quadrature weights
- `V`     : reference volume/area/length
- `mat`   : material model
"""
struct CEElem{D,P,M,T,N,O} <: AbstractCElem{D,P,M,T,N,O}
  nodes::Vector{I} where I
  ∇N::NTuple{D,NTuple{P,SVector{N,T}}}
  wgt::NTuple{P,T}
  V::T
  mat::M
end

# ---------------------------------------------------------------------------
# CPElem — Phase-field / scalar-field elements
# ---------------------------------------------------------------------------

"""
CPElem{D,P,M,T,N,O}

Generic D-dimensional continuum finite element for scalar fields
(e.g., phase, concentration, or temperature).
Type parameters:
- `D`   : spatial dimension (1, 2, or 3)
- `P`   : number of quadrature points
- `M`   : material model type
- `T`   : numeric type
- `N`   : number of element nodes
- `O` : Element order (1=Linear, 2=Quadratic, etc.)

Fields:
- `nodes` : vector of nodal IDs (any integer type)
- `N`     : NTuple{P,SVector{N,T}} — shape-function values
- `∇N`    : NTuple{D,NTuple{P,SVector{N,T}}} — shape-function derivatives
- `wgt`   : NTuple{P,T} — quadrature weights
- `V`     : reference volume/area/length
- `mat`   : material model
"""
struct CPElem{D,P,M,T,N,O} <: AbstractCElem{D,P,M,T,N,O}
  nodes::Vector{I} where I
  N::NTuple{P,SVector{N,T}}
  ∇N::NTuple{D,NTuple{P,SVector{N,T}}}
  wgt::NTuple{P,T}
  V::T
  mat::M
end

"""
CASElem{P,M,T,N,O}

Axisymmetric continuum element.

Fields:
- `nodes` : nodal connectivity (length N)
- `N0`    : NTuple{P,SVector{N,T}}  — shape-function values at each GP
- `∇N`    : NTuple{2,NTuple{P,SVector{N,T}}} — ∂N/∂r and ∂N/∂z at each GP
- `r_GP`  : NTuple{P,T}  — reference radial coordinate at each GP
- `wgt`   : NTuple{P,T}  — integration weights (= det(J)*2π*r_GP*w_ref)
- `V`     : reference volume  (= ∫ 2π r dA over element)
- `mat`   : material model
"""
struct CASElem{D,P,M,T,N,O} <: AbstractCElem{D,P,M,T,N,O}
  nodes :: Vector{<:Integer}
  N     :: NTuple{P, SVector{N,T}}
  ∇N    :: NTuple{D, NTuple{P, SVector{N,T}}}
  r_GP  :: NTuple{P, T}
  wgt   :: NTuple{P, T}
  V     :: T
  mat   :: M
end

# ---------------------------------------------------------------------------
# Type aliases for convenience
# ---------------------------------------------------------------------------

# Aliases now implicitly accept the Order parameter, or default to Any
const C1DE{P,M,T,N,O} = CEElem{1,P,M,T,N,O}
const C2DE{P,M,T,N,O} = CEElem{2,P,M,T,N,O}
const C3DE{P,M,T,N,O} = CEElem{3,P,M,T,N,O}

const C1DP{P,M,T,N,O} = CPElem{1,P,M,T,N,O}
const C2DP{P,M,T,N,O} = CPElem{2,P,M,T,N,O}
const C3DP{P,M,T,N,O} = CPElem{3,P,M,T,N,O}

const CASE{P,M,T,N,O} = CASElem{2,P,M,T,N,O}

const C1D = Union{C1DE,C1DP}
const C2D = Union{C2DE,C2DP}
const C3D = Union{C3DE,C3DP}

# const CElem{D,P,M,T,N} = Union{CEElem{D,P,M,T,N}, CPElem{D,P,M,T,N}}
const CElem = AbstractCElem

# ---------------------------------------------------------------------------
# Constructors for CEElem (Backward Compatible, defaults to O=1)
# ---------------------------------------------------------------------------

"""
Create a 1D mechanical element.
"""
function C1DE(nodes, Nx, wgt, V, mat, ord=1)
  P  = length(wgt)
  Nn = length(Nx[1])
  return CEElem{1,P,typeof(mat),eltype(wgt),Nn,ord}(
                                                 
                nodes,
                (ntuple(ii -> SVector{Nn}(Nx[ii]), P),),
                wgt, V, mat)
end

"""
Create a 2D mechanical element.
"""
function C2DE(nodes, Nx, Ny, wgt, V, mat, ord=1)
  P  = length(wgt)
  Nn = length(Nx[1])
  return CEElem{2,P,typeof(mat),eltype(wgt),Nn,ord}(
                nodes,
                (ntuple(ii -> SVector{Nn}(Nx[ii]), P),
                 ntuple(ii -> SVector{Nn}(Ny[ii]), P) ),
                wgt, V, mat)
end

"""
Create a 3D mechanical element.
"""
function C3DE(nodes, Nx, Ny, Nz, wgt, V, mat, ord=1)
  P = length(wgt)
  Nn = length(Nx[1])
  return CEElem{3,P,typeof(mat),eltype(wgt),Nn,ord}(
                nodes,
                ( ntuple(ii -> SVector{Nn}(Nx[ii]), P),
                 ntuple(ii -> SVector{Nn}(Ny[ii]), P),
                 ntuple(ii -> SVector{Nn}(Nz[ii]), P) ),
                wgt, V, mat)
end

# ---------------------------------------------------------------------------
# Constructors for CPElem (Backward Compatible, defaults to O=1)
# ---------------------------------------------------------------------------

"""
Create a 1D scalar-field element.
"""
function C1DP(nodes, N, Nx, wgt, V, mat, ord=1)
  P = length(wgt)
  Nn = length(N[1])
  return CPElem{1,P,typeof(mat),eltype(wgt),Nn,ord}(
                nodes,
                ntuple(ii -> SVector{Nn}(N[ii]), P),
                (ntuple(ii -> SVector{Nn}(Nx[ii]), P),),
                 wgt, V, mat)
end

"""
Create a 2D scalar-field element.
"""
function C2DP(nodes, N, Nx, Ny, wgt, V, mat, ord=1)
  P = length(wgt)
  Nn = length(N[1])
  return CPElem{2,P,typeof(mat),eltype(wgt),Nn,ord}(
                nodes,
                ntuple(ii -> SVector{Nn}(N[ii]), P),
                ( ntuple(ii -> SVector{Nn}(Nx[ii]), P),
                ntuple(ii -> SVector{Nn}(Ny[ii]), P) ),
                wgt, V, mat)
end

"""
Create a 3D scalar-field element.
"""
function C3DP(nodes, N0, Nx, Ny, Nz, wgt, V::T, mat::M, O::Int=1) where {M<:Material, T}
  P = length(wgt)
  N = length(nodes)
  C3DP{P,M,T,N,O}(nodes,
                    ntuple(ii->SVector{N}(N0[ii]), P),
                    (ntuple(ii->SVector{N}(Nx[ii]), P),
                     ntuple(ii->SVector{N}(Ny[ii]), P),
                     ntuple(ii->SVector{N}(Nz[ii]), P) ),
                    wgt, V, mat)
end

# ---------------------------------------------------------------------------
# Constructor helper  CASELEM(nodes, N0, Nr, Nz, r_GP, wgt, V, mat, ord=1)
# ---------------------------------------------------------------------------
"""
    CASElem(nodes, N0, Nr, Nz, r_GP, wgt, V, mat, ord=1)

Low-level constructor that converts plain array-of-arrays into the
fully typed `CASElem`.  All tuple-of-arrays arguments accept the same
formats used by `C2DP`.
"""
function CASE(nodes, N0, Nr, Nz, r_GP, wgt, V, mat, ord=1)
  P   = length(wgt)
  Nn  = length(N0[1])
  M   = typeof(mat)
  T   = eltype(wgt)
  CASElem{2,P,M,T,Nn,ord}(
    nodes,
    ntuple(ii -> SVector{Nn,T}(N0[ii]),   P),
    ( ntuple(ii -> SVector{Nn,T}(Nr[ii]), P),
      ntuple(ii -> SVector{Nn,T}(Nz[ii]), P) ),
    ntuple(ii -> T(r_GP[ii]), P),
    ntuple(ii -> T(wgt[ii]),  P),
    T(V),
    mat)
end

# ---------------------------------------------------------------------------
# Generic Lagrange Element Generation (High Order Support)
# ---------------------------------------------------------------------------

"""
    get_gauss_points(order::Int)

Returns tuple of (points, weights) for Gauss-Legendre quadrature sufficient
for the given element order.
"""
function get_gauss_points(order::Int)
  # Simple lookup for common orders
  if order == 1
    return ([-0.577350269189626, 0.577350269189626], [1.0, 1.0])
  elseif order == 2
    return ([-0.774596669241483, 0.0, 0.774596669241483], 
            [0.555555555555556, 0.888888888888889, 0.555555555555556])
  elseif order == 3
    return ([-0.861136311594053, -0.339981043584856, 0.339981043584856, 0.861136311594053],
            [0.347854845137454, 0.652145154862546, 0.652145154862546, 0.347854845137454])
  else
    # Fallback to linear if undefined (or implement generic recurrence)
    return ([-0.577350269189626, 0.577350269189626], [1.0, 1.0])
  end
end

"""
    lagrange_basis(x, order, node_index)

Evaluates the 1D Lagrange polynomial of degree `order` for node `node_index` at coordinate `x`.
Nodes are assumed equidistant in [-1, 1].
"""
function lagrange_basis(x, order, node_index)
  xi = range(-1.0, stop=1.0, length=order+1)
  val = one(x)
  @inbounds for j in 1:order+1
    if j != node_index
      val *= (x - xi[j]) / (xi[node_index] - xi[j])
    end
  end
  return val
end

"""
    LagrangePoly(nodes, p0, mat; order=1, dim=2)

Generic constructor for Tensor Product Lagrange elements of arbitrary order.
Supports 1D (Line), 2D (Quad), and 3D (Hex).

Arguments:
- `nodes`: Vector of node indices.
- `p0`: Vector of node coordinates (only used for mapping).
- `mat`: Material.
- `order`: Polynomial order (1=Linear, 2=Quadratic, etc.).
- `dim`: Dimension of the element (1, 2, or 3).
"""
function LagrangePoly(nodes::Vector{<:Integer}, 
                      p0::Vector{<:AbstractVector{T}};
                      mat, order::Int=1, dim::Int=2) where T<:Number

  (gp_xi, gp_w) = get_gauss_points(order) # Use order+1 points for integration
  nGP_1d = length(gp_xi)
  nNodes_1d = order + 1

  # Generic shape function generation using Automatic Differentiation
  if dim == 1
    nGP = nGP_1d
    Nx = Vector{Vector{T}}(undef, nGP)
    wgt = Vector{T}(undef, nGP)
    Vol = zero(T)

    for (i, (xi, wi)) in enumerate(zip(gp_xi, gp_w))
      # Shape functions and derivatives via AD
      xi_ad = adiff.D1(xi)
      N_val = [lagrange_basis(xi_ad, order, a) for a in 1:nNodes_1d]

      # Map to physical space
      p = sum(N_val[a] * p0[a] for a in 1:nNodes_1d)
      J = p.g[1] # dx/dxi

      # Gradients in physical space: dN/dx = dN/dxi * dxi/dx
      grad_N = [N_val[a].g[1] / J for a in 1:nNodes_1d]

      Nx[i] = grad_N
      wgt[i] = abs(J) * wi
      Vol += wgt[i]
    end
    return C1DE(nodes, Nx, wgt, Vol, mat, order)

  elseif dim == 2
    nGP = nGP_1d * nGP_1d
    Nx = Matrix{Vector{T}}(undef, nGP_1d, nGP_1d)
    Ny = Matrix{Vector{T}}(undef, nGP_1d, nGP_1d)
    wgt = Matrix{T}(undef, nGP_1d, nGP_1d)
    Vol = zero(T)

    for (i, (xi, wxi)) in enumerate(zip(gp_xi, gp_w))
      for (j, (eta, weta)) in enumerate(zip(gp_xi, gp_w))

        # Tensor product shape functions
        xi_ad = adiff.D1([xi, eta])
        N_val = Vector{Any}(undef, nNodes_1d * nNodes_1d)
        idx = 1
        # Standard tensor product ordering (lexicographical)
        for b in 1:nNodes_1d # eta
          for a in 1:nNodes_1d # xi
            N_val[idx] = lagrange_basis(xi_ad[1], order, a) * lagrange_basis(xi_ad[2], order, b)
            idx += 1
          end
        end

        p = sum(N_val[k] * p0[k] for k in 1:length(N_val))
        J = [p.g[1] p.g[2];] # Jacobian matrix [dx/dxi dx/deta; dy/dxi dy/deta] -> Transposed in memory?
        # J here is actually [dx/dxi dy/dxi; dx/deta dy/deta] if using AD gradient logic
        # Let's be precise: p is a vector of D1s (x, y). 
        # p[1].g is [dx/dxi, dx/deta]. 
        J_mat = SMatrix{2,2}(p[m].g[n] for n in 1:2, m in 1:2) # [dx/dxi dy/dxi; dx/deta dy/deta]

        # Compute gradients in physical space: ∇N = J^(-T) * ∇N_parametric
        grads_param = hcat([adiff.grad(n) for n in N_val]...) # 2 x nNodes
        grads_phys = J_mat \ grads_param

        Nx[i, j] = grads_phys[1, :]
        Ny[i, j] = grads_phys[2, :]
        wgt[i, j] = detJ(J_mat) * wxi * weta
        Vol += wgt[i, j]
      end
    end
    return C2DE(nodes, 
                tuple(vec(Nx)...), 
                tuple(vec(Ny)...), 
                tuple(vec(wgt)...), 
                Vol, mat, order)

  elseif dim == 3
    # Similar logic for 3D Hex
    nGP = nGP_1d^3
    Nx = Array{Vector{T}, 3}(undef, nGP_1d, nGP_1d, nGP_1d)
    Ny = Array{Vector{T}, 3}(undef, nGP_1d, nGP_1d, nGP_1d)
    Nz = Array{Vector{T}, 3}(undef, nGP_1d, nGP_1d, nGP_1d)
    wgt = Array{T, 3}(undef, nGP_1d, nGP_1d, nGP_1d)
    Vol = zero(T)

    for (i, (xi, wxi)) in enumerate(zip(gp_xi, gp_w))
      for (j, (eta, weta)) in enumerate(zip(gp_xi, gp_w))
        for (k, (zeta, wzeta)) in enumerate(zip(gp_xi, gp_w))

          xi_ad = adiff.D1([xi, eta, zeta])
          N_val = Vector{Any}(undef, nNodes_1d^3)
          idx = 1
          for c in 1:nNodes_1d # zeta
            for b in 1:nNodes_1d # eta
              for a in 1:nNodes_1d # xi
                N_val[idx] = lagrange_basis(xi_ad[1], order, a) * lagrange_basis(xi_ad[2], order, b) *
                lagrange_basis(xi_ad[3], order, c)
                idx += 1
              end
            end
          end

          p = sum(N_val[n] * p0[n] for n in 1:length(N_val))
          J_mat = SMatrix{3,3}(p[m].g[n] for n in 1:3, m in 1:3)

          grads_param = hcat([adiff.grad(n) for n in N_val]...)
          grads_phys = J_mat \ grads_param

          Nx[i, j, k] = grads_phys[1, :]
          Ny[i, j, k] = grads_phys[2, :]
          Nz[i, j, k] = grads_phys[3, :]
          wgt[i, j, k] = detJ(J_mat) * wxi * weta * wzeta
          Vol += wgt[i, j, k]
        end
      end
    end
    return C3DE(nodes, 
                tuple(vec(Nx)...), 
                tuple(vec(Ny)...), 
                tuple(vec(Nz)...), 
                tuple(vec(wgt)...), 
                Vol, mat, order)
  else
    error("Dimension $dim not supported for Generic Lagrange Elements")
  end
end


include("./elements.toolkit.jl")
include("./elasticelements.jl")
include("./phasefieldelements.jl")

end # module Elements

