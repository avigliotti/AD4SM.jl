__precompile__()

module Elements

using LinearAlgebra, SparseArrays

using ..adiff, ..Materials
import ..Materials.getϕ

"""
Elements

Finite-element kernel for continuum and structural elements used within the
larger FE + automatic-differentiation framework.

This module provides:
- concrete element data types (C1D/C2D/C3D and their phase-field variants),
  structural element types (Rod, Beam),
- element-level kinematics: evaluation of deformation gradient F at
  quadrature points, Jacobians, and volume measures,
- utilities to assemble element contributions into global residuals and
  tangent (stiffness) matrices using adiff.D2 objects,
- helper routines for quadrature generation and element-level inertial terms.

Conventions and notes
- `u` fields are arranged columnwise per node: rows correspond to local DOFs,
  columns to nodes. Linear indexing (`vec(u)` / `LinearIndices(u)`) is used
  in assembly routines.
- Shape-function derivative arrays (Nx, Ny, Nz) are stored per integration
  point and used to compute ∇u and F = I + ∇u.
- Functions returning energies and derivatives typically return `adiff.D1` or
  `adiff.D2` objects which are processed by `makeϕrKt` to obtain scalars,
  residual vectors, and sparse stiffness matrices.
"""
# ------------------------------------------------------------------
# Element types
# ------------------------------------------------------------------

"""
C1D{P,M,T,I}

One-dimensional continuous element type storing node indices, shape-function
derivatives at P integration points, quadrature weights, element volume `V`,
and material `mat`.
"""
struct C1D{P,M,T<:Number,I<:Integer}
  nodes::Vector{I}
  Nx::NTuple{P,Vector{T}}
  wgt::NTuple{P,T}
  V::T
  mat::M
end

"""
C2D{P,M,T,I}

Two-dimensional continuous element type. Stores node indices, per-quadrature
shape-function derivatives `Nx`, `Ny`, weights `wgt`, element volume `V`,
and material `mat`.
"""
struct C2D{P,M,T<:Number,I<:Integer}
  nodes::Vector{I}
  Nx::NTuple{P,Vector{T}}
  Ny::NTuple{P,Vector{T}}
  wgt::NTuple{P,T}
  V::T
  mat::M
end

"""
C3D{P,M,T,I}

Three-dimensional continuous element type with per-point shape-function
derivatives `Nx`, `Ny`, `Nz`, quadrature weights `wgt`, volume `V`, and `mat`.
"""
struct C3D{P,M,T<:Number,I<:Integer}
  nodes::Vector{I}
  Nx::NTuple{P,Vector{T}}
  Ny::NTuple{P,Vector{T}}
  Nz::NTuple{P,Vector{T}}
  wgt::NTuple{P,T}
  V::T
  mat::M
end

"""
CAS{P,M,T,I}

Axisymmetric continuous element with shape-function values `N0` and
derivatives `Nx`, `Ny`. `X0` stores radius-related reference values used
in axisymmetric kinematics. `wgt` are quadrature weights, `V` element volume,
and `mat` material.
"""
struct CAS{P,M,T<:Number,I<:Integer}
  nodes::Vector{I}
  N0::NTuple{P,Vector{T}}
  Nx::NTuple{P,Vector{T}}
  Ny::NTuple{P,Vector{T}}
  X0::NTuple{P,T}
  wgt::NTuple{P,T}
  V::T
  mat::M
end

"""
Rod{M,T,I}

One-dimensional structural rod element with end-node coordinates `r0`,
reference length `l0`, cross-sectional area `A` and material `mat`.
"""
struct Rod{M,T<:Number,I<:Integer}
  nodes::Vector{I}
  r0::Vector{T}
  l0::T
  A::T
  mat::M
end

"""
Beam{M,T,I}

Beam element storing geometry `r0`, length `L`, thickness `t`, width `w`,
and integration rules (`lgwx`, `lgwy`) for bending; `mat` is the constitutive model.
"""
struct Beam{M,T<:Number,I<:Integer}
  nodes::Vector{I}
  r0::Vector{T}
  L::T
  t::T
  w::T
  lgwx::Array{Tuple{T,T},1}
  lgwy::Array{Tuple{T,T},1}
  mat::M
end

"""
C1DP, C2DP, C3DP

Phase-field variants of continuous elements. They store shape-function values
`N0` (for approximating the scalar phase/field variable) in addition to the
gradient derivatives and quadrature data. Useful when an additional scalar
field (phase, concentration) is coupled to the mechanics.
"""
struct C1DP{P,M,T<:Number,I<:Integer}
  nodes::Vector{I}
  N0::NTuple{P,Vector{T}}
  Nx::NTuple{P,Vector{T}}
  wgt::NTuple{P,T}
  V::T
  mat::M
end
struct C2DP{P,M,T<:Number,I<:Integer}
  nodes::Vector{I}
  N0::NTuple{P,Vector{T}}
  Nx::NTuple{P,Vector{T}}
  Ny::NTuple{P,Vector{T}}
  wgt::NTuple{P,T}
  V::T
  mat::M
end
struct C3DP{P,M,T<:Number,I<:Integer}
  nodes::Vector{I}
  N0::NTuple{P,Vector{T}}
  Nx::NTuple{P,Vector{T}}
  Ny::NTuple{P,Vector{T}}
  Nz::NTuple{P,Vector{T}}
  wgt::NTuple{P,T}
  V::T
  mat::M
end

C1DElems{P,M,T,I} = Union{C1D{P,M,T,I}, C1DP{P,M,T,I}}
C2DElems{P,M,T,I} = Union{C2D{P,M,T,I}, C2DP{P,M,T,I}}
C3DElems{P,M,T,I} = Union{C3D{P,M,T,I}, C3DP{P,M,T,I}}
CElems{P,M,T,I}   = Union{C2D{P,M,T,I}, C3D{P,M,T,I}, CAS{P,M,T,I},
                          C2DP{P,M,T,I}, C3DP{P,M,T,I}}
CPElems{P,M,T,I}  = Union{C2DP{P,M,T,I}, C3DP{P,M,T,I}, CAS{P,M,T,I}}
Elems             = Union{Rod, Beam, CElems}

export C3DP, C3D, C2DP, C2D, CElems, Rod
export C2DElems, C3DElems, CAS, CPElems, Elems

include("elasticelements.jl")
include("phasefieldelements.jl")

# ------------------------------------------------------------------
# Parameter accessors
# ------------------------------------------------------------------

"""
getP(elem)

Return the number of quadrature points `P` associated with a continuous element.
"""
getP(::CElems{P,M,T,I}) where {P,M,T,I} = P

"""
getM(elem)

Return the material type parameter stored in the element.
"""
getM(::CElems{P,M,T,I}) where {P,M,T,I} = M

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

# ------------------------------------------------------------------
# Assembly helpers: form residual and (sparse) stiffness from AD objects
# ------------------------------------------------------------------

"""
getϕ(elems, u)

Evaluate element-level free-energy `ϕ` objects for a collection of continuous
elements given the full nodal DOF matrix `u`. Returns a vector of `adiff.D2`
or `adiff.D1` objects (depending on the element/energy).
"""
getϕ(elems::Array{<:CElems}, u::Array) = [getϕ(elem, u[:,elem.nodes]) for elem in elems]

"""
makeϕrKt(Φ, elems, u) -> (ϕ, r, Kt)

Assemble contributions from a vector `Φ` of `adiff.D2` element energies and a
matching vector of `elems` into:
- scalar energy `ϕ` (sum of element energies),
- residual vector `r` (assembled internal forces),
- sparse tangent `Kt` (global stiffness) as a sparse matrix.

Notes
- `u` is the global DOF matrix used for indexing; linear indices into `u` are
  generated via `LinearIndices(u)` and used for assembly.
- This routine expects `Φ[ii]` to correspond to `elems[ii]`.
"""
function makeϕrKt(Φ::Vector{<:adiff.D2}, elems::Vector{<:Elems}, u)

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

  ϕ, r, dropzeros(sparse(II,JJ,Kt,N,N))
end

"""
makeϕr(Φ, elems, u) -> (ϕ, r)

Assemble scalar energy `ϕ` and residual vector `r` from a vector `Φ` of AD
objects (without forming a stiffness). Useful for quasi-static checks or when
only residual is required.
"""
function makeϕr(Φ::Vector{<:adiff.Duals}, elems::Vector{<:Elems}, u)

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
# Deformation gradient evaluation at integration points
# ------------------------------------------------------------------

"""
getF(elems, u)

Return a vector of element-wise averaged deformation gradients for a list of
continuous elements evaluated from nodal DOFs `u`.
"""
getF(elems::Array{<:CElems}, u::Array) = [getF(elem, u[:,elem.nodes]) for elem in elems]

"""
getF(elem::C3DElems{P}, u)

Compute the volume-averaged deformation gradient F for a 3D element from the
local nodal DOFs `u`. The returned array has shape (D,3,3) where D is the
number of DOFs per node (commonly 3 for displacements).
"""
function getF(elem::C3DElems{P}, u::Array{D}) where {P,D}
  F = zeros(D,3,3)
  for ii=1:P
    @inline F .+= elem.wgt[ii]*getF(elem, u, ii)
  end
  F/elem.V
end

"""
getF(elem::C3DElems, u, ii)

Compute the deformation gradient F at integration point `ii` for a 3D element.
Shape-function derivatives Nx,Ny,Nz are contracted with nodal displacement
components to form the standard small/finite-strain kinematic measure; the
identity `I` is added (i.e. F = I + ∇u).
"""
function getF(elem::C3DElems, u::Matrix, ii::Integer)
  Nx, Ny, Nz = elem.Nx[ii], elem.Ny[ii], elem.Nz[ii]
  u0, v0, w0 = u[1:3:end],  u[2:3:end],  u[3:3:end]

  [Nx⋅u0 Ny⋅u0 Nz⋅u0;
   Nx⋅v0 Ny⋅v0 Nz⋅v0;
   Nx⋅w0 Ny⋅w0 Nz⋅w0 ] + I
end

"""
getF(elem::C2DElems{P}, u)

Compute the volume-averaged deformation gradient F for a 2D element.
Returns a D×2×2 tensor where D is DOFs per node (typically 2).
"""
function getF(elem::C2DElems{P}, u::Array{D}) where {P,D}
  F = zeros(D,2,2)
  for ii=1:P
    @inline F .+= elem.wgt[ii]*getF(elem, u, ii)
  end
  F/elem.V
end

"""
getF(elem::C2DElems, u, ii)

Deformation gradient at integration point `ii` for a 2D element. The routine
constructs F = I + ∇u from shape-function derivatives `Nx`, `Ny`.
"""
function getF(elem::C2DElems{P,M,T,I} where {M,T,I}, u::Array{D}, ii::Integer) where {P,D}
  u0, v0 = u[1:2:end],  u[2:2:end]
  Nx, Ny = elem.Nx[ii], elem.Ny[ii]
  [Nx⋅u0 Ny⋅u0;
   Nx⋅v0 Ny⋅v0] + I
end

"""
getF(elem::CAS, u, ii)

Compute the deformation gradient at integration point `ii` for an axisymmetric
element. The axial/radial coupling and the out-of-plane stretch are included
according to axisymmetric kinematics; `N0` and `X0` are used to compute the
circumferential contribution.
"""
function getF(elem::CAS,   u::Array{D}, ii::Int64)  where D
  Nx,  Ny   = elem.Nx[ii], elem.Ny[ii]
  N0,  X0   = elem.N0[ii], elem.X0[ii]
  u0,  v0   = u[1:2:end],  u[2:2:end]
  u0x, u0y  = Nx⋅u0, Ny⋅u0
  v0x, v0y  = Nx⋅v0, Ny⋅v0
  w0z       = N0⋅u0/X0
  my0       = zero(D)

  [u0x  u0y   my0;
   v0x  v0y   my0;
   my0  my0   w0z] + I
end

# ------------------------------------------------------------------
# Jacobian, determinant, and volume utilities
# ------------------------------------------------------------------

"""
detJ(F)

Compute the determinant of the deformation gradient `F`. Supports 2×2 and
3×3 `F` stored in vector form (length 4 or 9) or as matrices; returns scalar J.
"""
function detJ(F)
  if (length(F)==9)
    F[1]*(F[5]F[9]-F[6]F[8])-
    F[2]*(F[4]F[9]-F[6]F[7])+
    F[3]*(F[4]F[8]-F[5]F[7])
  elseif length(F)==4
    F[1]F[4]-F[2]F[3]
  else
    throw(ArgumentError("detJ: unsupported matrix size $(size(F))"))
  end
end

detJ(elem,u,ii)  = detJ(getF(elem, u, ii))

"""
detJ(elem::C3DElems{P}, u0)

Compute the averaged Jacobian determinant J = det(F) for a 3D element.
"""
function detJ(elem::CElems{P}, u0::Matrix{U})  where {P, U}

  wgt = elem.wgt
  J   = zero(U)
  @inbounds for ii=1:P
    F  = getF(elem, u0, ii)
    J += detJ(F)*wgt[ii]
  end
  return J/elem.V
end

"""
detJ(elems, u)

Return element-wise determinants for a collection of elements evaluated at
their averaged deformation gradients.
"""
detJ(elems::Vector, u::Array) = [detJ(elem, u[:,elem.nodes]) for elem in elems]

"""
getI3(elem, u, ii)

Return the squared determinant of F at integration point `ii` (I3 = det(F)^2),
useful for invariants in some material models.
"""
getI3(elem,u,ii) = detJ(getF(elem, u, ii))^2
getI3(elem,u)    = sum(elem.wgt[ii]getI3(elem,u,ii) for ii in 1:length(elem.wgt))/elem.V

"""
getV(elem, u)

Compute the element volume/area as the quadrature-weighted sum of det(F)
over integration points.
"""
getV(elem,u) = sum([elem.wgt[ii]detJ(elem,u,ii) for ii in 1:length(elem.wgt)])
getV(elems::Vector, u::Array) = sum([getV(elem, u[:,elem.nodes]) for elem in elems])

# ------------------------------------------------------------------
# Stress evaluation
# ------------------------------------------------------------------

"""
getσ(elem, u)

Compute the volume-averaged Cauchy stress tensor for an element by averaging
the pointwise Cauchy stress over quadrature points. The function dispatches
to the appropriate typed implementation depending on element dimension.
"""
function getσ(elem::C3DElems{P}, u::Array{M}) where {P,M}
  σ = zeros(M,3,3)
  @inline for ii=1:P
    σ .+= elem.wgt[ii]*getσ(elem, u, ii)
  end
  σ/elem.V
end
function getσ(elem::C2DElems{P}, u::Array{M}) where {P,M}
  σ = zeros(M,2,2)
  @inline for ii=1:P
    σ .+= elem.wgt[ii]*getσ(elem, u, ii)
  end
  σ/elem.V
end

"""
getσ(elem::CElems, u, ii)

Compute the Cauchy stress at integration point `ii` for generic continuous
elements. Uses material routine `getϕ` to obtain the first Piola or energy,
then forms P = ∂ϕ/∂F and σ = J^{-1} P F^T.
"""
function getσ(elem::CElems, u::Array, ii::Int)

  F  = getF(elem, u, ii)
  δϕ = getϕ(adiff.D1(F), elem.mat)
  P  = reshape(adiff.grad(δϕ), size(F))
  J  = getJ(F,elem.mat)

  return 1/J*P*F'
end
getσ(elems::Array{<:CElems}, u::Array) = [getσ(elem, u[:,elem.nodes]) for elem in elems]

# ------------------------------------------------------------------
# Inertia / kinetic energy contributions
# ------------------------------------------------------------------

"""
getT(elem, udot0)

Compute the kinetic energy contribution (scalar) for an element given nodal
velocity field `udot0`. Implemented for C3D, C2D, and Rod element variants.
"""
function getT(elem::C3DElems{P}, udot0::Matrix{T}) where {T,P}
  ϕ   = zero(T)
  for ii=1:P
    N0 = elem.N0[ii]
    d  = [N0⋅udot0[1:3:end], N0⋅udot0[2:3:end], N0⋅udot0[3:3:end]]
    ϕ += elem.mat.mat.ρ*elem.wgt[ii]* (d⋅d)
  end
  ϕ
end
function getT(elem::Rod{<:PhaseField},
              udot0::Matrix{T}) where T
  udot = (udot0[:,1]+udot0[:,2])/2
  Vol  = elem.A*elem.l0
  elem.mat.mat.ρ*Vol/2*(udot⋅udot)
end
function getT(elem::C2DElems{P,M} where M,
              udot0::Matrix{T}) where {T,P}
  ϕ   = zero(T)
  for ii=1:P
    N0 = elem.N0[ii]
    d  = [N0⋅udot0[1:2:end], N0⋅udot0[2:2:end]]
    ϕ += elem.mat.mat.ρ*elem.wgt[ii]* (d⋅d)
  end
  ϕ
end

"""
getT(elems, udot)

Assemble kinetic-energy contributions into global residual and tangent using
`makeϕrKt`. Parallelized with `Threads.@threads` over elements.
"""
function getT(elems::Vector{<:Elems},
              udot::Matrix{T}) where T
  nElems = length(elems)
  Φ = Vector{adiff.D2}(undef, nElems)
  Threads.@threads for ii=1:nElems
    Φ[ii] = getT(elems[ii], adiff.D2(udot[:,elems[ii].nodes]))
  end
  makeϕrKt(Φ, elems, udot)
end

# ------------------------------------------------------------------
# Utility: element info retrieval
# ------------------------------------------------------------------

"""
getinfo(elem, u; info=:detF)

Return requested per-element scalar information (e.g., :detF or other measures)
by computing the averaged deformation gradient and dispatching to
`Materials.getinfo`. The `info` keyword selects what quantity to return.
"""
function getinfo(elem::CElems{P}, u::Matrix{<:Number}; info=:detF) where P
  F = sum([getF(elem, u, ii) for ii=1:P])/P
  Materials.getinfo(F, elem.mat, info=info)
end
getinfo(elems::Array, u; info=:detF) =  [getinfo(elem, u[:,elem.nodes], info=info) for elem in elems]

# ------------------------------------------------------------------
# Quadrature helper
# ------------------------------------------------------------------

"""
lgwt(N; a=0, b=1)

Compute Gauss-Legendre nodes and weights for N points on interval [a,b].

Returns a vector of tuples `(x,w)` for each quadrature point.
"""
function lgwt(N::Integer; a=0, b=1)

  N, N1, N2 = N-1, N, N+1
  xu   = range(-1, stop=1,length=N1)
  y    = cos.((2collect(0:N) .+ 1)*pi/(2N+2)) .+ (0.27/N1)*sin.(π*xu*N/N2)
  L    = zeros(N1,N2)
  dTol = 1e-16
  y0   = 2

  while maximum(abs.(y.-y0)) > dTol
    L[:,1] .= 1
    L[:,2] .= y
    for k = 2:N1
      L[:,k+1]=((2k-1)*y.*L[:,k] .- (k-1)*L[:,k-1])/k
    end
    global Lp = N2*(L[:,N1] .- y .* L[:,N2])./(1 .- y.^2)
    y0 = y
    y  = y0 .- L[:,N2]./Lp
  end

  x = (a.*(1 .- y) .+ b.* (1 .+ y))./2
  w = (b-a)./((1 .- y.^2).*Lp.^2).*(N2/N1)^2

  return [(x[ii], w[ii]) for ii ∈ 1:N1]
end

end
