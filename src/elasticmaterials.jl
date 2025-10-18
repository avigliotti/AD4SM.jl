"""
Hooke(E,ν;ρ=1;small=false)

Linear isotropic Hookean elasticity (3D) type.

Fields
- `E` : Young's modulus.
- `ν` : Poisson's ratio.
- `ρ` : material density (default `1`).
- `small` : `true` if small-strain kinematics are intended; otherwise large-strain
  (Green–Lagrange) kinematics are used.

Notes
The Lamé parameters used internally are
\begin{align*}
\lambda &= \dfrac{E\nu}{(1+\nu)(1-2\nu)},\quad
\mu=\dfrac{E}{2(1+\nu)}.
\end{align*}
The struct implements a simple container for linear elasticity parameters and
controls whether small- or large-strain formulas are evaluated by `getϕ`.
"""
struct Hooke{T} <: Mat3D
  E     ::T
  ν     ::T
  ρ     ::T
  small ::Bool
  Hooke(E::T,ν::T,ρ=one(T);small=false) where T<:Number = new{T}(E,ν,ρ,small)
end
"""
Hooke1D(E;ρ=1;small=false)

One-dimensional linear elastic material.

Arguments
- `E` : Young's modulus (scalar).
- `ρ` : density (default `1`).
- `small` : whether to use small-strain kinematics.

Returns the 1D Hooke material object. Used by `getϕ` overloads for 1D problems.
"""
struct Hooke1D{T} <: Mat1D
  E     ::T
  ρ     ::T
  small ::Bool
  Hooke1D(E::T,ρ=one(T);small=false) where T<:Number = new{T}(E,ρ,small)
end
"""
Hooke2D(E,ν;ρ=1;small=true,plane_stress=true)

Two-dimensional (plane) linear elastic material. The second type parameter is
used internally to distinguish `$:\\text{plane\_stress}$` and
`$:\\text{plane\_strain}$` variants.

Fields
- `E` : Young's modulus.
- `ν` : Poisson's ratio.
- `ρ` : density.
- `small` : small-strain flag.
- the type parameter `:plane_stress` or `:plane_strain` selects the kinematic
  assumption used when evaluating `getϕ`.
"""
struct Hooke2D{T,P} <: Mat2D
  E     ::T
  ν     ::T
  ρ     ::T
  small ::Bool
  Hooke2D(E::T,ν::T,ρ=one(T); small=true, plane_stress=true) where T<:Number = plane_stress ? 
  new{T,:plane_stress}(E,ν,ρ,small) : 
  new{T,:plane_strain}(E,ν,ρ,small)
end
"""
MooneyRivlin(C1,C2;K=-1;small=false)

Mooney–Rivlin hyperelastic constitutive type for near-incompressible rubber-like
materials.

Parameters
- `C1,C2` : material constants appearing in the strain energy density.
- `K` : bulk/modifier parameter; if negative a special treatment for 2D -> 3D
  reconstruction is used in the code (see `getϕ` overloads).
- `small` : flag for small-strain approximations (unused for strongly nonlinear
  models but kept for API consistency).
"""
struct MooneyRivlin{T} <: Mat3D
  C1    ::T
  C2    ::T
  K     ::T
  small ::Bool
  MooneyRivlin(C1::T, C2::T)       where T<:Number = new{T}(C1, C2, T(-1), false)
  MooneyRivlin(C1::T, C2::T, K::T) where T<:Number = new{T}(C1, C2, K, false) 
end
"""
NeoHooke(μ;K=-1;small=false)

Neo-Hookean hyperelastic material with shear modulus `μ` and optional bulk
parameter `K` (used for plane->3D reconstruction in 2D calls)."""
struct NeoHooke{T} <: Mat3D
  C1    ::T 
  K     ::T
  ρ     ::T
  small ::Bool
  NeoHooke(C1::T,K::T,ρ=one(T)) where T<:Number = new{T}(C1, K, ρ, false)
end
"""
Ogden(α,μ;K=-1)

Ogden-type hyperelastic model parameterized by `α` (exponent) and `μ` (modulus).
`K` handles volumetric/plane->3D coupling when negative indicates special
2D reconstruction.
"""
struct Ogden{T} <: Mat3D
  α   ::T
  μ   ::T
  K   ::T
  Ogden(α::T, μ::T)       where T<:Number = new{T}(α, μ, T(-1), false) 
  Ogden(α::T, μ::T, K::T) where T<:Number = new{T}(α, μ, K, false) 
end

HyperEla = Union{MooneyRivlin,NeoHooke,Ogden} 
dTol     = 1e-7
maxiter  = 30
"""
setmaxiter(x)

Set the global maximum number of Newton iterations used by nonlinear material
routines. `x` is converted to `Int64` and stored in the package-global
`maxiter` variable.
"""
function setmaxiter(x)
  global maxiter = Int64(x)
end
"""
setdTol(x)

Set the global tolerance `dTol` used for convergence checks in iterative
routines. `x` should be a numeric scalar.
"""
function setdTol(x)
  global dTol = x
end
"""
getϕ(F,mat) -> ϕ

Evaluate the strain energy density `ϕ` for a given deformation gradient `F`
and constitutive `mat` (one of the hyperelastic or linear material types).
Overloads exist for 1D, 2D (plane stress/strain) and 3D material types.

Arguments
- `F` : deformation gradient (2×2 or 3×3 `Array{<:Number,2}`).
- `mat` : material object (e.g. `Hooke`, `MooneyRivlin`, `Ogden`, ...).

Returns the scalar energy density `ϕ`. For 2D plane problems some overloads
reconstruct a compatible 3D deformation gradient (using `mat.K`) before
computing invariants; this behaviour is documented per-overload in the source.
"""
function getϕ(F::Array{<:Number,2}, mat::M where M <:HyperEla)
  C = transpose(F)F
  if length(C) == 9
    (I1,I2,I3) = getInvariants(C)
  else
    L3 = mat.K<0 ? (F[1]F[4]-F[2]F[3])^-2 : 1
    (I1,I2,I3) = getInvariants(C, L3)
  end
  getϕ(I1,I2,I3,mat)
end
function getϕ(F::Array{<:Number,2}, mat::Ogden)

  α, μ, K = mat.α, mat.μ, mat.K

  if length(F) == 9
    C = transpose(F)F
  else
    F33 = K<0 ? 1/(F[1]F[4]-F[2]F[3]) : one(F[1]) 
    F3D = fill(zero(F[1]), (3,3))
    F3D[1:2,1:2] = F
    F3D[9] = F33
    C   = transpose(F3D)*F3D
  end

  λ = sqrt.(svdvals(C))
    
  if K<0
    ϕ = μ/α * (sum(λ.^α) - 3)
  else
    J = prod(λ)
    ϕ = μ/α *(sum(λ.^α)/J^(α/3) - 3) + K*(J-1)^2
  end

  return ϕ
end
function getϕ(I1, I2, I3, mat::MooneyRivlin)

  C1, C2, K = mat.C1, mat.C2, mat.K
  if K<0
    ϕ  = C1*(I1-3) + C2*(I2-3)
  else
    J  = sqrt(I3)
    I1 = I1*J^(-2/3)
    I2 = I2*J^(-4/3)
    ϕ  = C1*(I1-3) + C2*(I2-3) + K*(J-1)^2
  end
  return ϕ
end
function getϕ(I1, I2, I3, mat::NeoHooke)

  C1, K  = mat.C1, mat.K
  if K<0 
    ϕ  = C1*(I1-3)
  else
    J  = sqrt(I3)
    I1 = I1*J^(-2/3)
    ϕ  = C1*(I1-3) + K*(J-1)^2
  end
  return ϕ
end
function getϕ(F11::Number, mat::Hooke1D)
  ϕ = (mat.E*(F11-1)^2)/2
end
function getϕ(F::Array{<:Number,2}, mat::Hooke)

  if mat.small
    E = (F+transpose(F)-2I)/2   # the symmetric part of G
  else
    E = (transpose(F)F-I)/2   # the Green-Lagrange strain tensor
  end

  ν, Es = mat.ν, mat.E
  λ = Es*ν/(1+ν)/(1-2ν) 
  μ = Es/2/(1+ν) 

  ϕ =  (μ+λ/2) * (E[1]^2   + E[5]^2   + E[9]^2)
  ϕ += λ       * (E[1]E[5] + E[5]E[9] + E[9]E[1])
  ϕ += 2μ      * (E[2]^2   + E[3]^2   + E[6]^2)

  return ϕ
end
function getϕ(F::Array{<:Number,2}, mat::Hooke2D{T,:plane_strain} where T)

  if mat.small
    E = (F+transpose(F)-2I)/2   # the symmetric part of G
  else
    E = (transpose(F)F-I)/2   # the Green-Lagrange strain tensor
  end

  ν, Es = mat.ν, mat.E
  λ = Es*ν/(1+ν)/(1-2ν) 
  μ = Es/2/(1+ν) 

  (μ+λ/2)*(E[1]^2+E[4]^2) + λ*E[1]E[4] + 2μ*E[2]^2
end
function getϕ(F::Array{<:Number,2}, mat::Hooke2D{T,:plane_stress} where T)

  if mat.small
    E = (F+transpose(F)-2I)/2   # the symmetric part of G
  else
    E = (transpose(F)F-I)/2   # the Green-Lagrange strain tensor
  end

  ν, Es = mat.ν, mat.E

  (Es/(1-ν^2))*(E[1]^2+E[4]^2+ν*E[1]E[4]) + (Es/(1+ν))*E[2]^2
end
"""
gethyddevdecomp(F,mat) -> (I1,I1sq)

Compute hydrostatic/deviatoric contributions required for 2D plane elasticity.

For plane-strain and plane-stress Hooke types this function reconstructs the
out-of-plane strain component consistent with the chosen kinematic assumption
and returns the first invariant `$I_1$` and the squared sum `$I_1^2$` (used in
energy expressions).

Arguments
- `F` : 2×2 deformation gradient.
- `mat` : `Hooke2D` material specifying plane assumption and Poisson ratio.

Returns
- `I1` : scalar first invariant of the small/Green–Lagrange strain tensor.
- `I1sq` : sum of squared components entering volumetric energy terms.
"""
function gethyddevdecomp(F::Array{<:Number,2}, mat::Hooke2D{T,:plane_strain} where T)

  if mat.small
    E = (F+transpose(F)-2I)/2   # the symmetric part of G
  else
    E = (transpose(F)F-I)/2   # the Green-Lagrange strain 
  end

  I1 = E[1]+E[4]
  ϵd = E - I*I1/3

  return I1, ϵd⋅ϵd + (I1/3)^2
end
function gethyddevdecomp(F::Array{<:Number,2}, mat::Hooke2D{T,:plane_stress} where T)

  if mat.small
    E = (F+transpose(F)-2I)/2   # the symmetric part of G
  else
    E = (transpose(F)F-I)/2   # the Green-Lagrange strain 
  end

  E33  = mat.ν/(mat.ν-1) * (E[1]+E[4])

  I1   = E[1]+E[4]+E33
  ϵd   = E - I*I1/3

  return I1, ϵd⋅ϵd + (E33-I1/3)^2
end
function gethyddevdecomp(F::Array{<:Number,2}, mat::Hooke1D)
  F[1], F[1]^2
end
"""
get1stinvariants(F,mat) -> (I1,I1sq)

Return the first invariant(s) appropriate for lower-dimensional materials.
- For `Hooke1D` the invariant is simply `F[1]` and its square.
- For planar Hooke variants the function reconstructs the needed out-of-plane
  strain and returns `(I1,I1^2)`.

This small helper centralizes the computation of the linear combinations used
by the energy routines.
"""
function get1stinvariants(F::Array{<:Number,2}, mat::Hooke2D{T,:plane_strain} where T)

  if mat.small
    E = (F+transpose(F)-2I)/2   # the symmetric part of G
  else
    E = (transpose(F)F-I)/2   # the Green-Lagrange strain 
  end

  I1   = E[1]+E[4]
  I1sq = E[1]^2+E[4]^2+2*E[3]^2

  return I1, I1sq
end
function get1stinvariants(F::Array{<:Number,2}, mat::Hooke2D{T,:plane_stress} where T)

  if mat.small
    E = (F+transpose(F)-2I)/2   # the symmetric part of G
  else
    E = (transpose(F)F-I)/2   # the Green-Lagrange strain 
  end

  E33  = mat.ν/(mat.ν-1) * (E[1]+E[4])

  I1   = E[1]+E[4]+E33
  I1sq = E[1]^2+E[4]^2+E33^2+2*E[3]^2

  return I1, I1sq
end
function get1stinvariants(F::Array{<:Number,2}, mat::Hooke1D)
  F[1], F[1]^2
end

