
# Material types
struct Hooke{T} <: Mat3D
  E     ::T
  ν     ::T
  ρ     ::T
  small ::Bool
  Hooke(E::T,ν::T,ρ=one(T);small=false) where T<:Number = new{T}(E,ν,ρ,small)
end
struct Hooke1D{T} <: Mat1D
  E     ::T
  ρ     ::T
  small ::Bool
  Hooke1D(E::T,ρ=one(T);small=false) where T<:Number = new{T}(E,ρ,small)
end
struct Hooke2D{T,P} <: Mat2D
  E     ::T
  ν     ::T
  ρ     ::T
  small ::Bool
  Hooke2D(E::T,ν::T,ρ=one(T); small=true, plane_stress=true) where T<:Number = plane_stress ? 
  new{T,:plane_stress}(E,ν,ρ,small) : 
  new{T,:plane_strain}(E,ν,ρ,small)
end
struct MooneyRivlin{T} <: Mat3D
  C1  ::T
  C2  ::T
  K   ::T
  MooneyRivlin(C1::T, C2::T)       where T<:Number = new{T}(C1, C2, T(-1))
  MooneyRivlin(C1::T, C2::T, K::T) where T<:Number = new{T}(C1, C2, K) 
end
struct NeoHooke{T} <: Mat3D
  C1   ::T 
  K    ::T
  ρ    ::T
  # NeoHooke(C1::T)               where T<:Number = new{T}(C1, T(-1), T(1))
  # NeoHooke(C1::T,K::T)          where T<:Number = new{T}(C1, K, T(1))
  NeoHooke(C1::T,K::T,ρ=one(T)) where T<:Number = new{T}(C1, K, ρ)
end
struct Ogden{T} <: Mat3D
  α   ::T
  μ   ::T
  K   ::T
  Ogden(α::T, μ::T)       where T<:Number = new{T}(α, μ, T(-1)) 
  Ogden(α::T, μ::T, K::T) where T<:Number = new{T}(α, μ, K) 
end
#

HyperEla = Union{MooneyRivlin,NeoHooke,Ogden} 

#
# default convergence tolerance for 2D stress
dTol     = 1e-7
maxiter  = 30
function setmaxiter(x)
  global maxiter = Int64(x)
end
function setdTol(x)
  global dTol = x
end
# elastic energy evaluation functions for materials
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
    # λ/J^(1/3) are the principal stretches of the deviatoric part
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
    # ϕ  = C1*(I1-3) + C2*(I2-3) + K*log(J)^2
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
    # ϕ  = C1*(I1-3) + K*log(J)^2
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
  # μ = Es/2/(1+ν) 

  (Es/(1-ν^2))*(E[1]^2+E[4]^2+ν*E[1]E[4]) + (Es/(1+ν))*E[2]^2
end
# 
# these functions compute the hydrostatic / deviatoric decomposition
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
# get1stinvariants for Hooke materials
# I1    is the first invariant of F
# I1sq  is the first invariant of C=transpose(F)F
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

