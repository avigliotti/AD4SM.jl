__precompile__()

module Materials

using LinearAlgebra 
using ..adiff

# Material types
struct Hooke{T}
  E     ::T
  ν     ::T
  ρ     ::T
  small ::Bool
  # Hooke(E, ν; small=false) = Hooke(promote(E,ν)..., small)
  Hooke(E::T,ν::T,ρ=one(T);small=false) where T<:Number = new{T}(E,ν,ρ,small)
end
struct Hooke1D{T}
  E     ::T
  ρ     ::T
  small ::Bool
  Hooke1D(E::T,ρ=one(T);small=false) where T<:Number = new{T}(E,ρ,small)
end
struct Hooke2D{T,P}
  E     ::T
  ν     ::T
  ρ     ::T
  small ::Bool
  Hooke2D(E::T,ν::T,ρ=one(T); small=true, plane_stress=true) where T<:Number = plane_stress ? 
  new{T,:plane_stress}(E,ν,ρ,small) : 
  new{T,:plane_strain}(E,ν,ρ,small)
end
struct MooneyRivlin{T}
  C1  ::T
  C2  ::T
  K   ::T
  MooneyRivlin(C1::T, C2::T)       where T<:Number = new{T}(C1, C2, T(-1))
  MooneyRivlin(C1::T, C2::T, K::T) where T<:Number = new{T}(C1, C2, K) 
end
struct NeoHooke{T}
  C1   ::T 
  K    ::T
  NeoHooke(C1::T)       where T<:Number = new{T}(C1, T(-1))
  NeoHooke(C1::T, K::T) where T<:Number = new{T}(C1, K)
end
struct Ogden{T}
  α   ::T
  μ   ::T
  K   ::T
  Ogden(α::T, μ::T)       where T<:Number = new{T}(α, μ, T(-1)) 
  Ogden(α::T, μ::T, K::T) where T<:Number = new{T}(α, μ, K) 
end
Material = Union{Hooke,Hooke1D,Hooke2D,MooneyRivlin,NeoHooke,Ogden} 
HyperEla = Union{MooneyRivlin,NeoHooke,Ogden} 
Mat3D    = Union{Hooke,MooneyRivlin,NeoHooke,Ogden}
Mat2D    = Hooke2D
Mat1D    = Hooke1D
dims(mat::M) where M<:Mat3D = 3
dims(mat::M) where M<:Mat2D = 2
dims(mat::M) where M<:Mat1D = 1
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
function getϕ(F::Array{N,2}, mat::M) where {N<:Number, M<:HyperEla}
  C = transpose(F)F
  if length(C) == 9
    (I1,I2,I3) = getInvariants(C)
  else
    L3 = mat.K<0 ? (F[1]F[4]-F[2]F[3])^-2 : 1
    (I1,I2,I3) = getInvariants(C, L3)
  end
  getϕ(I1,I2,I3,mat)
end
function getϕ(F::Array{N,2}, mat::Ogden) where {N<:Number}

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
function getϕ(F11::N where N<:Number, mat::Hooke1D)
  ϕ = (mat.E*(F11-1)^2)/2
end
# function getϕ(C11::N where N<:Number, mat::Hooke1D)
#   ϕ = mat.small ? mat.E*(C11-1) : 0.5mat.E*(C11-1)^2  
# end
function getϕ(F::Array{N,2} where N<:Number, mat::Hooke)

  if mat.small
    E = (F+transpose(F)-2I)/2   # the symmetric part of G
  else
    E = (transpose(F)F-I)/2   # the Green-Lagrange strain tensor
  end

  ν, Es = mat.ν, mat.E
  λ = Es*ν/(1+ν)/(1-2ν) 
  μ = Es/2/(1+ν) 

<<<<<<< HEAD
  if size(F) == (3,3)
    ϕ =  (μ+λ/2) * (E[1]^2   + E[5]^2   + E[9]^2)
    ϕ += λ       * (E[1]E[5] + E[5]E[9] + E[9]E[1])
    ϕ += 2μ      * (E[2]^2   + E[3]^2   + E[6]^2)
  else
    # 2D is plain strain
    ϕ = (μ+λ/2)*(E[1]^2+E[4]^2) + λ*E[1]E[4] + 2μ*E[2]^2
  end

  return ϕ
=======
  ϕ =  (μ+λ/2) * (E[1]^2   + E[5]^2   + E[9]^2)
  ϕ += λ       * (E[1]E[5] + E[5]E[9] + E[9]E[1])
  ϕ += 2μ      * (E[2]^2   + E[3]^2   + E[6]^2)

  return ϕ
end
function getϕ(F::Array{N,2} where N<:Number, mat::Hooke2D{T,:plane_strain} where T)

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
function getϕ(F::Array{N,2} where N<:Number, mat::Hooke2D{T,:plane_stress} where T)

  if mat.small
    E = (F+transpose(F)-2I)/2   # the symmetric part of G
  else
    E = (transpose(F)F-I)/2   # the Green-Lagrange strain tensor
  end

  ν, Es = mat.ν, mat.E
  # μ = Es/2/(1+ν) 

  (Es/(1-ν^2))*(E[1]^2+E[4]^2+ν*E[1]E[4]) + (Es/(1+ν))*E[2]^2
>>>>>>> candidate
end
# status retrieving functions
# function getP(F::Array{Float64,2}, mat) # 1st PK tensor from F
#   dims_ = dims(mat)
#   ϕ     = getϕ(adiff.D2(F), mat)               
#   reshape(adiff.grad(ϕ), (dims_,dims_))
# end
function getP(F::Array{Float64,2}, mat::Material) # 1st PK tensor from F
  # dims_ = dims(mat)
  ϕ     = getϕ(adiff.D2(F), mat)               
  reshape(adiff.grad(ϕ), size(F))
end
function getσ(F::Array{Float64,2}, mat::Material) # Cauchy stress 
  J = det(F)
  P = getP(F,mat)         # 1st PK stress
  (P*transpose(F))/J
end
function getinfo(F::Array{Float64,2}, mat::Material; info = :ϕ)
  if info == :ϕ
    getϕ(F, mat)     
  elseif info == :detF 
    det(F)
  elseif info == :G
    F-I
  elseif info == :P
    getP(F, mat)
  elseif info == :σ
    getσ(F, mat)
  elseif info == :σVM
    σ = getσ(F, mat)
    sqrt((σ[1]-σ[5])^2+(σ[5]-σ[9])^2+(σ[9]-σ[1])^2+6(σ[4]^2+σ[7]^2+σ[8]^2))/2
  elseif info == :S
    F\getP(F, mat)
  elseif info == :I
    getInvariants(transpose(F)F)
  elseif info == :LE
    svdF = svd(F)
    svdF.V*diagm(0=>log.(svdF.S))*transpose(svdF.V)
  else
    F
  end
end
# helper functions for 2D and 1D stress 
function getϕ(F11::N, mat::M; binc=true) where {N<:Number, M<:HyperEla}

  C11 = F11^2
  L1  = Real(C11)
  L3  = binc ? getL3(L3->getϕ(L1,L3,L3,mat), L1) : 
  getL3(L3->getϕ(L1,L3,L3,mat), √L1)

  if isa(C11, adiff.D2)
    ϕ1    = getϕ(adiff.D2([L1, L3, L3])..., mat)
    dL21  = -ϕ1.h[2]/(ϕ1.h[3]+ϕ1.h[5])
    L2    = adiff.D2(L3, dL21*C11.g) 
    ϕ     = getϕ(C11,L2,L2,mat)
  else
    ϕ  = getϕ(C11,L3,L3,mat)
  end

  return ϕ
end
function getL3(func, L3)
  iter   = 0 
  bdone  = false
  while !bdone
    ϕ = func(adiff.D2(L3))
    r = adiff.grad(ϕ)[1]
    if abs(r) < dTol
      bdone = true
    else
      updt = -r/adiff.hess(ϕ)[1]
      L3   = L3+updt<0 ? L3/2 : L3+updt
    end
    iter +=1
    if iter > maxiter
      println("failed in getL3, with L3: ",L3," r: ", r, " and iter: ",iter)
      error("failed in getL3")
      bdone = true
    end
  end
  return  L3
end
function getInvariants(C, C33)
  I1 = C[1]+C[4]+C33
  I2 = C[1]C[4]+C[4]C33+C[1]C33-C[2]^2
  I3 = C[1]C[4]C33-C33*C[2]^2 
  
  (I1,I2,I3)
end
getI1(C) = C[1]+C[5]+C[9]
getI2(C) = C[1]C[5]+C[5]C[9]+C[1]C[9]-C[2]^2-C[3]^2-C[6]^2
getI3(C) = C[1]C[5]C[9]+2C[2]C[3]C[6]-C[1]C[6]^2-C[5]C[3]^2-C[9]C[2]^2
getInvariants(C) = (getI1(C),getI2(C),getI3(C))
# function getInvariants(C)
#   I1 = C[1]+C[5]+C[9]
#   I2 = C[1]C[5]+C[5]C[9]+C[1]C[9]-C[2]^2-C[3]^2-C[6]^2
#   I3 = C[1]C[5]C[9]+2C[2]C[3]C[6]-C[1]C[6]^2-C[5]C[3]^2-C[9]C[2]^2
#   
#   (I1,I2,I3)
# end
end
