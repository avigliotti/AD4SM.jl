__precompile__()

module Materials

using LinearAlgebra 
using ..adiff

# Material types
struct Hooke
  E     ::Float64
  ν     ::Float64
  small ::Bool
  Hooke(E,ν;small=false) = new(Float64(E),Float64(ν), small)
end
struct MooneyRivlin
  C1  ::Float64
  C2  ::Float64
  K   ::Float64
  MooneyRivlin(C1, C2)    = new(Float64(C1), Float64(C2), NaN) 
  MooneyRivlin(C1, C2, K) = new(Float64(C1), Float64(C2), Float64(K)) 
end
struct NeoHooke
  C1   ::Float64
  K    ::Float64
  NeoHooke(C1)    = new(Float64(C1), NaN) 
  NeoHooke(C1, K) = new(Float64(C1), Float64(K))
end
Material = Union{Hooke,MooneyRivlin,NeoHooke} 
HyperEla = Union{MooneyRivlin,NeoHooke} 
# default convergence tolerance for 2D stress
dTol     = 1e-7
maxiter  = 30
function setmaxiter(x)
  global maxiter = Int64(x)
end
function setdTol(x)
  global dTol = Float64(x)
end
# elastic energy evaluation functions for materials
function getϕ(F::Array{N,2}, mat::M) where {N<:Number, M<:HyperEla}
  C = transpose(F)F
  if length(C) == 9
    (I1,I2,I3) = getInvariants(C)
  else
    L3 = isnan(mat.K) ? (F[1]F[4]-F[2]F[3])^-2 : 1
    (I1,I2,I3) = getInvariants(C, L3)
  end
  Materials.getϕ(I1,I2,I3,mat)
end
function getϕ(I1, I2, I3, mat::MooneyRivlin)

  C1, C2, K = mat.C1, mat.C2, mat.K
  if isnan(K) 
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
  if isnan(K) 
    ϕ  = C1*(I1-3)
  else
    J  = sqrt(I3)
    I1 = I1*J^(-2/3)
    # ϕ  = C1*(I1-3) + K*log(J)^2
    ϕ  = C1*(I1-3) + K*(J-1)^2
  end
  return ϕ
end
function getϕ(C11::N where N<:Number, mat::Hooke)
  ϕ = 0.5mat.E*(C11-1)^2  
end
function getϕ(F::Array{N,2} where N<:Number, mat::Hooke)
  if mat.small
    E = 0.5(F+transpose(F))-I   # the symmetric part of G
  else
    E = 0.5(transpose(F)F-I)    # the Green-Lagrange strain tensor
  end

  if size(F) == (3,3)
    ν, Es = mat.ν, mat.E
    λ = Es*ν/(1+ν)/(1-2ν) 
    μ = Es/2/(1+ν) 

    I1 = E[1]+E[5]+E[9]
    S  = λ*I1*I + 2μ*E
    ϕ  = 0.5sum(E.*S)
  else
    ν, Es = mat.ν, mat.E
    ϕ = Es/(1-ν^2)/2*(E[1]^2+E[2]^2+2ν*E[1]E[2]+(1-ν)*E[3]^2)
  end
  return ϕ
end
# status retrieving functions
function getP(F::Array{Float64,2}, mat) # 1st PK tensor from F
  ϕ = getϕ(adiff.D2(F), mat)               
  reshape(adiff.grad(ϕ), (3,3))
end
function getσ(F::Array{Float64,2}, mat) # Cauchy stress 
  J = det(F)
  P = getP(F,mat)         # 1st PK stress
  (P*transpose(F))/J
end
function getinfo(F::Array{Float64,2}, mat; info = :ϕ)
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
    sqrt(0.5)*sqrt((σ[1]-σ[5])^2+(σ[5]-σ[9])^2+(σ[9]-σ[1])^2+6(σ[4]^2+σ[7]^2+σ[8]^2))
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
function getInvariants(C)
  I1 = C[1]+C[5]+C[9]
  I2 = C[1]C[5]+C[5]C[9]+C[1]C[9]-C[2]^2-C[3]^2-C[6]^2
  I3 = C[1]C[5]C[9]+2C[2]C[3]C[6]-C[1]C[6]^2-C[5]C[3]^2-C[9]C[2]^2
  
  (I1,I2,I3)
end
end
