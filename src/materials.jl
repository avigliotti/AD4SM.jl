__precompile__()

module Materials

export Material, Mat3D, Mat2D, Mat1D
export getϕ, getJ

using LinearAlgebra, StaticArrays 
using ..adiff

# union of all materials
abstract type Material end
abstract type Mat3D <: Material end
abstract type Mat2D <: Material end
abstract type Mat1D <: Material end

# support for elastic materials
include("elasticmaterials.jl")

# support for phase field materials
include("phasefieldmaterials.jl")

# status retrieving functions
function getP(F::Array{Float64,2}, mat::Material) # 1st PK tensor from F
  ϕ     = getϕ(adiff.D1(F), mat)               
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
function getϕ(F11::Number, mat::M where M <: HyperEla; binc=true)

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
function getL3(func, L3; maxiter=30, dTol=1e-7)
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
# functions for evaluating the determinat of F 
getJ(F, mat::Materials.Mat2D) = F[1]F[4]-F[2]F[3]
getJ(F, mat::Materials.Mat3D) = F[1]F[5]F[9]-F[1]F[6]F[8]-
                                F[2]F[4]F[9]+F[2]F[6]F[7]+
                                F[3]F[4]F[8]-F[3]F[5]F[7]
# these functions assume C is symmetrical
getI1(C)         = C[1]+C[5]+C[9]
getI2(C)         = C[1]C[5]+C[5]C[9]+C[1]C[9]-C[2]^2-C[3]^2-C[6]^2
getI3(C)         = C[1]C[5]C[9]+2C[2]C[3]C[6]-C[1]C[6]^2-C[5]C[3]^2-C[9]C[2]^2
getInvariants(C) = (getI1(C),getI2(C),getI3(C))
# 
# these functions compute the hydrostatic / deviatoric decomposition
function gethyddevdecomp(F::Array{<:Number,2}, mat::Material)

  if mat.small
    E = (F+transpose(F)-2I)/2 # the symmetric part of G
  else
    E = (transpose(F)F-I)/2   # the Green-Lagrange strain 
  end

  I1 = E[1]+E[5]+E[9]
  ϵd = E - I*I1/3

  return I1, ϵd⋅ϵd
end
# get1stinvariants 
# I1    is the first invariant of F
# I1sq  is the first invariant of C=transpose(F)F
#=
function get1stinvariants(F::Array{<:Number,2}, mat::Material)

E    = (transpose(F)F-I)/2   # the Green-Lagrange strain 
I1   = E[1]+E[5]+E[9]
I1sq = E[1]^2+E[5]^2+E[9]^2+2*(E[2]^2+E[3]^2+E[6]^2)

return I1, I1sq
end
=#
function get1stinvariants(F::AbstractMatrix{<:Number}, mat::Material)

  if mat.small
    E = (F+transpose(F)-2I)/2 # the symmetric part of G
  else
    E = (transpose(F)F-I)/2   # the Green-Lagrange strain 
  end

  I1   = E[1]+E[5]+E[9]
  I1sq = E[1]^2+E[5]^2+E[9]^2+2*(E[2]^2+E[3]^2+E[6]^2)

  return I1, I1sq
end

end
