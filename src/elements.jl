__precompile__()

module Elements

using LinearAlgebra, SparseArrays

using ..adiff, ..Materials
# import Base.copy
import ..Materials.getϕ

export makeϕrKt
export getF, getϕ, getδϕ

# continous elements
# these elements hold the tools to evaluate the gradient of a function at the 
# integration points, this is done trough the Nx,Ny,Nz vectors, one for each
# integration point
struct C1D{P,M,T<:Number,I<:Integer}
  nodes::Vector{I}
  Nx::NTuple{P,Vector{T}}
  wgt::NTuple{P,T}
  V::T
  mat::M
end
struct C2D{P,M,T<:Number,I<:Integer}
  nodes::Vector{I}
  Nx::NTuple{P,Vector{T}}
  Ny::NTuple{P,Vector{T}}
  wgt::NTuple{P,T}
  V::T
  mat::M
end
struct C3D{P,M,T<:Number,I<:Integer}
  nodes::Vector{I}
  Nx::NTuple{P,Vector{T}}
  Ny::NTuple{P,Vector{T}}
  Nz::NTuple{P,Vector{T}}
  wgt::NTuple{P,T}
  V::T
  mat::M
end
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
#
#  structural elements
struct Rod{M,T<:Number,I<:Integer}
  nodes::Vector{I}
  r0::Vector{T}
  l0::T
  A::T
  mat::M
end
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
#
# continous elements with phase
# these elements also hold the mean to evaluate the value of the function at 
# the integration point, not only its gradient this is done trough 
# the N0 arrays
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

# parameters retriving functions 
getP(::CElems{P,M,T,I}) where {P,M,T,I} = P
getM(::CElems{P,M,T,I}) where {P,M,T,I} = M
#
# × operator
# this operator allows to use the chain rule decoupling the calculation of
# the free energy density from the degrees of freedom of the displacement
#
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
function ×(ϕ::adiff.D1{N,T},F::Array{adiff.D1{P,T}}) where {N,P,T}
  val  = ϕ.v
  grad = adiff.Grad(zeros(T,P))
  for ii=1:N
    grad += ϕ.g[ii]*F[ii].g
  end  
  adiff.D1(val, grad)
end
# functions for calculating residuals and stiffness matrix
function makeϕrKt(Φ::Vector{<:adiff.D2}, elems::Vector{<:Elems}, u)

  N  = length(u) 
  Nt = 0
  for ϕ in Φ
    # Nt += length(ϕ.g)*length(ϕ.g)
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
#
# methods for evaluating the deformation gradient at integration points
#
function getF(elem::C3DElems{P,M,T,I} where {M,T,I}, u::Array{D}) where {P,D}
  u0, v0, w0 = u[1:3:end],  u[2:3:end],  u[3:3:end]
  F = fill(Array{D,2}(undef,3,3), P)
  @inbounds for ii = 1:P
    Nx, Ny, Nz = elem.Nx[ii], elem.Ny[ii], elem.Nz[ii]
    F[ii] = [Nx⋅u0 Ny⋅u0 Nz⋅u0;
             Nx⋅v0 Ny⋅v0 Nz⋅v0;
             Nx⋅w0 Ny⋅w0 Nz⋅w0 ] + I
  end
  F
end
function getF(elem::C3DElems, u::Matrix, ii::Integer)
  Nx, Ny, Nz = elem.Nx[ii], elem.Ny[ii], elem.Nz[ii]
  u0, v0, w0 = u[1:3:end],  u[2:3:end],  u[3:3:end]

  [Nx⋅u0 Ny⋅u0 Nz⋅u0;
   Nx⋅v0 Ny⋅v0 Nz⋅v0;
   Nx⋅w0 Ny⋅w0 Nz⋅w0 ] + I
end
function getF(elem::C2DElems{P,M,T,I} where {M,T,I}, u::Array{D}) where {P,D}
  u0, v0 = u[1:2:end],  u[2:2:end]
  F = fill(Array{D,2}(undef,2,2), P)
  @inbounds for ii = 1:P
    Nx, Ny = elem.Nx[ii], elem.Ny[ii]
    F[ii]  = [Nx⋅u0 Ny⋅u0;
              Nx⋅v0 Ny⋅v0] + I
  end
  F
end
function getF(elem::C2DElems{P,M,T,I} where {M,T,I}, u::Array{D}, ii::Integer) where {P,D}
  u0, v0 = u[1:2:end],  u[2:2:end]
  Nx, Ny = elem.Nx[ii], elem.Ny[ii]
  [Nx⋅u0 Ny⋅u0;
   Nx⋅v0 Ny⋅v0] + I
end
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
 function detJ(F)
   if (length(F)==9)
     # F[1]F[5]F[9]-F[1]F[6]F[8]-F[2]F[4]F[9]+F[2]F[6]F[7]+F[3]F[4]F[8]-F[3]F[5]F[7]
     F[1]*(F[5]F[9]-F[6]F[8])-F[2]*(F[4]F[9]-F[6]F[7])+F[3]*(F[4]F[8]-F[5]F[7])
   else
     F[1]F[4]-F[2]F[3]
   end
 end
 detJ(elem,u,ii)  = detJ(getF(elem, u, ii))
 getV(elem,u)     = sum([elem.wgt[ii]detJ(elem,u,ii) for ii in 1:length(elem.wgt)])
 detJ(elem,u)     = getV(elem,u)/elem.V
 getI3(elem,u,ii) = detJ(getF(elem, u, ii))^2
 getI3(elem,u)    = sum([elem.wgt[ii]getI3(elem,u,ii) for ii in 1:length(elem.wgt)])/elem.V
 #
 function getinfo(elem::CElems{P}, u::Matrix{<:Number}; info=:detF) where P
   F = sum([getF(elem, u, ii) for ii=1:P])/P
   Materials.getinfo(F, elem.mat, info=info)
 end
 getinfo(elems::Array, u; info=:detF) =  [getinfo(elem, u[:,elem.nodes], info=info) for elem in elems]
 # helper functions
 # find the Gauss-Legendre quadrature points and weights
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
   # (x,w)
 end

end

