__precompile__()

module Elements

using LinearAlgebra, Printf
using Distributed, SparseArrays
using ProgressMeter, Dates, StatsBase

using ..adiff, ..Materials
import ..Materials.getϕ
import Base.copy

p = Int64(nworkers())
function setp(x)
  global p = Int64(x)
end
# continous elements 
struct C2D{M}
  nodes::Vector{<:Integer}
  Nx::NTuple{M,Array{Float64,1}}
  Ny::NTuple{M,Array{Float64,1}}
  wgt::NTuple{M,Float64}
  V::Real
  mat::Materials.Material
end
struct C3D{M}
  nodes::Vector{<:Integer}
  Nx::NTuple{M,Array{Float64,1}}
  Ny::NTuple{M,Array{Float64,1}}
  Nz::NTuple{M,Array{Float64,1}}
  wgt::NTuple{M,Float64}
  V::Real
  mat::Materials.Material
end
struct CAS{M} 
  nodes::Vector{<:Integer}
  N0::NTuple{M,Array{Float64,1}}
  Nx::NTuple{M,Array{Float64,1}}
  Ny::NTuple{M,Array{Float64,1}}
  X0::NTuple{M,Float64}
  wgt::NTuple{M,Float64}
  V::Real
  mat::Materials.Material
end
#  structural elements
struct Rod
  nodes::Vector{<:Integer}         # node id
  r0::Array{Float64}          # 
  l0::Float64
  A::Float64                  # area 
  mat::Materials.Material     # material properties
end
struct Beam
  nodes::Vector{<:Integer}
  r0::Array{Float64,1}
  L::Real
  t::Real
  w::Real
  lgwx::Array{Tuple{Float64,Float64},1}
  lgwy::Array{Tuple{Float64,Float64},1}
  mat::Materials.Material
end
Elems   = Union{Rod, Beam, C2D, C3D, CAS}
CElems  = Union{C2D, C3D, CAS}
CartEls = Union{C2D, C3D}
# copy
copy(elem::C2D{M}) where M = C2D{M}(copy(elem.nodes), elem.Nx, elem.Ny, elem.wgt, elem.V, elem.mat)
copy(elem::C3D{M}) where M = C3D{M}(copy(elem.nodes), elem.Nx, elem.Ny, elem.Nz, elem.wgt, elem.V, elem.mat)
copy(elem::CAS{M}) where M = CAS{M}(copy(elem.nodes), elem.N0, elem.Nx, elem.Ny, elem.X0, elem.wgt, elem.V, elem.mat)
copy(elem::Rod)            = Rod(copy(elem.nodes), elem.r0, elem.l0, elem.A, elem.mat)
copy(elem::Beam)           = Beam(copy(elem.nodes), elem.r0, elem.L, elem.w, elem.lgwx, elem.lgwy, elem.mat)
# clone
function clone(elem::C2D{M}) where M
  nodes = copy(elem.nodes)
  Nx    = tuple([copy(x) for x in elem.Nx]...)
  Ny    = tuple([copy(x) for x in elem.Ny]...)

  C2D{M}(nodes, Nx, Ny, elem.wgt, elem.V, elem.mat)
end
function clone(elem::C3D{M}) where M
  nodes = copy(elem.nodes)
  Nx    = tuple([copy(x) for x in elem.Nx]...)
  Ny    = tuple([copy(x) for x in elem.Ny]...)
  Nz    = tuple([copy(x) for x in elem.Nz]...)

  C3D{M}(nodes, Nx, Ny, Nz, elem.wgt, elem.V, elem.mat)
end
function clone(elem::CAS{M}) where M
  nodes = copy(elem.nodes)
  N0    = tuple([copy(x) for x in elem.N0]...)
  Nx    = tuple([copy(x) for x in elem.Nx]...)
  Ny    = tuple([copy(x) for x in elem.Ny]...)

  CAS{M}(nodes, N0, Nx, Ny, elem.X0, elem.wgt, elem.V, elem.mat)
end
# structure for constraint eqs
struct ConstEq
  func
  iDoFs::Array{Int64}
  D::Type
end
ConstEq(func, iDoFs) = ConstEq(func, iDoFs, adiff.D2)
# constructors
function Rod(nodes, p0, A; mat=Materials.Hooke()) 

  r0  = p0[2]-p0[1] 
  l0  = norm(r0)
  Rod(nodes.|>Int64, r0.|>Float64, l0|>Float64, A|>Float64, mat)
end
function Beam(nodes, p0, t, w; mat=Materials.Hooke(1, 0.3), Nx = 5, Ny = 3)

  lgwx = lgwt(Nx)
  lgwy = lgwt(Ny, a=-0.5, b=0.5)

  d0  = p0[2]-p0[1] 
  L   = norm(d0)
  r0  = d0/L

  Beam(nodes, r0, L, Float64(t), Float64(w), lgwx, lgwy, mat)
end
function Tria(nodes::Vector{<:Integer}, 
              p0::Vector{Vector{T}} where T<:Number ;
              mat=Materials.Hooke())
  (Nx,Ny,wgt,A) = begin
    (x1,x2,x3) = (p0[1][1],p0[2][1],p0[3][1])
    (y1,y2,y3) = (p0[1][2],p0[2][2],p0[3][2])
    Delta      = x1*y2-x2*y1-x1*y3+x3*y1+x2*y3-x3*y2
    Nx         = [y2-y3,y3-y1,y1-y2]./Delta
    Ny         = [x3-x2,x1-x3,x2-x1]./Delta
    A          = abs(Delta)/2
    ((Nx,),(Ny,),(A,), A)
  end

  C2D(nodes,Nx,Ny,wgt,A,mat) 
end
function Quad(nodes::Vector{<:Integer}, 
              p0::Vector{Vector{T}} where T<:Number;
              mat=Materials.Hooke())

  (A,Nx,Ny,wgt) = begin 
    r        = [-1, 1]*0.577350269189626 # √3/3
    N(ξ,η)   = [(1-ξ)*(1-η),(1+ξ)*(1-η),(1+ξ)*(1+η),(1-ξ)*(1+η)]/4

    Nx  = Array{Array{Float64,1},2}(undef,2,2)
    Ny  = Array{Array{Float64,1},2}(undef,2,2)
    wgt = Array{Float64,2}(undef,2,2)
    A   = 0
    for (ii, ξ) in enumerate(r), (jj, η) in enumerate(r)
      N0  = N(adiff.D2([ξ,η])...)
      p   = sum([N0[ii]p0[ii] for ii in 1:4])
      J   = [p[ii].g[jj] for jj in 1:2, ii in 1:2]
      Nxy = J\hcat(adiff.grad.(N0)...)

      Nx[ii,jj]  = Nxy[1,:]
      Ny[ii,jj]  = Nxy[2,:]
      wgt[ii,jj] = abs(det(J))

      A += wgt[ii,jj]
    end
    (A,tuple(Nx...),tuple(Ny...),tuple(wgt...))
  end

  C2D(nodes,Nx,Ny,wgt,A,mat) 
end
function Tet04(nodes::Vector{<:Integer}, 
               p0::Vector{Vector{T}} where T<:Number;
               mat=Materials.Hooke())
  (V, Nx, Ny, Nz) = begin
    A        = ones(4,4)
    A[1,2:4] = p0[1]
    A[2,2:4] = p0[2]
    A[3,2:4] = p0[3]
    A[4,2:4] = p0[4]
    C        = inv(A)
    V        = det(A)/6
    (V,(C[2,:],),(C[3,:],),(C[4,:],))
  end
  C3D(nodes,Nx,Ny,Nz,(1.0,),V,mat) 
end
function Tet10(nodes::Vector{<:Integer}, 
               p0::Vector{Vector{T}} where T<:Number;
               mat=Materials.Hooke())

  (V, Nx, Ny, Nz) = begin

    La = 0.585410196624968
    Lb = 0.138196601125010
    ξ  = [La, Lb, Lb, Lb] 
    A  = hcat([[1,p[1],p[2],p[3],p[1]^2,p[2]^2,p[3]^2,p[2]p[3],p[1]p[3],p[1]p[2]] for p in p0]...)
    C  = inv(A)
    V  = det(A[1:4,1:4])/6

    Nx,Ny,Nz = zeros(10,4),zeros(10,4),zeros(10,4)

    for ii in 1:4
      ξ        = circshift([La, Lb, Lb, Lb], ii)
      p        = p0[1]ξ[1]+p0[2]ξ[2]+p0[3]ξ[3]+p0[4]ξ[4]
      Nx[:,ii] = C[:,2]+2C[:,5]p[1]+C[:,9]p[3]+C[:,10]p[2] 
      Ny[:,ii] = C[:,3]+2C[:,6]p[2]+C[:,8]p[3]+C[:,10]p[1]
      Nz[:,ii] = C[:,4]+2C[:,7]p[3]+C[:,8]p[2]+C[:, 8]p[1] 
    end

    (V,
     (tuple([Nx[:,ii] for ii in 1:4]...),), 
     (tuple([Ny[:,ii] for ii in 1:4]...),), 
     (tuple([Nz[:,ii] for ii in 1:4]...),))
  end

  C3D(nodes,Nx,Ny,Nz,V,mat) 
end
function Hex08(nodes::Vector{<:Integer}, 
               p0::Vector{Vector{T}} where T<:Number;
               mat=Materials.Hooke())
  (V, Nx, Ny, Nz, wgt) = begin
    r        = [-1, 1]*0.577350269189626 # √3/3
    # N(ξ,η,ζ) = [(1-ξ)*(1-η)*(1+ζ),(1-ξ)*(1-η)*(1-ζ),(1-ξ)*(1+η)*(1-ζ),(1-ξ)*(1+η)*(1+ζ),
    #             (1+ξ)*(1-η)*(1+ζ),(1+ξ)*(1-η)*(1-ζ),(1+ξ)*(1+η)*(1-ζ),(1+ξ)*(1+η)*(1+ζ)]/8
    N(ξ,η,ζ) = [(1-ξ)*(1-η)*(1-ζ),(1+ξ)*(1-η)*(1-ζ),
                (1+ξ)*(1+η)*(1-ζ),(1-ξ)*(1+η)*(1-ζ),
                (1-ξ)*(1-η)*(1+ζ),(1+ξ)*(1-η)*(1+ζ),
                (1+ξ)*(1+η)*(1+ζ),(1-ξ)*(1+η)*(1+ζ)]/8
    Nx  = Array{Array{Float64,1},3}(undef,2,2,2)
    Ny  = Array{Array{Float64,1},3}(undef,2,2,2)
    Nz  = Array{Array{Float64,1},3}(undef,2,2,2)
    wgt = Array{Float64,3}(undef,2,2,2)
    V   = 0
    for (ii, ξ) in enumerate(r), (jj, η) in enumerate(r), (kk, ζ) in enumerate(r)
      N0   = N(adiff.D2([ξ,η,ζ])...)
      p    = sum([N0[ii]p0[ii] for ii in 1:8])
      J    = [p[ii].g[jj] for jj in 1:3, ii in 1:3]
      Nxyz = J\hcat(adiff.grad.(N0)...)

      Nx[ii,jj,kk]  = Nxyz[1,:]
      Ny[ii,jj,kk]  = Nxyz[2,:]
      Nz[ii,jj,kk]  = Nxyz[3,:]
      wgt[ii,jj,kk] = det(J)

      V +=wgt[ii,jj,kk]
    end
    (V,tuple(Nx...),tuple(Ny...),tuple(Nz...),tuple(wgt...))
  end
  C3D(nodes,Nx,Ny,Nz,wgt,V,mat) 
end
function ASTria(nodes::Vector{<:Integer},
                p0::Vector{Vector{T}} where T<:Number;
                mat=Materials.Hooke())
  (N,Nx,Ny,X0,wgt,A) = begin 
    (x1, x2, x3) = (p0[1][1], p0[2][1], p0[3][1])
    (y1, y2, y3) = (p0[1][2], p0[2][2], p0[3][2])

    Delta = x1*y2-x2*y1-x1*y3+x3*y1+x2*y3-x3*y2
    N     = [1, 1, 1]/3
    Nx    = [y2-y3, y3-y1, y1-y2]/Delta
    Ny    = [x3-x2, x1-x3, x2-x1]/Delta
    A     = abs(Delta)/2
    X0    = (x1+x2+x3)/3
    wgt   = A*2π*X0
    ((N,),(Nx,),(Ny,),(X0,),(wgt,),A)
  end
  CAS(nodes,N,Nx,Ny,X0,wgt,A,mat) 
end
function ASQuad(nodes::Vector{<:Integer},
                p0::Vector{Vector{T}} where T<:Number;
                mat=Materials.Hooke())
  (V,N0,Nx,Ny,X0,wgt) = begin
    r        = [-1, 1]*0.577350269189626 # √3/3
    N(ξ,η)   = [(1-ξ)*(1-η),(1+ξ)*(1-η),(1+ξ)*(1+η),(1-ξ)*(1+η)]/4

    N0  = Array{Array{Float64,1},2}(undef,2,2)
    Nx  = Array{Array{Float64,1},2}(undef,2,2)
    Ny  = Array{Array{Float64,1},2}(undef,2,2)
    X0  = Array{Float64,2}(undef,2,2)
    wgt = Array{Float64,2}(undef,2,2)
    V   = 0
    for (ii, ξ) in enumerate(r), (jj, η) in enumerate(r)
      Nij = N(adiff.D2([ξ,η])...)
      p   = sum([Nij[ii]p0[ii] for ii in 1:4])
      J   = [p[ii].g[jj] for jj in 1:2, ii in 1:2]
      Nxy = J\hcat(adiff.grad.(Nij)...)

      N0[ii,jj]  = adiff.val.(Nij)
      Nx[ii,jj]  = Nxy[1,:]
      Ny[ii,jj]  = Nxy[2,:]
      X0[ii,jj]  = p[1].v 
      # wgt[ii,jj] = det(J)
      wgt[ii,jj] = det(J)*2π*p[1].v

      V +=wgt[ii,jj]
    end
    (V,tuple(N0...),tuple(Nx...),tuple(Ny...),tuple(X0...),tuple(wgt...))
  end
  CAS(nodes,N0,Nx,Ny,X0,wgt,V,mat) 
end
# elastic energy evaluation functions for elements
function getϕ(elem::Rod,  u::Array{<:Number,2})

  l   = norm(elem.r0+u[:,2]-u[:,1])
  F11 = l/elem.l0
  elem.A*elem.l0*getϕ(F11, elem.mat)    

end
function getϕ(elem::Beam, u::Array{<:Number,2})

  L, r0, t, w = elem.L, elem.r0, elem.t, elem.w
  T    = [r0[1] r0[2]; -r0[2] r0[1]]
  u0   = vcat(T*u[1:2,1], u[3,1], T*u[1:2,2], u[3,2])
  u0_x = (u0[4]-u0[1])/L

  ϕ  = 0
  for (r,wr) in elem.lgwx
    x, dx     = r*L, wr*L

    v0_x  = (-6x/L^2 + 6x^2/L^3)u0[2] +
    (1 - 4x/L + 3x^2/L^2)u0[3] +
    (6x/L^2 - 6x^2/L^3)u0[5] +
    (-2x/L + 3x^2/L^2)u0[6]

    v0_xx = (-6/L^2 + 12x/L^3)u0[2] + 
    (-4/L + 6x/L^2)u0[3] + 
    (6/L^2 - 12x/L^3)u0[5] + 
    (-2/L + 6x/L^2)u0[6]

    for (s,ws) in elem.lgwy
      y, dy  = s*elem.t, ws*elem.t

      dV   = dx*dy*elem.w
      C11 = (1+u0_x-v0_xx*y)^2 + v0_x^2
      ϕ   += getϕ(C11, elem.mat)*dV
    end
  end
  return ϕ
end
function getϕ(elem::T where T<:CElems, u::Array{<:Number,2})
  M = length(elem.wgt)
  if isa(u[1], adiff.D2) 
    ϕ = sum([begin
               F = getF(elem,u,ii)
               ϕ = getϕ(adiff.D2(getfield.(F,:v)),elem.mat)
               elem.wgt[ii]cross(ϕ,F)
             end  for ii in 1:M])
  else
    ϕ = sum([elem.wgt[ii]getϕ(getF(elem,u,ii), elem.mat) for ii in 1:M])
  end 
end
function cross(ϕ, F)
  N = length(F)
  g = sum([ϕ.g[ii]F[ii].g for ii in 1:N])
  h = sum([0.5ϕ.h[ii,jj]*(F[ii].g*F[jj].g+F[jj].g*F[ii].g) for jj=1:N for ii=1:N])
  adiff.D2(ϕ.v, g, h) 
end
# methods for evaluating def. gradient
function getF(elem::C3D, u::Array{N} where N, ii::Int64)

  Nx, Ny, Nz = elem.Nx[ii], elem.Ny[ii], elem.Nz[ii]

  [Nx⋅u[1:3:end] Ny⋅u[1:3:end] Nz⋅u[1:3:end];
   Nx⋅u[2:3:end] Ny⋅u[2:3:end] Nz⋅u[2:3:end];
   Nx⋅u[3:3:end] Ny⋅u[3:3:end] Nz⋅u[3:3:end] ] + I
end
function getF(elem::C2D, u::Array{N} where N, ii::Int64) 

  Nx, Ny = elem.Nx[ii], elem.Ny[ii]
  my0    = zero(u[1])

  # [Nx⋅u[1:2:end] Ny⋅u[1:2:end] my0;
  #  Nx⋅u[2:2:end] Ny⋅u[2:2:end] my0;
  #  my0           my0           my0] + I
  [Nx⋅u[1:2:end] Ny⋅u[1:2:end];
   Nx⋅u[2:2:end] Ny⋅u[2:2:end]] + I
end
function getF(elem::CAS, u::Array{N} where N, ii::Int64) 
  Nx,  Ny   = elem.Nx[ii],   elem.Ny[ii]
  N0,  X0   = elem.N0[ii],   elem.X0[ii]
  u0x, u0y  = Nx⋅u[1:2:end], Ny⋅u[1:2:end]
  v0x, v0y  = Nx⋅u[2:2:end], Ny⋅u[2:2:end]
  w0z       = N0⋅u[1:2:end]/X0
  my0       = zero(u[1])

  [u0x  u0y   my0;
   v0x  v0y   my0;
   my0  my0   w0z] + I
end
function getJ(F)
  if (length(F)==9)
    F[1]F[5]F[9]-F[1]F[6]F[8]-F[2]F[4]F[9]+F[2]F[6]F[7]+F[3]F[4]F[8]-F[3]F[5]F[7]
  else
    F[1]F[4]-F[2]F[3]
  end
end
getV(elem,u)     = sum([elem.wgt[ii]getJ(elem,u,ii) for ii in 1:length(elem.wgt)])
getJ(elem,u,ii)  = getJ(getF(elem, u, ii))
getJ(elem,u)     = getV(elem,u)/elem.V
getI3(elem,u,ii) = getJ(getF(elem, u, ii))^2
getI3(elem,u)    = sum([elem.wgt[ii]getI3(elem,u,ii) for ii in 1:length(elem.wgt)])/elem.V
#
# elastic energy evaluation functions for models (lists of elements)
function getϕ(elems::Array, u; T=Threads.nthreads())

  nDoFs  = length(u)
  nElems = length(elems)

  Φ = zeros(size(elems))
  r = zeros(nDoFs)
  C = [spzeros(nDoFs, nDoFs) for ii = 1:T]
  Threads.@threads for kk = 1:T
    for ii = kk:T:nElems
      elem           =  elems[ii]
      nodes          =  elem.nodes
      iDoFs          =  LinearIndices(u)[:,nodes][:]
      ϕ              =  Elements.getϕ(elem, adiff.D2(u[:,nodes]))
      Φ[ii]          =  adiff.val(ϕ)
      r[iDoFs]       += adiff.grad(ϕ)
      C[kk][iDoFs,iDoFs] += adiff.hess(ϕ)
    end
  end

  (Φ, r, sum(C))
end 
#=
function getϕ(elems::Array, u)

nElems = length(elems)
if p==1 || nElems<=p
(Φ, r, C) = getϕ(elems, u, 1:nElems)
else
nDoFs  = length(u)

Φ = zeros(nElems)
r = zeros(nDoFs)
C = spzeros(nDoFs, nDoFs)

chunks = split(nElems, Elements.p)
procs  = [@spawn getϕ(elems, u, chunk)  for chunk in chunks]

for (ii,chunk) in enumerate(chunks)
retval =   fetch(procs[ii])
Φ      .+= retval[1]
r      .+= retval[2]
C      .+= retval[3]
end
end
(Φ, r, C)

end
function getϕ(elems::Array, u, chunk)

nDoFs  = length(u)
nElems = length(elems)

Φ = zeros(size(elems))
r = zeros(nDoFs)
C = spzeros(nDoFs, nDoFs)
for ii ∈ chunk 
elem           =  elems[ii]
nodes          =  elem.nodes
iDoFs          =  LinearIndices(u)[:,nodes][:]  
ϕ              =  getϕ(elem, adiff.D2(u[:,nodes]))
Φ[ii]          =  adiff.val(ϕ)
r[iDoFs]       += adiff.grad(ϕ)
C[iDoFs,iDoFs] += adiff.hess(ϕ)
end
(Φ, r, C)
end
=#
function getϕ(eqns::Array{ConstEq}, u::Array{Float64}, λ::Array{Float64})

  nEqs   = length(eqns)
  if p==1 || nEqs<=p
    (veqs, reqs, Keqs) = getϕ(eqns, u, λ, 1:nEqs)
  else
    nDoFs  = length(u)
    veqs   = zeros(nEqs)
    reqs   = spzeros(nDoFs, nEqs)
    Keqs   = spzeros(nDoFs, nDoFs)
    chunks = split(nEqs, Elements.p)
    procs  = [@spawn getϕ(eqns, u, λ, chunk)  for chunk in chunks]

    for ii in 1:p 
      retval  = fetch(procs[ii])
      veqs .+= retval[1]
      reqs .+= retval[2]
      Keqs .+= retval[3]
    end
  end
  (veqs, reqs, Keqs)
end
function getϕ(eqns::Array{ConstEq}, u::Array{Float64}, λ::Array{Float64}, chunk)

  nEqs  = length(eqns)
  nDoFs = length(u)

  veqs  = zeros(nEqs)
  reqs  = spzeros(nDoFs, nEqs)
  Keqs  = spzeros(nDoFs, nDoFs)

  for ii ∈ chunk 
    eqn               =  eqns[ii]
    iDoFs             =  eqn.iDoFs
    ϕ                 =  eqn.func(eqn.D(u[iDoFs]))
    veqs[ii]          =  adiff.val(ϕ)
    reqs[iDoFs,ii]    =  adiff.grad(ϕ)
    Keqs[iDoFs,iDoFs] += λ[ii]adiff.hess(ϕ)
  end
  (veqs, reqs, Keqs)  
end
function getinfo(elem::Elems, u::Array{<:Number,2}; info=:detF)
  M = length(elem.Nx)
  F = sum([getF(elem, u, ii) for ii in 1:M])/M
  Materials.getinfo(F, elem.mat, info=info)
end
getinfo(elems::Array, u; info=:detF) =  [getinfo(elem, u[:,elem.nodes], info=info) for elem in elems]
# solver 
function solve(elems, u;
               N          = 11,
               LF         = range(1e-4, stop=1, length=N), 
               eqns       = [],
               λ          = zeros(length(eqns)),
               ifree      = isnan.(u),
               fe         = zeros(size(u)),
               bprogress  = false,
               becho      = false,
               dTol       = 1e-5,
               dTolu      = dTol,
               dTole      = 1e2dTol,
               dNoise     = 1e-12,
               maxiter    = 11,
               bechoi     = false,
               bprogressi = false,
               ballus     = true,
               bpredict   = true,
               maxupdt    = NaN)

  N     = length(LF)
  t0    = Base.time_ns()
  beqns = length(eqns)>0
  if bprogress; p    = ProgressMeter.Progress(length(LF)); end
  if ballus;    allu = [];  end

  fnew  = copy(fe)
  icnst = .!ifree
  unew  = copy(u)
  unew[ifree] .= 0
  uold  = zeros(size(unew))
  λnew  = copy(λ)

  for (ii,LF) in enumerate(LF)
    unew[icnst] .= u[icnst]*LF 
    fnew[ifree] .= fe[ifree]*LF 
    lastu       = copy(unew)
    lastλ       = copy(λnew)
    T           = @elapsed (bfailed, normr, iter) = 
    try 
      solvestep!(elems, uold, unew, ifree, 
                 eqns      = eqns,
                 λ         = λnew,
                 fe        = fnew, 
                 dTole     = dTole,
                 dTolu     = dTolu,
                 dNoise    = dNoise,
                 maxiter   = maxiter,
                 bprogress = bprogressi,
                 becho     = bechoi,
                 bpredict  = bpredict,
                 maxupdt   = maxupdt)
    catch
      (true, Inf, 0)
    end
    if bfailed 
      @printf("\n!! failed at LF: %.3f, with normr/dTol: %.3e\n", LF, normr/dTol)
      unew = lastu
      λnew = lastλ
      break
    else
      uold[:] .= unew[:]
      if ballus
        if beqns
          push!(allu, (copy(unew), copy(fnew), copy(λnew)))
        else
          push!(allu, (copy(unew), copy(fnew)))
        end
      end
      bprogress && ProgressMeter.next!(p)
      becho     && @printf("step %3i/%i, LF=%.3f, done in %2i iter, after %.2f sec.\n",
                           ii,N,LF,iter,T)
    end
    becho && flush(stdout)
  end
  becho && @printf("completed in %s\n",(Base.time_ns()-t0)÷1e9|>
                   Dates.Second|>Dates.CompoundPeriod|>Dates.canonicalize)
  becho && flush(stdout)

  ballus ? allu : unew
end  
function solvestep!(elems, uold, unew, bfreeu;
                    fe        = zeros(length(unew)),
                    eqns      = [],
                    λ         = zeros(length(eqns)),
                    dTol      = 1e-5,
                    dTolu     = dTol,
                    dTole     = 1e2dTol,
                    dNoise    = 1e-12,
                    maxiter   = 11,
                    becho     = false,
                    bprogress = false,
                    bpredict  = true,
                    maxupdt   = NaN)

  if bprogress
    p = ProgressMeter.ProgressThresh(dTolu)
  end

  ifreeu    = findall(bfreeu[:])
  icnstu    = findall(.!bfreeu[:])

  nEqs      = length(eqns)
  nfreeu    = length(ifreeu)
  ncnstu    = length(icnstu)
  nDoFs     = nfreeu + nEqs
  iius      = 1:nfreeu
  iieqs     = nfreeu .+ (1:nEqs)

  bdone     = false
  bfailed   = false
  iter      = 0
  normupdt  = 0
  normre    = NaN
  oldupdt   = zeros(nDoFs)
  updt      = zeros(nDoFs)
  if nEqs != 0 
    H       = spzeros(nDoFs,nDoFs)
  end

  # predictor step
  if bpredict
    deltat = @elapsed begin
      Δucnst    = unew[icnstu]-uold[icnstu]
      (Φ,fi,Kt) = getϕ(elems, uold)
      if nEqs == 0
        res              = fi[ifreeu]-fe[ifreeu]
        res[:]         .-= Kt[ifreeu,icnstu]*Δucnst
        # updt[:]          = Kt[ifreeu,ifreeu]\res
        updt[:]          = qr(Kt[ifreeu,ifreeu])\res
        unew[ifreeu]    .= uold[ifreeu] .+ updt 
        normupdt         = maximum(abs.(updt))
      else
        (vEqs,rEqs,KEqs) = getϕ(eqns, uold, λ)
        resu             = fi[ifreeu]-fe[ifreeu]-rEqs[ifreeu,:]*λ
        resu[:]        .-= (Kt[ifreeu,icnstu]-KEqs[ifreeu,icnstu])*Δucnst
        rese             = -vEqs
        res              = vcat(resu, rese)

        H[iius,iius]     = Kt[ifreeu,ifreeu]-KEqs[ifreeu,ifreeu]
        H[iius,iieqs]    = -rEqs[ifreeu,:]
        H[iieqs,iius]    = transpose(H[iius,iieqs])
        H[iieqs,iieqs]   = spdiagm(0=>dNoise*randn(nEqs))
        updt[:]          = qr(H)\res
        unew[ifreeu]    .= uold[ifreeu] .+ updt[iius]
        λ              .-= updt[iieqs]
        normupdt         = maximum(abs.(updt[iius]))
      end
    end
    becho && @printf("\npredictor step done in %.2f sec., ", deltat)
    becho && @printf("with normupdt: %.2e, starting corrector loop\n", normupdt); flush(stdout)
  else
    unew[ifreeu] .= uold[ifreeu]
  end
  # corrector loop
  while !bdone & !bfailed 
    global normru
    oldupdt = copy(updt)
    tic     = Base.time_ns()
    (Φ,fi,Kt) = getϕ(elems, unew)

    if nEqs == 0
      res    = fe[ifreeu]-fi[ifreeu]      
      norm0  = ncnstu > 0     ? norm(fi[icnstu])/ncnstu : 0
      normru = norm0  > dTolu ? norm(res)/nfreeu/norm0  : norm(res)/nfreeu
      bdone  = (normru<dTolu)
    else
      (vEqs,rEqs,KEqs) = getϕ(eqns, unew, λ)
      resu   = fi[ifreeu]-fe[ifreeu]-rEqs[ifreeu,:]*λ
      rese   = -vEqs
      res    = -vcat(resu, rese)
      norm0  = ncnstu > 0     ? norm(fi[icnstu])/ncnstu : 0
      normru = norm0  > dTolu ? norm(res)/nfreeu/norm0  : norm(res)/nfreeu
      normre = maximum(abs.(rese))
      bdone  = (normru<dTolu) && (normre<dTole)
    end

    if bdone
      fe[:]   = nEqs==0 ? fi[:] : fi[:]-rEqs*λ
    elseif iter < maxiter
      if nEqs == 0
        updt[:]         = qr(Kt[ifreeu,ifreeu])\res
        normupdt        = maximum(abs.(updt))
        if !isnan(maxupdt)
          if normupdt > maxupdt  
            updt      .*= (maxupdt/normupdt)
            normupdt    = maximum(abs.(updt))
          end
        end
        unew[ifreeu]  .+= updt
      else
        H[iius,iius]    = Kt[ifreeu,ifreeu]-KEqs[ifreeu,ifreeu]
        H[iius,iieqs]   = -rEqs[ifreeu,:]
        H[iieqs,iius]   = transpose(H[iius,iieqs])
        H[iieqs,iieqs]  = spdiagm(0=>dNoise*randn(nEqs))
        updt[:]         = qr(H)\res
        normupdt        = maximum(abs.(updt[iius]))
        if !isnan(maxupdt)
          if normupdt > maxupdt  
            updt      .*= (maxupdt/normupdt)
            normupdt    = maximum(abs.(updt[iius]))
          end
        end
        unew[ifreeu]  .+= updt[iius]
        λ             .+= updt[iieqs]
      end              
    else
      bfailed = true
    end    
    bprogress && ProgressMeter.update!(p, normru)
    if becho 
      if (bdone | bfailed) 
        @printf("iter: %2i, norm0: %.2e, normru: %.2e, normre: %.2e, eltime: %.2f sec.\n", 
                iter, norm0, normru, normre, Int64(Base.time_ns()-tic)/1e9)
      else
        @printf("iter: %2i, norm0: %.2e, normru: %.2e, normre: %.2e, normupdt: %.2e, α: %6.3f, eltime: %.2f sec.\n", 
                iter, norm0, normru, normre, normupdt, 
                oldupdt⋅updt/norm(updt)/norm(oldupdt), Int64(Base.time_ns()-tic)/1e9)
      end
      flush(stdout)
    end
    iter  += 1
  end

  (bfailed, normru, iter)
end
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
function renumber!(nodes, elems)

  unodes   = sort(unique(vcat([elem.nodes for elem in elems]...)))
  el_nodes = hcat([elem.nodes for elem in elems]...)
  shift    = maximum(unodes) + 1

  el_nodes .+= shift
  for (ii, nodeid) in enumerate(unodes)
    nodeid += shift
    el_nodes[el_nodes .== nodeid] .= ii
  end
  nodes = nodes[unodes]

  for (ii,elem) in enumerate(elems)
    elem.nodes[:] .= el_nodes[:,ii]
  end
  (nodes, elems)
end
function split(N::Int64, p::Int64)
  n    = Int64(floor(N/p))
  nEls = ones(Int64, p)*n
  nEls[1:N-p*n] .+= 1

  slice = [ range(sum(nEls[1:ii-1])+1, length=nEls[ii])
           for ii in 1:p]
end

end
