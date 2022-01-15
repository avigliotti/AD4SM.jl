__precompile__()

module Elements

using LinearAlgebra

using ..adiff, ..Materials
import ..Materials.getϕ
import Base.copy

# continous elements 
struct C1D{N,P,M}
  nodes::Vector{<:Integer}
  Nx::NTuple{P,Array{<:Number,1}}
  wgt::NTuple{P,Number}
  V::Number
  mat::M
  C1D(nodes,Nx,wgt,V,mat)=new{length(nodes),length(wgt),typeof(mat)}(nodes,Nx,wgt,V,mat)
end
struct C2D{N,P,M}
  nodes::Vector{<:Integer}
  Nx::NTuple{P,Array{<:Number,1}}
  Ny::NTuple{P,Array{<:Number,1}}
  wgt::NTuple{P,Number}
  V::Number
  mat::M
  C2D(nodes,Nx,Ny,wgt,V,mat)=new{length(nodes),length(wgt),typeof(mat)}(nodes,Nx,Ny,wgt,V,mat)
end
struct C3D{N,P,M}
  nodes::Vector{<:Integer}
  Nx::NTuple{P,Array{<:Number,1}}
  Ny::NTuple{P,Array{<:Number,1}}
  Nz::NTuple{P,Array{<:Number,1}}
  wgt::NTuple{P,Number}
  V::Number
  mat::M
  C3D(nodes,Nx,Ny,Nz,wgt,V,mat)=new{length(nodes),length(wgt),typeof(mat)}(nodes,Nx,Ny,Nz,wgt,V,mat)
end
struct CAS{N,P,M} 
  nodes::Vector{<:Integer}
  N0::NTuple{P,Array{<:Number,1}}
  Nx::NTuple{P,Array{<:Number,1}}
  Ny::NTuple{P,Array{<:Number,1}}
  X0::NTuple{P,Number}
  wgt::NTuple{P,Number}
  V::Number
  mat::M
  CAS(nodes,N0,Nx,Ny,X0,wgt,V,mat)=new{length(nodes),length(wgt),typeof(mat)}(nodes,N0,Nx,Ny,X0,wgt,V,mat)
end
# continous elements with phase
struct C2DP{N,P,M}
  nodes::Vector{<:Integer}
  N0::NTuple{P,Array{<:Number,1}}
  Nx::NTuple{P,Array{<:Number,1}}
  Ny::NTuple{P,Array{<:Number,1}}
  wgt::NTuple{P,Number}
  V::Number
  mat::M
  C2DP(nodes,N0,Nx,Ny,wgt,V,mat)=new{length(nodes),length(wgt),typeof(mat)}(nodes,N0,Nx,Ny,wgt,V,mat)
end
struct C3DP{N,P,M}
  nodes::Vector{<:Integer}
  N0::NTuple{P,Array{<:Number,1}}
  Nx::NTuple{P,Array{<:Number,1}}
  Ny::NTuple{P,Array{<:Number,1}}
  Nz::NTuple{P,Array{<:Number,1}}
  wgt::NTuple{P,Number}
  V::Number
  mat::M
  C3DP(nodes,N0,Nx,Ny,Nz,wgt,V,mat)=new{length(nodes),length(wgt),typeof(mat)}(nodes,N0,Nx,Ny,Nz,wgt,V,mat)
end
#  structural elements
struct Rod
  nodes::Vector{<:Integer}         # node id
  r0::Array{<:Number}          # 
  l0::Number
  A::Number                  # area 
  mat::Materials.Material     # material properties
end
struct Beam
  nodes::Vector{<:Integer}
  r0::Array{<:Number,1}
  L::Number
  t::Number
  w::Number
  lgwx::Array{Tuple{Number,Number},1}
  lgwy::Array{Tuple{Number,Number},1}
  mat::Materials.Material
end
Elems           = Union{Rod, Beam, C2D, C3D, CAS}
CElems{N,P,M}   = Union{C2D{N,P,M}, C3D{N,P,M}, CAS{N,P,M}, C2DP{N,P,M}, C3DP{N,P,M}}
CartEls{N,P,M}  = Union{C2D{N,P,M}, C3D{N,P,M}, C3DP{N,P,M}, C2DP{N,P,M}}
C2DElems{N,P,M} = Union{C2D{N,P,M}, C2DP{N,P,M}}
C3DElems{N,P,M} = Union{C3D{N,P,M}, C3DP{N,P,M}}
CPElems{N,P,M}  = Union{C2DP{N,P,M}, C3DP{N,P,M}}
# parameters retriving functions 
getN(::CElems{N,P,M}) where {N,P,M} = N
getP(::CElems{N,P,M}) where {N,P,M} = P
getM(::CElems{N,P,M}) where {N,P,M} = M
# constructors
function Rod(nodes, p0, A; mat=Materials.Hooke()) 

  r0  = p0[2]-p0[1] 
  l0  = norm(r0)
  Rod(nodes, r0, l0, A, mat)
end
function Beam(nodes, p0, t, w; mat=Materials.Hooke(1, 0.3), Nx = 5, Ny = 3)

  lgwx = lgwt(Nx)
  lgwy = lgwt(Ny, a=-0.5, b=0.5)

  d0  = p0[2]-p0[1] 
  L   = norm(d0)
  r0  = d0/L

  Beam(nodes, r0, L, t, w, lgwx, lgwy, mat)
end
function Line(nodes::Vector{<:Integer}, 
              p0::Vector{Vector{T}} where T<:Number ;
              mat=Materials.Hooke1D())
  (Nx,wgt,A) = begin
    x1,x2   = p0[1],p0[2]
    L       = abs(x2-x1) 
    Nx      = (x2-x1)/L
    ((Nx,),(1.,), L)
  end

  C2D(nodes,Nx,wgt,A,mat) 
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
              p0::Vector{Vector{T}};
              mat=Materials.Hooke()) where T<:Number

  (A,Nx,Ny,wgt) = begin 
    r        = [-1, 1]*0.577350269189626 # √3/3
    N(ξ,η)   = [(1-ξ)*(1-η),(1+ξ)*(1-η),(1+ξ)*(1+η),(1-ξ)*(1+η)]/4

    Nx  = Array{Array{T,1},2}(undef,2,2)
    Ny  = Array{Array{T,1},2}(undef,2,2)
    wgt = Array{T,2}(undef,2,2)
    A   = 0
    for (ii, ξ) in enumerate(r), (jj, η) in enumerate(r)
      N0  = N(adiff.D2([ξ,η])...)
      p   = sum([N0[ii]p0[ii] for ii in 1:4])
      J   = [p[ii].g[jj] for jj in 1:2, ii in 1:2]
      Nxy = J\hcat(adiff.grad.(N0)...)

      Nx[ii,jj]  = Nxy[1,:]
      Ny[ii,jj]  = Nxy[2,:]
      wgt[ii,jj] = detJ(J)
      # wgt[ii,jj] = abs(detJ(J))    
      # having the absolute value in front of det J makes irrelevant whether 
      # the nodes are numbered clockwise or counter-clockwise

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
    V        = detJ(A)/6
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
    V  = detJ(A[1:4,1:4])/6

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
               p0::Vector{Vector{T}};
               mat=Materials.Hooke()) where T<:Number
  (V, Nx, Ny, Nz, wgt) = begin
    r        = [-1, 1]*0.577350269189626 # √3/3
    # N(ξ,η,ζ) = [(1-ξ)*(1-η)*(1+ζ),(1-ξ)*(1-η)*(1-ζ),(1-ξ)*(1+η)*(1-ζ),(1-ξ)*(1+η)*(1+ζ),
    #             (1+ξ)*(1-η)*(1+ζ),(1+ξ)*(1-η)*(1-ζ),(1+ξ)*(1+η)*(1-ζ),(1+ξ)*(1+η)*(1+ζ)]/8
    N(ξ,η,ζ) = [(1-ξ)*(1-η)*(1-ζ),(1+ξ)*(1-η)*(1-ζ),
                (1+ξ)*(1+η)*(1-ζ),(1-ξ)*(1+η)*(1-ζ),
                (1-ξ)*(1-η)*(1+ζ),(1+ξ)*(1-η)*(1+ζ),
                (1+ξ)*(1+η)*(1+ζ),(1-ξ)*(1+η)*(1+ζ)]/8
    Nx  = Array{Array{T,1},3}(undef,2,2,2)
    Ny  = Array{Array{T,1},3}(undef,2,2,2)
    Nz  = Array{Array{T,1},3}(undef,2,2,2)
    wgt = Array{T,3}(undef,2,2,2)
    V   = 0
    for (ii, ξ) in enumerate(r), (jj, η) in enumerate(r), (kk, ζ) in enumerate(r)
      N0   = N(adiff.D2([ξ,η,ζ])...)
      p    = sum([N0[ii]p0[ii] for ii in 1:8])
      J    = [p[ii].g[jj] for jj in 1:3, ii in 1:3]
      Nxyz = J\hcat(adiff.grad.(N0)...)

      Nx[ii,jj,kk]  = Nxyz[1,:]
      Ny[ii,jj,kk]  = Nxyz[2,:]
      Nz[ii,jj,kk]  = Nxyz[3,:]
      # wgt[ii,jj,kk] = abs(detJ(J))
      wgt[ii,jj,kk] = detJ(J)

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
                p0::Vector{Vector{T}};
                mat=Materials.Hooke()) where T<:Number
  (V,N0,Nx,Ny,X0,wgt) = begin
    r        = [-1, 1]*0.577350269189626 # √3/3
    N(ξ,η)   = [(1-ξ)*(1-η),(1+ξ)*(1-η),(1+ξ)*(1+η),(1-ξ)*(1+η)]/4

    N0  = Array{Array{T,1},2}(undef,2,2)
    Nx  = Array{Array{T,1},2}(undef,2,2)
    Ny  = Array{Array{T,1},2}(undef,2,2)
    X0  = Array{T,2}(undef,2,2)
    wgt = Array{T,2}(undef,2,2)
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
      # wgt[ii,jj] = abs(detJ(J))*2π*p[1].v
      wgt[ii,jj] = detJ(J)*2π*p[1].v

      V +=wgt[ii,jj]
    end
    (V,tuple(N0...),tuple(Nx...),tuple(Ny...),tuple(X0...),tuple(wgt...))
  end
  CAS(nodes,N0,Nx,Ny,X0,wgt,V,mat) 
end
# elements with support for phase field
function TriaP(nodes::Vector{<:Integer}, 
              p0::Vector{Vector{T}} where T<:Number ;
              mat=Materials.Hooke())
  (N0,Nx,Ny,wgt,A) = begin
    (x1,x2,x3) = (p0[1][1],p0[2][1],p0[3][1])
    (y1,y2,y3) = (p0[1][2],p0[2][2],p0[3][2])
    Delta      = x1*y2-x2*y1-x1*y3+x3*y1+x2*y3-x3*y2
    Nx         = [y2-y3,y3-y1,y1-y2]./Delta
    Ny         = [x3-x2,x1-x3,x2-x1]./Delta
    A          = abs(Delta)/2
    N0         = [1/3, 1/3, 1/3]
    ((N0,),(Nx,),(Ny,),(A,), A)
  end

  C2DP(nodes,N0,Nx,Ny,wgt,A,mat) 
end
function QuadP(nodes::Vector{<:Integer}, 
              p0::Vector{Vector{T}};
              mat=Materials.Hooke()) where T<:Number

  (A,N0,Nx,Ny,wgt) = begin 
    r        = [-1, 1]*T(√3/3) #0.577350269189626 # √3/3
    N(ξ,η)   = [(1-ξ)*(1-η),(1+ξ)*(1-η),(1+ξ)*(1+η),(1-ξ)*(1+η)]/4

    N0  = Array{Array{T,1},2}(undef,2,2)
    Nx  = Array{Array{T,1},2}(undef,2,2)
    Ny  = Array{Array{T,1},2}(undef,2,2)
    wgt = Array{T,2}(undef,2,2)
    A   = 0
    for (ii, ξ) in enumerate(r), (jj, η) in enumerate(r)
      Nij = N(adiff.D2([ξ,η])...)
      p   = sum([Nij[ii]p0[ii] for ii in 1:4])
      J   = [p[ii].g[jj] for jj in 1:2, ii in 1:2]
      Nxy = J\hcat(adiff.grad.(Nij)...)

      N0[ii,jj]  = adiff.val.(Nij)
      Nx[ii,jj]  = Nxy[1,:]
      Ny[ii,jj]  = Nxy[2,:]
      wgt[ii,jj] = detJ(J)
      # wgt[ii,jj] = abs(detJ(J))    
      # having the absolute value in front of det J makes irrelevant whether 
      # the nodes are numbered clockwise or counter-clockwise

      A += wgt[ii,jj]
    end
    (A,tuple(N0...),tuple(Nx...),tuple(Ny...),tuple(wgt...))
  end

  C2DP(nodes,N0,Nx,Ny,wgt,A,mat) 
end
function Hex08P(nodes::Vector{<:Integer}, 
               p0::Vector{Vector{T}};
               mat=Materials.Hooke()) where T<:Number
  (V, N0, Nx, Ny, Nz, wgt) = begin
    r        = [-1, 1]*0.577350269189626 # √3/3
    N(ξ,η,ζ) = [(1-ξ)*(1-η)*(1-ζ),(1+ξ)*(1-η)*(1-ζ),
                (1+ξ)*(1+η)*(1-ζ),(1-ξ)*(1+η)*(1-ζ),
                (1-ξ)*(1-η)*(1+ζ),(1+ξ)*(1-η)*(1+ζ),
                (1+ξ)*(1+η)*(1+ζ),(1-ξ)*(1+η)*(1+ζ)]/8

    N0  = Array{Array{T,1},3}(undef,2,2,2)
    Nx  = Array{Array{T,1},3}(undef,2,2,2)
    Ny  = Array{Array{T,1},3}(undef,2,2,2)
    Nz  = Array{Array{T,1},3}(undef,2,2,2)
    wgt = Array{T,3}(undef,2,2,2)
    V   = 0

    for (ii, ξ) in enumerate(r), (jj, η) in enumerate(r), (kk, ζ) in enumerate(r)
      Nij  = N(adiff.D2([ξ,η,ζ])...)
      p    = sum([Nij[ii]p0[ii] for ii in 1:8])
      J    = [p[ii].g[jj] for jj in 1:3, ii in 1:3]
      Nxyz = J\hcat(adiff.grad.(Nij)...)

      N0[ii,jj,kk]  = adiff.val.(Nij)
      Nx[ii,jj,kk]  = Nxyz[1,:]
      Ny[ii,jj,kk]  = Nxyz[2,:]
      Nz[ii,jj,kk]  = Nxyz[3,:]
      wgt[ii,jj,kk] = detJ(J)
      # wgt[ii,jj,kk] = abs(detJ(J))
      # having the absolute value in front of det J makes irrelevant whether 
      # the nodes are numbered clockwise or counter-clockwise

      V +=wgt[ii,jj,kk]
    end
    (V,tuple(N0...),tuple(Nx...),tuple(Ny...),tuple(Nz...),tuple(wgt...))
  end
  C3DP(nodes,N0,Nx,Ny,Nz,wgt,V,mat) 
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
function getϕ(elem::CElems{N,P,M} where {N,M}, u::Array{U,2})  where {U, P}
  ϕ = zero(U)
  F = getF(elem,u)
  for ii=1:P
    ϕ += elem.wgt[ii]getϕ(F[ii],elem.mat)
  end 
  ϕ
end
#=
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
=#
# methods for evaluating def. gradient
function getF(elem::C3DElems{N,P,M} where {N,M}, u::Array{D}) where {P,D}
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
function getF(elem::C3DElems, u::Array{N} where N, ii::Int64)
  Nx, Ny, Nz = elem.Nx[ii], elem.Ny[ii], elem.Nz[ii]
  u0, v0, w0 = u[1:3:end],  u[2:3:end],  u[3:3:end]
  
  [Nx⋅u0 Ny⋅u0 Nz⋅u0;
   Nx⋅v0 Ny⋅v0 Nz⋅v0;
   Nx⋅w0 Ny⋅w0 Nz⋅w0 ] + I
end
function getF(elem::C2DElems{N,P,M} where {N,M}, u::Array{D}) where {P,D}
  u0, v0 = u[1:2:end],  u[2:2:end]
  F = fill(Array{D,2}(undef,2,2), P)
  @inbounds for ii = 1:P
    Nx, Ny = elem.Nx[ii], elem.Ny[ii]
    F[ii]  = [Nx⋅u0 Ny⋅u0;
              Nx⋅v0 Ny⋅v0] + I
  end
  F
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
getV(elem,u)     = sum([elem.wgt[ii]detJ(elem,u,ii) for ii in 1:length(elem.wgt)])
detJ(elem,u,ii)  = detJ(getF(elem, u, ii))
detJ(elem,u)     = getV(elem,u)/elem.V
getI3(elem,u,ii) = detJ(getF(elem, u, ii))^2
getI3(elem,u)    = sum([elem.wgt[ii]getI3(elem,u,ii) for ii in 1:length(elem.wgt)])/elem.V
#
function getinfo(elem::Elems, u::Array{<:Number,2}; info=:detF)
  M = length(elem.Nx)
  F = sum(getF(elem, u))/M
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
