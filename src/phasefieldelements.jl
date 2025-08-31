
export makeϕrKt, makeϕrKt_d

#
# elements with support for phase field
#
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
               GP=((-0.577350269189626, 1.0), (0.577350269189626, 1.0)), # √3/3
               mat=Materials.Hooke()) where T<:Number
  #r        = [-1, 1]*T(√3/3) #0.577350269189626 # √3/3
  N(ξ,η)   = [(1-ξ)*(1-η),(1+ξ)*(1-η),(1+ξ)*(1+η),(1-ξ)*(1+η)]/4

  nGP = length(GP)
  N0  = Array{Array{T,1},2}(undef,nGP,nGP)
  Nx  = Array{Array{T,1},2}(undef,nGP,nGP)
  Ny  = Array{Array{T,1},2}(undef,nGP,nGP)
  wgt = Array{T,2}(undef,nGP,nGP)
  A   = 0
  for (ii, (ξ,wξ)) in enumerate(GP),
    (jj, (η,wη)) in enumerate(GP) 
    Nij = N(adiff.D1([ξ,η])...)
    p   = sum([Nij[ii]p0[ii] for ii in 1:4])
    J   = [p[ii].g[jj] for jj in 1:2, ii in 1:2]
    Nxy = J\hcat(adiff.grad.(Nij)...)

    N0[ii,jj]  = adiff.val.(Nij)
    Nx[ii,jj]  = Nxy[1,:]
    Ny[ii,jj]  = Nxy[2,:]
    wgt[ii,jj] = detJ(J)*wξ*wη

    A += wgt[ii,jj]
  end

  C2DP(nodes,tuple(N0...),tuple(Nx...),tuple(Ny...),tuple(wgt...),A,mat) 
end
QuadPR(nodes, p0;mat=Materials.Hooke()) = QuadP(nodes, p0, mat=mat, GP=((0.0,1.0),))
function Hex08P(nodes::Vector{<:Integer}, 
                p0::Vector{Vector{T}};
                GP=((T(-0.577350269189626), one(T)), 
                    (T(0.577350269189626), one(T))), # √3/3
                mat=Materials.Hooke()) where T<:Number

  N(ξ,η,ζ) = [(1-ξ)*(1-η)*(1-ζ),(1+ξ)*(1-η)*(1-ζ),
              (1+ξ)*(1+η)*(1-ζ),(1-ξ)*(1+η)*(1-ζ),
              (1-ξ)*(1-η)*(1+ζ),(1+ξ)*(1-η)*(1+ζ),
              (1+ξ)*(1+η)*(1+ζ),(1-ξ)*(1+η)*(1+ζ)]/8
  nGP = length(GP)
  N0  = Array{Array{T,1},3}(undef,nGP,nGP,nGP)
  Nx  = Array{Array{T,1},3}(undef,nGP,nGP,nGP)
  Ny  = Array{Array{T,1},3}(undef,nGP,nGP,nGP)
  Nz  = Array{Array{T,1},3}(undef,nGP,nGP,nGP)
  wgt = Array{T,3}(undef,nGP,nGP,nGP)
  V   = 0

  for (ii, (ξ,wξ)) in enumerate(GP),
    (jj, (η,wη)) in enumerate(GP), 
    (kk, (ζ,wζ)) in enumerate(GP)

    Nijk = N(adiff.D1([ξ,η,ζ])...)
    p    = sum([Nijk[ii]p0[ii] for ii in 1:8])
    J    = [p[ii].g[jj] for jj in 1:3, ii in 1:3]
    Nxyz = J\hcat(adiff.grad.(Nijk)...)

    N0[ii,jj,kk]  = adiff.val.(Nijk)
    Nx[ii,jj,kk]  = Nxyz[1,:]
    Ny[ii,jj,kk]  = Nxyz[2,:]
    Nz[ii,jj,kk]  = Nxyz[3,:]
    wgt[ii,jj,kk] = detJ(J)*wξ*wη*wζ

    V +=wgt[ii,jj,kk]
  end
  C3DP(nodes,tuple(N0...),tuple(Nx...),tuple(Ny...),
                tuple(Nz...),tuple(wgt...),V,mat) 
end
Hex08PR(nodes, p0;mat=Materials.Hooke()) = Hex08P(nodes, p0, mat=mat, GP=((0.0,1.0),))
function Wdg06P(nodes::Vector{<:Integer}, 
                p0::Vector{Vector{T}};
                mat=Materials.Hooke())  where T<:Number
  N(ξ,η,ζ) = [(1-ζ)*(1-ξ-η), (1-ζ)*ξ, (1-ζ)*η,
              (1+ζ)*(1-ξ-η), (1+ζ)*ξ, (1+ζ)*η]/2

  GPs =  [([2/3,1/6,√3/3], 1/3), ([2/3,1/6,-√3/3], 1/3),
          ([1/6,2/3,√3/3], 1/3), ([1/6,2/3,-√3/3], 1/3),
          ([1/6,1/6,√3/3], 1/3), ([1/6,1/6,-√3/3], 1/3)]
  nGP = length(GPs)

  N0  = Array{Array{T,1},1}(undef,nGP)
  Nx  = Array{Array{T,1},1}(undef,nGP)
  Ny  = Array{Array{T,1},1}(undef,nGP)
  Nz  = Array{Array{T,1},1}(undef,nGP)
  wgt = Array{T,1}(undef,nGP)
  Vol = 0

  for (ii, (Pii, wii)) in enumerate(GPs)
    Nii     = N(adiff.D1(Pii)...)
    # p       = sum([N0[ii]p0[ii] for ii in 1:6])
    p       = transpose(Nii)*p0
    J       = [p[ii].g[jj] for jj in 1:3, ii in 1:3]
    Nxyz    = J\hcat(adiff.grad.(Nii)...)
    wgt[ii] = det(J)*wii
    Vol    += wgt[ii]

    N0[ii]  = adiff.val.(Nii)
    Nx[ii]  = Nxyz[1,:]
    Ny[ii]  = Nxyz[2,:]
    Nz[ii]  = Nxyz[3,:]
  end
  C3DP(nodes,tuple(N0...),tuple(Nx...),tuple(Ny...),
       tuple(Nz...),tuple(wgt...),Vol,mat) 
end
function Tet04P(nodes::Vector{<:Integer}, 
                p0::Vector{Vector{T}} where T<:Number;
                mat=Materials.Hooke())
  (V, N0, Nx, Ny, Nz) = begin
    A        = ones(4,4)
    A[1,2:4] = p0[1]
    A[2,2:4] = p0[2]
    A[3,2:4] = p0[3]
    A[4,2:4] = p0[4]
    C        = inv(A)
    V        = det(A)/6
    N0       = [1/4, 1/4, 1/4, 1/4]
    (V,(N0,),(C[2,:],),(C[3,:],),(C[4,:],))
  end
  C3DP(nodes,N0,Nx,Ny,Nz,(V,),V,mat) 
end
#
# functions for phase field
#
function getdand∇d(elem::C3DP, d0::Vector{D}, ii) where D
  N0,Nx,Ny,Nz = elem.N0[ii],elem.Nx[ii],elem.Ny[ii],elem.Nz[ii]
  ∇d = [Nx⋅d0, Ny⋅d0, Nz⋅d0]
  d  = N0⋅d0

  return (d, ∇d)
end
function getdand∇d(elem::Union{C2DP,CAS}, d0::Vector{D}, ii) where D
  N0,Nx,Ny = elem.N0[ii],elem.Nx[ii],elem.Ny[ii]
  ∇d = [Nx⋅d0, Ny⋅d0]
  d  = N0⋅d0

  return (d, ∇d)
end
#
# get the average of d over the element
function getd(elem::CPElems{P}, d0::Vector{T}) where {P,T}
  d       = zero(T)
  for ii=1:P
    d += elem.wgt[ii]*(elem.N0[ii]⋅d0)
  end  
  d/elem.V
end
#
# get free energy density for the elment without history
function getϕ(elem::CPElems{P}, u0::Matrix, d0::Vector) where P

  ϕ = 0
  @inline for ii=1:P
    F    = getF(elem, u0, ii)
    d,∇d = getdand∇d(elem,d0,ii)
    ϕ   += elem.wgt[ii]getϕ(F, d, ∇d, elem.mat)
  end
  ϕ
end
# get free energy density for the elment with history
function getϕ(elem::CPElems{P}, u0::Matrix, d0::Vector, ϕmax::Vector) where P

  wgt = elem.wgt
  ϕ   = 0
  @inline for ii=1:P
    F    = getF(elem, u0, ii)
    d,∇d = getdand∇d(elem,d0,ii)
    ϕii,ϕmax[ii] = getϕ(F, d, ∇d, elem.mat, ϕmax[ii])
    ϕ   += wgt[ii]ϕii
  end
  ϕ
end
#
# calling getϕ with dual numbers on 3D elements
#
# these functions are optimized in case getϕ is called with a dual type for 
# the displacement field trough the use of the × operators for the chain 
# derivative, the other use the standard implementation common for all
# on newer CPU this might disppear
#
# without history
function getϕ(elem::C3DP{P}, u0::Matrix{D}, d0::Vector) where {P,D<:adiff.D2}

  u0 = adiff.D1.(u0)
  ϕ  = zero(D) 
  @inline for ii=1:P
    F    = getF(elem, u0, ii)
    d,∇d = getdand∇d(elem,d0, ii)
    valF = adiff.val.(F)
    δϕ   = getϕ(adiff.D2(valF), d, ∇d, elem.mat)
    ϕ   += elem.wgt[ii]δϕ×F
  end
  ϕ
end
#
# with history
function getϕ(elem::C3DP{P}, u0::Matrix{D}, d0::Vector, ϕmax::Vector) where {P,D<:adiff.D2}

  u0 = adiff.D1.(u0)
  ϕ  = zero(D) 
  @inline for ii=1:P
    F    = getF(elem, u0, ii)
    d,∇d = getdand∇d(elem,d0, ii)
    valF = adiff.val.(F)
    δϕ,ϕmax[ii] = getϕ(adiff.D2(valF), d, ∇d, elem.mat, ϕmax[ii])
    ϕ   += elem.wgt[ii]δϕ×F
  end
  ϕ
end
#
# structural  elements
function getϕ(elem::Rod{<:PhaseField{<:Hooke,:ATn}}, u::Matrix{U}, d0::Vector{D}) where {U<:Number,D<:Number}
  mat     = elem.mat
  l0,Gc,n = mat.l0,mat.Gc,mat.n

  l   = norm(elem.r0+u[:,2]-u[:,1])
  ϵ11 = l/elem.l0-1
  ϕ   = mat.mat.E/2*ϵ11^2

  d   = (d0[2]+d0[1])/2
  ∇d  = (d0[2]-d0[1])/elem.l0
  γ   = d^n + l0^2*∇d^2

  Vol = elem.A*elem.l0
  Vol*((1-d)^2*ϕ + Gc/2l0*γ)
end
function getϕ(elem::Rod{<:PhaseField{M,:ATn} where M}, u::Matrix{<:Number}, ϕmax::U, d0::Vector{D}) where {U<:Number,D<:Number}
  mat     = elem.mat
  l0,Gc,n = mat.l0,mat.Gc,mat.n

  l   = norm(elem.r0+u[:,2]-u[:,1])
  ϵ11 = l/elem.l0-1
  ϕ   = mat.mat.E/2*ϵ11^2

  d   = (d0[2]+d0[1])/2
  ∇d  = (d0[2]-d0[1])/elem.l0
  γ   = d^n + l0^2*∇d^2
  ϕ   = max(ϕ, ϕmax)

  Vol = elem.A*elem.l0
  Vol*((1-d)^2*ϕ + Gc/2l0*γ), ϕ
end
# 
# functions for array of elements
# 
function makeϕrKt(elems::Vector{<:CPElems}, u::Matrix{T}, d::Matrix{T}) where T
  nElems = length(elems)
  N      = length(u[:,elems[1].nodes])
  M      = (N+1)N÷2

  Φ = Vector{adiff.D2{N,M,T}}(undef, nElems)
  Threads.@threads for ii=1:nElems
    Φ[ii] = getϕ(elems[ii], adiff.D2(u[:,elems[ii].nodes]), d[elems[ii].nodes])
  end

  makeϕrKt(Φ, elems, u)
end
function makeϕrKt_d(elems::Vector{<:CPElems}, u::Matrix{T}, d::Matrix{T}) where T

  nElems = length(elems)
  N      = length(d[elems[1].nodes])
  M      = (N+1)N÷2

  Φ = Vector{adiff.D2{N,M,T}}(undef, nElems)
  Threads.@threads for ii=1:nElems
    nodes = elems[ii].nodes
    Φ[ii] = getϕ(elems[ii], u[:,elems[ii].nodes], adiff.D2(d[elems[ii].nodes]))
  end
  makeϕrKt(Φ, elems, d)
end
# with history
function makeϕrKt_d(elems::Vector{<:CPElems}, u::Matrix{T}, d::Matrix{T}, ϕmax::Vector) where T

  nElems = length(elems)
  N      = length(d[elems[1].nodes])
  M      = (N+1)N÷2

  Φ = Vector{adiff.D2{N,M,T}}(undef, nElems)
  Threads.@threads for ii=1:nElems
    nodes = elems[ii].nodes
    Φ[ii] = getϕ(elems[ii], u[:,elems[ii].nodes], adiff.D2(d[elems[ii].nodes]), ϕmax[ii])
  end
  makeϕrKt(Φ, elems, d)
end
#
#
function getδϕu(elem::C3DP{P,<:PhaseField}, u0::Matrix{T}, d0::Vector{T})  where {P,T}

  u, v, w = u0[1:3:end], u0[2:3:end], u0[3:3:end]
  N       = lastindex(u0)  
  wgt     = elem.wgt
  val     = zero(T)
  grad    = zeros(T,N)
  hess    = zeros(T,(N+1)N÷2)
  δF      = zeros(T,N,9)

  for ii=1:P
    N0,Nx,Ny,Nz = elem.N0[ii],elem.Nx[ii],elem.Ny[ii],elem.Nz[ii]
    δF[1:3:N,1] = δF[2:3:N,2] = δF[3:3:N,3] = Nx
    δF[1:3:N,4] = δF[2:3:N,5] = δF[3:3:N,6] = Ny
    δF[1:3:N,7] = δF[2:3:N,8] = δF[3:3:N,9] = Nz

    F    = [Nx⋅u Ny⋅u Nz⋅u;
            Nx⋅v Ny⋅v Nz⋅v;
            Nx⋅w Ny⋅w Nz⋅w ] + I
    d    = N0⋅d0
    ∇d   = [Nx⋅d0, Ny⋅d0, Nz⋅d0]
    ϕ    = getϕ(adiff.D2(F), d, ∇d, elem.mat)::adiff.D2{9, 45, T}
    val += wgt[ii]ϕ.v
    for jj=1:9,i1=1:N
      grad[i1] += wgt[ii]*ϕ.g[jj]*δF[i1,jj]
      for kk=1:9,i2=1:i1
        hess[(i1-1)i1÷2+i2] += wgt[ii]*ϕ.h[jj,kk]*δF[i1,jj]*δF[i2,kk]
      end   
    end
  end

  adiff.D2(val, adiff.Grad(grad), adiff.Grad(hess))
end
#
getδϕd(elem::C3DElems{P,<:PhaseField}, u0::Matrix, d0::Vector) where P = getϕ(elem, u0, adiff.D2(d0))
getδϕd(elem::C2DElems{P,<:PhaseField}, u0::Matrix, d0::Vector) where P = getϕ(elem,u0,adiff.D2(d0))
getδϕd(elem::Rod{<:PhaseField}, u0::Matrix, d0::Vector)                = getϕ(elem,u0,adiff.D2(d0))
getδϕu(elem::Rod{<:PhaseField}, u0::Matrix, d0::Vector)               = getϕ(elem,adiff.D2(u0),d0)
#
function getd(elem::CElems{P}, d0::Vector{T}) where {P,T}
  d       = zero(T)
  for ii=1:P
    d += elem.wgt[ii]*(elem.N0[ii]⋅d0)
  end  
  d/elem.V
end
# getVd
function getVd(elem::CPElems{P}, d0::Vector{T}) where {T, P}
  Vd = zero(T)
  for ii=1:P
    Vd += elem.wgt[ii]elem.N0[ii]⋅d0
  end
  Vd
end
function getVd(elems::Vector{<:CPElems}, d::Matrix{T}) where T
  Vd = zero(T)
  for elem in elems
    Vd += getVd(elem, d[elem.nodes])
  end
  Vd
end

