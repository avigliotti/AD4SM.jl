# elasticelements.jl
#
# constructors
#
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
              GP=((T(-0.577350269189626), one(T)), 
                  (T(0.577350269189626), one(T))), # √3/3
              mat=Materials.Hooke()) where T<:Number

  # r        = [-1, 1]*0.577350269189626 # √3/3
  N(ξ,η)   = [(1-ξ)*(1-η),(1+ξ)*(1-η),(1+ξ)*(1+η),(1-ξ)*(1+η)]/4

  nGP = length(GP)
  Nx  = Array{Array{T,1},2}(undef,nGP,nGP)
  Ny  = Array{Array{T,1},2}(undef,nGP,nGP)
  wgt = Array{T,2}(undef,nGP,nGP)
  A   = 0
  for (ii, (ξ,wξ)) in enumerate(GP),
    (jj, (η,wη)) in enumerate(GP) 
    N0  = N(adiff.D1([ξ,η])...)
    p   = sum(N0[ii]p0[ii] for ii in 1:4)
    J   = [p[ii].g[jj] for jj in 1:2, ii in 1:2]
    Nxy = J\hcat(adiff.grad.(N0)...)

    Nx[ii,jj]  = Nxy[1,:]
    Ny[ii,jj]  = Nxy[2,:]
    F          = SMatrix{2,2}(J)
    wgt[ii,jj] = detJ(F)*wξ*wη

    A += wgt[ii,jj]
  end

  C2D(nodes,tuple(Nx...),tuple(Ny...),tuple(wgt...),A,mat) 
end
QuadR(nodes,p0;mat=Materials.Hooke()) = Quad(nodes,p0,mat=mat,GP=((0.0,1.0),))
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
  C3D(nodes,Nx,Ny,Nz,(V,),V,mat) 
end
function Tet10(nodes::Vector{<:Integer}, 
               p0::Vector{Vector{T}} where T<:Number;
               mat=Materials.Hooke())

  (V, Nx, Ny, Nz) = begin

    La = 0.585410196624968
    Lb = 0.138196601125010
    ξ  = [La, Lb, Lb, Lb] 
    A  = hcat([[1,p[1],p[2],p[3],p[1]^2,p[2]^2,p[3]^2,p[2]p[3],p[1]p[3],p[1]p[2]] 
               for p in p0]...)
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
               GP=((T(-0.577350269189626), one(T)), (T(0.577350269189626), one(T))), # √3/3
               mat=Materials.Hooke()) where T<:Number

  N(ξ,η,ζ) = [(1-ξ)*(1-η)*(1-ζ),(1+ξ)*(1-η)*(1-ζ),
              (1+ξ)*(1+η)*(1-ζ),(1-ξ)*(1+η)*(1-ζ),
              (1-ξ)*(1-η)*(1+ζ),(1+ξ)*(1-η)*(1+ζ),
              (1+ξ)*(1+η)*(1+ζ),(1-ξ)*(1+η)*(1+ζ)]/8
  nGP = length(GP)
  Nx  = Array{Array{T,1},3}(undef,nGP,nGP,nGP)
  Ny  = Array{Array{T,1},3}(undef,nGP,nGP,nGP)
  Nz  = Array{Array{T,1},3}(undef,nGP,nGP,nGP)
  wgt = Array{T,3}(undef,nGP,nGP,nGP)
  V   = 0
  for (ii, (ξ,wξ)) in enumerate(GP),
    (jj, (η,wη)) in enumerate(GP), 
    (kk, (ζ,wζ)) in enumerate(GP)

    N0   = N(adiff.D1([ξ,η,ζ])...)
    p    = sum([N0[ii]p0[ii] for ii in 1:8])
    J    = [p[ii].g[jj] for jj in 1:3, ii in 1:3]
    Nxyz = J\hcat(adiff.grad.(N0)...)

    Nx[ii,jj,kk]  = Nxyz[1,:]
    Ny[ii,jj,kk]  = Nxyz[2,:]
    Nz[ii,jj,kk]  = Nxyz[3,:]
    F             = SMatrix{3,3}(J)
    wgt[ii,jj,kk] = detJ(F)*wξ*wη*wζ

    V +=wgt[ii,jj,kk]
  end
  C3D(nodes,tuple(Nx...),tuple(Ny...),tuple(Nz...),tuple(wgt...),V,mat) 
end
function Wdg06(nodes::Vector{<:Integer}, 
               p0::Vector{Vector{T}} where T<:Number;
               mat=Materials.Hooke())
  N(ξ,η,ζ) = [(1-ζ)*(1-ξ-η), (1-ζ)*ξ, (1-ζ)*η,
              (1+ζ)*(1-ξ-η), (1+ζ)*ξ, (1+ζ)*η]/2

  GPs =  [([2/3,1/6,√3/3], 1/3), ([2/3,1/6,-√3/3], 1/3),
          ([1/6,2/3,√3/3], 1/3), ([1/6,2/3,-√3/3], 1/3),
          ([1/6,1/6,√3/3], 1/3), ([1/6,1/6,-√3/3], 1/3)]
  nGP = length(GPs)

  Nx  = Array{Array{T,1},1}(undef,nGP)
  Ny  = Array{Array{T,1},1}(undef,nGP)
  Nz  = Array{Array{T,1},1}(undef,nGP)
  wgt = Array{T,1}(undef,nGP)
  Vol = 0

  for (ii, (Pii, wii)) in enumerate(GPs)
    Nii      = N(adiff.D1(Pii)...)
    # p        = sum([N0[ii]p0[ii] for ii in 1:6])
    p        = transpose(Nii)*p0
    J        = [p[ii].g[jj] for jj in 1:3, ii in 1:3]
    Nxyz     = J\hcat(adiff.grad.(Nii)...)
    wgt[ii]  = det(J)*wii
    Vol     += wgt[ii]

    Nx[ii]  = Nxyz[1,:]
    Ny[ii]  = Nxyz[2,:]
    Nz[ii]  = Nxyz[3,:]
  end

  C3DP(nodes,tuple(Nx...),tuple(Ny...),
       tuple(Nz...),tuple(wgt...),Vol,mat) 
end
Hex08R(nodes, p0;mat=Materials.Hooke()) = Hex08(nodes, p0, mat=mat, GP=((0.0,1.0),))
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
      Nij = N(adiff.D1([ξ,η])...)
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
#
# elastic energy evaluation functions for elements
#=
function getϕ(elem::Rod,  u::Matrix{<:Number})
  l   = norm(elem.r0+u[:,2]-u[:,1])
  F11 = l/elem.l0
  elem.A*elem.l0*getϕ(F11, elem.mat)    
end
function getϕ(elem::Beam, u::Matrix{<:Number})

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
=#
# General CElem energy integrator (works with CElem/CPElem)
function getϕ(elem::CElem{P}, u::Array{D}) where {P,D}
  ϕ = zero(D)
  for ii=1:P
    Fii = getF(elem, u, ii)
    ϕ  += elem.wgt[ii] * getϕ(Fii, elem.mat)
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
function getϕ(elem::C3D{P}, u0::Array{D}) where {P,D<:adiff.D2}

  u0 = adiff.D1.(u0)
  ϕ  = zero(D) 
  @inline for ii=1:P
    F    = getF(elem, u0, ii)
    valF = adiff.val.(F)
    δϕ   = getϕ(adiff.D2(valF), elem.mat)
    ϕ   += elem.wgt[ii]δϕ×F
  end
  ϕ
end
#
#
# functions for evaluating the residual and the tangent stiffness matrix over
# an array of elements
#
function makeϕrKt(elems::Array{<:CElem{D,P,M,S,L}} where {P,M,S}, u::Array{T}) where {D,L,T}

  nElems = length(elems)
  N      = D*L # length(u[:,elems[1].nodes])
  M      = (N+1)N÷2

  Φ = Vector{adiff.D2{N,M,T}}(undef, nElems)
  Threads.@threads for ii=1:nElems
    Φ[ii] = getϕ(elems[ii], adiff.D2(u[:,elems[ii].nodes]))
  end

  makeϕrKt(Φ, elems, u)
end
#
#
# function getδϕ(elem::C3D{P}, u0::Array{T})  where {P,T}  
# evaluates the strain energy density as a dual D2 number 
#
getδϕ(elem::AbstractElement, u::Array{<:Number}) = getϕ(elem, adiff.D2(u))

function getδϕ(elem::C3D{P}, u0::Array{T})  where {P,T}
  #
  # This implementation computes the D2 dual for the element internal energy.
  # It builds the sensitivity of F with respect to nodal DOFs (δF) using the
  # element's ∇N data (assumed stored as ∇N[1][ii], ∇N[2][ii], ∇N[3][ii])
  # where each ∇N[*][ii] is a static-vector (SVector) or array with length nNodes.
  #

  u, v, w = u0[1:3:end], u0[2:3:end], u0[3:3:end]
  nnode   = length(u)             # number of nodes
  Ndofs   = 3 * nnode             # total number of nodal displacement DOFs
  wgt     = elem.wgt
  val     = zero(T)
  grad    = zeros(T, Ndofs)
  hess    = zeros(T, (Ndofs+1)*Ndofs ÷ 2)  # triangular storage
  δF      = zeros(T, Ndofs, 9)

  @inbounds for ii=1:P
    # get shape-function derivative arrays at Gauss point ii
    Nx = elem.∇N[1][ii]
    Ny = elem.∇N[2][ii]
    Nz = elem.∇N[3][ii]

    # Build δF: derivative of each F component wrt each nodal DOF
    # Column ordering of F components: (F11,F12,F13,F21,F22,F23,F31,F32,F33)
    # For node a, nodal DOFs indices (ux,uy,uz) = (3*(a-1)+1,...+3)
    for a=1:nnode
      idx = 3*(a-1)
      nx = Nx[a]
      ny = Ny[a]
      nz = Nz[a]

      δF[idx+1, 1] = nx   # dF11/d(ux_a)
      δF[idx+2, 2] = nx   # dF12/d(uy_a) ??? (kept same mapping as original code)
      δF[idx+3, 3] = nx   # dF13/d(uz_a)

      δF[idx+1, 4] = ny   # dF21/d(ux_a)
      δF[idx+2, 5] = ny   # dF22/d(uy_a)
      δF[idx+3, 6] = ny   # dF23/d(uz_a)

      δF[idx+1, 7] = nz   # dF31/d(ux_a)
      δF[idx+2, 8] = nz   # dF32/d(uy_a)
      δF[idx+3, 9] = nz   # dF33/d(uz_a)
    end

    # Evaluate F at this Gauss point
    F = SMatrix{3,3,T}(
            (Nx⋅u + 1) , (Nx⋅v) , (Nx⋅w),
            (Ny⋅u)      , (Ny⋅v + 1) , (Ny⋅w),
            (Nz⋅u)      , (Nz⋅v) , (Nz⋅w + 1)
         )

    # Evaluate constitutive D2 energy for F (material returns adiff.D2)
    ϕ = getϕ(adiff.D2(adiff.val.(F)), elem.mat)::adiff.D2{9, 45, T}

    # accumulate energy, gradient and (triangular) Hessian using δF mapping
    val += wgt[ii] * ϕ.v

    # Gradient: grad[i] += wgt * sum_j ϕ.g[j] * δF[i,j]
    @inbounds for j = 1:9
      coeff = wgt[ii] * ϕ.g[j]
      for i1 = 1:Ndofs
        grad[i1] += coeff * δF[i1, j]
      end
    end

    # Hessian: hess[index(i1,i2)] += wgt * sum_{j,k} ϕ.h[j,k] * δF[i1,j]*δF[i2,k]
    @inbounds for j = 1:9
      for k = 1:j
        hjk = wgt[ii] * ϕ.h[j,k]
        if hjk == zero(hjk)
          continue
        end
        for i1 = 1:Ndofs
          c1 = δF[i1, j]
          if c1 == zero(c1)
            continue
          end
          for i2 = 1:i1
            # triangular index mapping (i2 <= i1)
            idx_tri = (i1-1)*i1 ÷ 2 + i2
            hess[idx_tri] += hjk * c1 * δF[i2, k]
          end
        end
      end
    end
  end

  adiff.D2(val, adiff.Grad(grad), adiff.Grad(hess))
end

function getδϕ(elems::Vector{<:CElem}, u::Array{T,2}) where T
  nElems = length(elems)
  N      = length(u[:,elems[1].nodes])
  M      = (N+1)N÷2

  Φ = Vector{adiff.D2{N,M,T}}(undef, nElems)
  Threads.@threads for ii=1:nElems
    Φ[ii] = getδϕ(elems[ii], u[:,elems[ii].nodes])
  end
  Φ
end

