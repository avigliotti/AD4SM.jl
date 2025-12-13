export makeϕrKt
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
#
# continuous elements
#
function Line(nodes::Vector{<:Integer}, 
              p0::Vector{Vector{T}} where T<:Number ;
              mat=Materials.Hooke1D())
  (Nx,wgt,A) = begin
    x1,x2   = p0[1],p0[2]
    L       = abs(x2-x1) 
    Nx      = (x2-x1)/L
    ((Nx,),(1.,), L)
  end

  C2DE(nodes,Nx,wgt,A,mat) 
end
function Tria03(nodes::Vector{<:Integer}, 
                p0::Vector{Vector{T}}; 
                mat::M=Materials.Hooke(),
                bReduced::Bool=false) where {T<:Number, M<:Material}

  # Shape functions for 3-node triangle (Linear)
  N(ξ,η) = SVector(1.0-ξ-η, ξ, η)

  # Integration Rule: 1-point Gauss (Centroid)
  # Weight = 0.5 (Area of reference triangle with vertices (0,0), (1,0), (0,1))
  GPs = ((SVector(1/3, 1/3), 0.5),)
  
  nGP = length(GPs)
  nN  = length(nodes)

  Nx  = Vector{Vector{T}}(undef, nGP)
  Ny  = Vector{Vector{T}}(undef, nGP)
  wgt = Vector{T}(undef, nGP)
  V   = zero(T)

  @inbounds for (ii, (coords, wii)) in enumerate(GPs)
    # Evaluate shape functions and their derivatives (Dual numbers)
    N_dual = N(adiff.D1(coords)...)
    
    # Interpolate physical coordinates
    # p[1] is x, p[2] is y
    p = sum(N_dual[a] * p0[a] for a in 1:nN)

    # Transposed Jacobian Jᵀ (2x2)
    # J_ij = d(x_i)/d(ξ_j)
    # p[ii].g[jj] gets derivative of coordinate ii w.r.t param jj
    Jᵀ = SMatrix{2,2,T}(p[k].g[j] for j in 1:2, k in 1:2)

    # Map gradients to physical space: ∇_x N = J⁻ᵀ ∇_ξ N
    # grads_phys becomes a 2x3 matrix (2 rows: d/dx, d/dy; 3 cols: nodes)
    grads_phys = Jᵀ \ hcat(adiff.grad.(N_dual)...)

    Nx[ii]  = grads_phys[1, :]
    Ny[ii]  = grads_phys[2, :]
    
    # Determinant of 2x2 Jacobian
    wgt[ii] = det(Jᵀ) * wii 
    V += wgt[ii]
  end

  ∇N = tuple(tuple(Nx...), tuple(Ny...),)
  
  # C2D implies a 2D element struct
  C2D{nGP,M,T,nN,1}(nodes, ∇N, tuple(wgt...), V, mat) 
end
function Quad04(nodes::Vector{<:Integer}, 
                p0::Vector{Vector{T}}; 
                mat::M=Materials.Hooke(), 
                bReduced::Bool=false) where {T<:Number, M<:Material}

  # Shape functions for 4-node Quadrilateral (Bilinear)
  # Standard FEM order: (-1,-1), (1,-1), (1,1), (-1,1)
  function N(ξ, η)
    one_min_xi  = 1.0 - ξ
    one_pl_xi   = 1.0 + ξ
    one_min_eta = 1.0 - η
    one_pl_eta  = 1.0 + η

    SVector(0.25 * one_min_xi * one_min_eta, # Node 1
            0.25 * one_pl_xi  * one_min_eta, # Node 2
            0.25 * one_pl_xi  * one_pl_eta,  # Node 3
            0.25 * one_min_xi * one_pl_eta,) # Node 4
  end

  # Gaussian Quadrature
  GPs = if bReduced
    # Reduced Integration: 1-point rule (Centroid)
    # Weight = 4.0 (Area of reference domain [-1,1]x[-1,1])
    ((SVector(0.0, 0.0), 4.0),)
  else
    # Full Integration: 2x2 Gauss-Legendre Rule (Standard)
    # Points at ±1/√3, Weight = 1.0 for each dim (1*1=1)
    g = 0.577350269189626
    w = 1.0
    ((SVector(-g, -g), w),
     (SVector( g, -g), w),
     (SVector( g,  g), w),
     (SVector(-g,  g), w),)
  end

  nGP = length(GPs)
  nN  = length(nodes)

  # Storage arrays (Flattened to Vector{Vector} to match C2D struct)
  Nx  = Vector{Vector{T}}(undef, nGP)
  Ny  = Vector{Vector{T}}(undef, nGP)
  wgt = Vector{T}(undef, nGP)
  V   = zero(T)

  @inbounds for (ii, (coords, wii)) in enumerate(GPs)
    # Evaluate shape functions and derivatives (Dual numbers)
    N_dual = N(adiff.D1(coords)...)
    
    # Interpolate physical coordinates
    # Sum over nodes k=1:4 (using 'k' to avoid index collision with 'ii')
    p = sum(N_dual[k] * p0[k] for k in 1:nN)

    # Transposed Jacobian Jᵀ (2x2)
    # J_ij = d(x_i)/d(ξ_j)
    Jᵀ = SMatrix{2,2,T}(p[k].g[j] for j in 1:2, k in 1:2)

    # Map gradients to physical space: ∇_x N = J⁻ᵀ ∇_ξ N
    # grads_phys is 2x4 matrix (rows: dx, dy; cols: nodes)
    grads_phys = Jᵀ \ hcat(adiff.grad.(N_dual)...)

    Nx[ii]  = grads_phys[1, :]
    Ny[ii]  = grads_phys[2, :]
    
    # Determinant of 2x2 Jacobian times Gauss weight
    wgt[ii] = det(Jᵀ) * wii 
    V += wgt[ii]
  end

  # Construct Element
  ∇N = tuple(tuple(Nx...), tuple(Ny...),)
  C2D{nGP,M,T,nN,1}(nodes, ∇N, tuple(wgt...), V, mat) 
end
function Tet04(nodes::Vector{<:Integer},                # nodal indexes
                p0::Vector{Vector{T}};                  # nodal coordinates 
                mat::M=Materials.Hooke(),
                bReduced::Bool=false) where {T<:Number, M<:Material}

  N(ξ,η,ζ) = SVector(1.0-ξ-η-ζ, ξ, η, ζ)

  GPs = if bReduced
    ((SVector(1/4, 1/4, 1/4), 1/6),)
  else 
    a = 0.5854101966249685
    b = 0.1381966011250105
    w = 1/24 
    ((SVector(a,b,b),w),
     (SVector(b,a,b),w),
     (SVector(b,b,a),w),
     (SVector(b,b,b),w)   )
  end

  nGP = length(GPs )
  nN  = length(nodes)

  Nx  = Vector{Vector{T}}(undef, nGP)
  Ny  = Vector{Vector{T}}(undef, nGP)
  Nz  = Vector{Vector{T}}(undef, nGP)
  wgt = Vector{T}(undef, nGP)
  V   = zero(T)

  @inbounds for (ii, (coords, wii)) in enumerate(GPs)
    N_dual = N(adiff.D1(coords)...)
    p = sum(N_dual[a]p0[a] for a in 1:nN)

    # transposed Jacobian
    Jᵀ = SMatrix{3,3,T}(p[ii].g[jj] for jj in 1:3, ii in 1:3)
    grads_phys = Jᵀ \ hcat(adiff.grad.(N_dual)...)

    Nx[ii]  = grads_phys[1, :]
    Ny[ii]  = grads_phys[2, :]
    Nz[ii]  = grads_phys[3, :]
    wgt[ii] = detJ(Jᵀ) * wii
    V += wgt[ii]
  end

  ∇N = tuple(tuple(Nx...), tuple(Ny...), tuple(Nz...),)
  C3D{nGP,M,T,nN,1}(nodes, ∇N, tuple(wgt...), V, mat) 
end
function Hex08(nodes::Vector{<:Integer}, 
               p0::Vector{Vector{T}};
               bReduced::Bool=false,
               mat=Materials.Hooke()) where T<:Number

  N(ξ,η,ζ) = [(1-ξ)*(1-η)*(1-ζ),(1+ξ)*(1-η)*(1-ζ),
              (1+ξ)*(1+η)*(1-ζ),(1-ξ)*(1+η)*(1-ζ),
              (1-ξ)*(1-η)*(1+ζ),(1+ξ)*(1-η)*(1+ζ),
              (1+ξ)*(1+η)*(1+ζ),(1-ξ)*(1+η)*(1+ζ)]/8

  GP=((T(-0.577350269189626), one(T)), 
      (T(0.577350269189626), one(T))) # √3/3
  nGP = length(GP)

  Nx  = Array{Array{T,1},3}(undef,nGP,nGP,nGP)
  Ny  = Array{Array{T,1},3}(undef,nGP,nGP,nGP)
  Nz  = Array{Array{T,1},3}(undef,nGP,nGP,nGP)
  wgt = Array{T,3}(undef,nGP,nGP,nGP)
  V   = zero(T) 
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
  C3DE(nodes,tuple(Nx...),tuple(Ny...),tuple(Nz...),tuple(wgt...),V,mat) 
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
# 
# elastic energy evaluation functions for elements
# General CEElem energy integrator (works with CEElem/CPElem)
function getϕ(elem::CEElem{<:Any,P}, u::Array{D}) where {P,D}
  ϕ = zero(D)
  for ii=1:P
    Fii = getF(elem, u, ii)
    ϕ  += elem.wgt[ii]getϕ(Fii, elem.mat)
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
function getϕ(elem::C3DE{P}, u0::Array{D}) where {P,D<:adiff.D2}

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
function makeϕrKt(elems::AbstractVector{<:CEElem}, u::AbstractMatrix{T}) where T
  nElems = length(elems)
  @assert nElems > 0 "makeϕrKt: `elems` is empty"  
  #=
  et = eltype(elems)
  D  = et.parameters[1]
  L  = et.parameters[5]
  N  = D*L 
  =# 
  N  = length(u[:,elems[1].nodes])
  M  = (N+1)N÷2

  Φ = Vector{adiff.D2{N,M,T}}(undef, nElems)
  Threads.@threads for ii=1:nElems
    Φ[ii] = getϕ(elems[ii], adiff.D2(u[:,elems[ii].nodes]))
  end

  makeϕrKt(Φ, elems, u)
end
#
#
# function getδϕ(elem::C3DE{P}, u0::Array{T})  where {P,T}  
# evaluates the strain energy density as a dual D2 number 
#
getδϕ(elem::AbstractElement, u::Array{<:Number}) = getϕ(elem, adiff.D2(u))

function getδϕ(elem::C3DE{P}, u0::Array{T})  where {P,T}
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

function getδϕ(elems::Vector{<:CEElem}, u::Array{T,2}) where T
  nElems = length(elems)
  N      = length(u[:,elems[1].nodes])
  M      = (N+1)N÷2

  Φ = Vector{adiff.D2{N,M,T}}(undef, nElems)
  Threads.@threads for ii=1:nElems
    Φ[ii] = getδϕ(elems[ii], u[:,elems[ii].nodes])
  end
  Φ
end

include("./elasticelements.2ndord.jl")

