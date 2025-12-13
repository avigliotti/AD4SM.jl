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
# ===========================================================================
# CONTINUOUS (MECHANICAL) ELEMENTS
# ===========================================================================

function Line(nodes::Vector{<:Integer}, 
              p0::Vector{Vector{T}};
              mat::M=Materials.Hooke1D(),
              bReduced::Bool=false) where {T<:Number, M<:Material}
  
  x1, x2 = p0[1], p0[2]
  L      = abs(x2 - x1)
  
  Nx  = [(x2-x1)/L] 
  wgt = [one(T)]       
  A   = L    

  C1DE(nodes, tuple(Nx...), tuple(wgt...), A, mat) 
end

function Tria03(nodes::Vector{<:Integer}, 
                p0::Vector{Vector{T}}; 
                mat::M=Materials.Hooke(),
                bReduced::Bool=false) where {T<:Number, M<:Material}

  N(ξ,η) = SVector(1-ξ-η, ξ, η)
  GPs = ((SVector{2,T}(1/3, 1/3), T(0.5)),)
  
  nGP = length(GPs)
  nN  = length(nodes)

  ∇N,wgt,V = _calculate_mech_fields_2d(N, GPs, nodes, p0)

  C2D{nGP,M,T,nN,1}(nodes, ∇N, wgt, V, mat) 
end

function Quad04(nodes::Vector{<:Integer}, 
                p0::Vector{Vector{T}};
                mat::M=Materials.Hooke(), 
                bReduced::Bool=false) where {T<:Number, M<:Material}

  function N(ξ, η)
      omx, opx = 1-ξ, 1+ξ
      ome, ope = 1-η, 1+η
      SVector(omx*ome, opx*ome, opx*ope, omx*ope) .* T(0.25)
  end

  GPs = if bReduced
      ((SVector{2,T}(0, 0), T(4.0)),)
  else
      g = T(1/√3) 
      w = one(T)
      ((SVector{2,T}(-g, -g), w), (SVector{2,T}( g, -g), w),
       (SVector{2,T}( g,  g), w), (SVector{2,T}(-g,  g), w))
  end

  nGP = length(GPs)
  nN  = length(nodes)

  ∇N,wgt,V = _calculate_mech_fields_2d(N, GPs, nodes, p0)

  C2D{nGP,M,T,nN,1}(nodes, ∇N, wgt, V, mat) 
end

function Tet04(nodes::Vector{<:Integer}, 
               p0::Vector{Vector{T}};
               mat::M=Materials.Hooke(),
               bReduced::Bool=false) where {T<:Number, M<:Material}

  N(ξ,η,ζ) = SVector(1-ξ-η-ζ, ξ, η, ζ)

  GPs = if bReduced
    ((SVector{3,T}(0.25, 0.25, 0.25), T(1/6)),)
  else 
    a, b = T(0.5854101966249685), T(0.1381966011250105)
    w    = T(1/24) 
    ((SVector{3,T}(a,b,b),w), (SVector{3,T}(b,a,b),w),
     (SVector{3,T}(b,b,a),w), (SVector{3,T}(b,b,b),w))
  end

  nGP = length(GPs)
  nN  = length(nodes)

  ∇N,wgt,V = _calculate_mech_fields_3d(N, GPs, nodes, p0)

  C3D{nGP,M,T,nN,1}(nodes, ∇N, wgt, V, mat) 
end

function Hex08(nodes::Vector{<:Integer}, 
               p0::Vector{Vector{T}};
               mat::M=Materials.Hooke(),
               bReduced::Bool=false) where {T<:Number, M<:Material}

  function N(ξ, η, ζ)
      omx, opx = 1-ξ, 1+ξ
      ome, ope = 1-η, 1+η
      omz, opz = 1-ζ, 1+ζ
      SVector(omx*ome*omz, opx*ome*omz, opx*ope*omz, omx*ope*omz,
              omx*ome*opz, opx*ome*opz, opx*ope*opz, omx*ope*opz) .* T(0.125)
  end

  GPs = if bReduced
      ((SVector{3,T}(0, 0, 0), T(8.0)),)
  else
      g = T(1/√3)
      w = one(T)
      pts = SVector{3,T}[]
      for k in (-g,g), j in (-g,g), i in (-g,g)
          push!(pts, SVector{3,T}(i,j,k))
      end
      Tuple((p, w) for p in pts)
  end

  nGP = length(GPs)
  nN  = length(nodes)

  ∇N,wgt,V = _calculate_mech_fields_3d(N, GPs, nodes, p0)

  C3D{nGP,M,T,nN,1}(nodes, ∇N, wgt, V, mat) 
end

function Wdg06(nodes::Vector{<:Integer}, 
                p0::Vector{Vector{T}};
               mat::M=Materials.Hooke(),
               bReduced::Bool=false) where {T<:Number, M<:Material}

  N(ξ,η,ζ) = SVector((1-ζ)*(1-ξ-η), (1-ζ)*ξ, (1-ζ)*η,
                     (1+ζ)*(1-ξ-η), (1+ζ)*ξ, (1+ζ)*η) .* T(0.5)

  GPs = if bReduced
      ((SVector{3,T}(1/3, 1/3, 0), T(1.0)),)
  else
      r23, r16, sq3 = T(2/3), T(1/6), T(1/√3)
      w = T(1/3)
      ((SVector{3,T}(r23, r16,  sq3), w), (SVector{3,T}(r23, r16, -sq3), w),
       (SVector{3,T}(r16, r23,  sq3), w), (SVector{3,T}(r16, r23, -sq3), w),
       (SVector{3,T}(r16, r16,  sq3), w), (SVector{3,T}(r16, r16, -sq3), w))
  end

  nGP = length(GPs)
  nN  = length(nodes)

  ∇N,wgt,V = _calculate_mech_fields_3d(N, GPs, nodes, p0)

  C3D{nGP,M,T,nN,1}(nodes, ∇N, wgt, V, mat) 
end

const Quad = Quad04       # backward compatilbilty, will be removed
const Tria = Tria03       # backward compatilbilty, will be removed

# 2D Mechanical Fields (Used by Tria03, Quad04)
function _calculate_mech_fields_2d(N::F, GPs, nodes::Vector, p0::Vector{Vector{T}}) where {F<:Function, T<:Number}
    nGP = length(GPs)
    nN  = length(nodes)

    Nx  = Vector{Vector{T}}(undef, nGP)
    Ny  = Vector{Vector{T}}(undef, nGP)
    wgt = Vector{T}(undef, nGP)
    V   = zero(T)

    @inbounds for (ii, (coords, wii)) in enumerate(GPs)
        N_dual = N(adiff.D1(coords)...)
        # Interpolate physical coordinates p = {x, y}
        p      = sum(N_dual[k] * p0[k] for k in 1:nN)
        # Transposed Jacobian Jᵀ (2x2)
        Jᵀ     = SMatrix{2,2,T}(p[k].g[j] for j in 1:2, k in 1:2)
        # Map gradients: J⁻ᵀ ∇_ξ N
        grads  = Jᵀ \ hcat(adiff.grad.(N_dual)...)

        Nx[ii]  = grads[1, :]
        Ny[ii]  = grads[2, :]
        # FIX: Use standard `det` function
        wgt[ii] = det(Jᵀ) * wii 
        V      += wgt[ii]
    end
    
    ∇N = tuple(tuple(Nx...), tuple(Ny...))
    # Return ∇N (tuple of tuples), wgt (tuple), V (scalar)
    return ∇N, tuple(wgt...), V
end

# 3D Mechanical Fields (Used by Tet04, Hex08, Wdg06)
function _calculate_mech_fields_3d(N::F, GPs, nodes::Vector, p0::Vector{Vector{T}}) where {F<:Function, T<:Number}
    nGP = length(GPs)
    nN  = length(nodes)

    Nx  = Vector{Vector{T}}(undef, nGP)
    Ny  = Vector{Vector{T}}(undef, nGP)
    Nz  = Vector{Vector{T}}(undef, nGP)
    wgt = Vector{T}(undef, nGP)
    V   = zero(T)

    @inbounds for (ii, (coords, wii)) in enumerate(GPs)
        N_dual = N(adiff.D1(coords)...)
        p      = sum(N_dual[k] * p0[k] for k in 1:nN)
        # Transposed Jacobian Jᵀ (3x3)
        Jᵀ     = SMatrix{3,3,T}(p[k].g[j] for j in 1:3, k in 1:3)
        grads  = Jᵀ \ hcat(adiff.grad.(N_dual)...)

        Nx[ii]  = grads[1, :]
        Ny[ii]  = grads[2, :]
        Nz[ii]  = grads[3, :]
        # FIX: Use standard `det` function
        wgt[ii] = det(Jᵀ) * wii
        V      += wgt[ii]
    end

    ∇N = tuple(tuple(Nx...), tuple(Ny...), tuple(Nz...))
    return ∇N, tuple(wgt...), V
end

#
# axysimmetric elments need to be fixed
#
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

