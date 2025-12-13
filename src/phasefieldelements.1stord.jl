# ===========================================================================
# PHASE FIELD (P) ELEMENTS
# ===========================================================================

function Tria03P(nodes::Vector{<:Integer}, 
                 p0::Vector{Vector{T}};
                 mat::M=Materials.Hooke(),
                 bReduced::Bool=false) where {T<:Number, M<:Material}
  
  N(ξ,η) = SVector(1-ξ-η, ξ, η)
  GPs    = ((SVector{2,T}(1/3, 1/3), T(0.5)),)
  
  nGP = length(GPs)
  nN  = length(nodes)

  N0,∇N,wgt,V = _calculate_pf_fields_2d(N, GPs, nodes, p0)

  C2DP{nGP,M,T,nN,1}(nodes, N0, ∇N, wgt, V, mat) 
end

function Quad04P(nodes::Vector{<:Integer}, 
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

  N0,∇N,wgt,V = _calculate_pf_fields_2d(N, GPs, nodes, p0)

  C2DP{nGP,M,T,nN,1}(nodes, N0, ∇N, wgt, V, mat) 
end

function Tet04P(nodes::Vector{<:Integer}, 
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

  N0,∇N,wgt,V = _calculate_pf_fields_3d(N, GPs, nodes, p0)

  C3DP{nGP,M,T,nN,1}(nodes, N0, ∇N, wgt, V, mat) 
end

function Hex08P(nodes::Vector{<:Integer}, 
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

  N0,∇N,wgt,V = _calculate_pf_fields_3d(N, GPs, nodes, p0)

  C3DP{nGP,M,T,nN,1}(nodes, N0, ∇N, wgt, V, mat) 
end

function Wdg06P(nodes::Vector{<:Integer}, 
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

  N0,∇N,wgt,V = _calculate_pf_fields_3d(N, GPs, nodes, p0)

  C3DP{nGP,M,T,nN,1}(nodes, N0, ∇N, wgt, V, mat) 
end

const QuadP = Quad04P     # backward compatilbilty, will be removed
const TriaP = Tria03P     # backward compatilbilty, will be removed

# 2D Phase Field Fields (includes N0 - shape function values)
function _calculate_pf_fields_2d(N::F, GPs, nodes::Vector, p0::Vector{Vector{T}}) where {F<:Function, T<:Number}
  nGP = length(GPs)
  N0  = Vector{Vector{T}}(undef, nGP)

  @inbounds for (ii, (coords, wii)) in enumerate(GPs)
    N0[ii]  = N(coords...)
  end

  ∇N, wgt, V = _calculate_mech_fields_2d(N, GPs, nodes, p0) 

  return tuple(N0...), ∇N, wgt, V
end

# 3D Phase Field Fields (includes N0 - shape function values)
function _calculate_pf_fields_3d(N::F, GPs, nodes::Vector, p0::Vector{Vector{T}}) where {F<:Function, T<:Number}
  nGP = length(GPs)
  N0  = Vector{Vector{T}}(undef, nGP)

  @inbounds for (ii, (coords, wii)) in enumerate(GPs)
    N0[ii]  = N(coords...)
  end

  ∇N, wgt, V = _calculate_mech_fields_3d(N, GPs, nodes, p0) 

  return tuple(N0...), ∇N, wgt, V
end

