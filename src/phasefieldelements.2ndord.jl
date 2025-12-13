# ===========================================================================
# phasefieldelements.2ndord.jl
# ---------------------------------------------------------------------------
# Second-order (Quadratic) Phase-Field Elements
# ===========================================================================

#=
# Theoretical Background
Phase field elements interpolate the scalar field $d$ (damage/phase).
This file mirrors the mechanical elements but constructs `CPElem` types which
store the shape function values `N` (needed for the $d^2$ term in free energy)
in addition to gradients `∇N` (needed for $\nabla d$).

The same improvements (Jacobian transpose fix, Reduced Integration option)
are applied here.
=#

# using ..Materials:PhaseField

# Reuse the Gauss rule helper if defining in same module, or redefine/import.
# Assuming standard access, we redefine locally for standalone safety or assume include order.
function get_gauss_rule(::Type{Val{:line}}, bReduced::Bool)
    if bReduced
        val = 0.577350269189626
        return ((-val, 1.0), (val, 1.0))
    else
        v1, w1 = 0.774596669241483, 0.555555555555556
        v2, w2 = 0.0, 0.888888888888889
        return ((-v1, w1), (v2, w2), (v1, w1))
    end
end

# ---------------------------------------------------------------------------
# 2D Phase Field Elements
# ---------------------------------------------------------------------------

"""
    Tria06P(nodes, p0; mat=Materials.Hooke(), bReduced=false)

Constructs a 6-node quadratic triangular phase-field element.
"""
function Tria06P(nodes::Vector{<:Integer}, 
                 p0::Vector{Vector{T}};
                 mat=Materials.Hooke(),
                 bReduced::Bool=false) where T<:Number

    function N(ξ, η) 
        λ = (1.0 - ξ - η, ξ, η)
        SVector(
            λ[1]*(2λ[1]-1), λ[2]*(2λ[2]-1), λ[3]*(2λ[3]-1), 
            4λ[1]λ[2], 4λ[2]λ[3], 4λ[3]λ[1]
        )
    end

    w = 1.0/3.0
    GPs = (
        (SVector(1.0/6.0, 1.0/6.0), w),
        (SVector(2.0/3.0, 1.0/6.0), w),
        (SVector(1.0/6.0, 2.0/3.0), w)
    )

    nGP = length(GPs)
    N0  = Matrix{Vector{T}}(undef, 1, nGP)
    Nx  = Matrix{Vector{T}}(undef, 1, nGP)
    Ny  = Matrix{Vector{T}}(undef, 1, nGP)
    wgt = Matrix{T}(undef, 1, nGP)
    A   = zero(T)

    @inbounds for (ii, (Pii, wii)) in enumerate(GPs)
        N_dual = N(adiff.D1(Pii)...)
        p = sum(N_dual[a] * p0[a] for a in 1:6)
        
        J = SMatrix{2,2,T}(p[i].g[j] for i in 1:2, j in 1:2)
        grads_phys = J' \ hcat(adiff.grad.(N_dual)...)

        N0[1, ii]  = adiff.val.(N_dual)
        Nx[1, ii]  = grads_phys[1, :]
        Ny[1, ii]  = grads_phys[2, :]
        wgt[1, ii] = 0.5 * abs(detJ(J)) * wii
        A += wgt[1, ii]
    end

    C2DP(nodes, tuple(N0[1,:]...), tuple(Nx[1,:]...), tuple(Ny[1,:]...), tuple(wgt[1,:]...), A, mat, 2)
end

"""
    Quad09P(nodes, p0; mat=Materials.Hooke(), bReduced=false)

Constructs a 9-node quadratic phase-field quadrilateral.
"""
function Quad09P(nodes::Vector{<:Integer}, 
                 p0::Vector{Vector{T}};
                 mat=Materials.Hooke(),
                 bReduced::Bool=false) where T<:Number

    poly(ξ) = SVector(0.5*ξ*(ξ-1), (1-ξ^2), 0.5*ξ*(ξ+1))
    
    function N(ξ, η)
        Nξ, Nη = poly(ξ), poly(η)
        SVector(
            Nξ[1]*Nη[1], Nξ[3]*Nη[1], Nξ[3]*Nη[3], Nξ[1]*Nη[3],
            Nξ[2]*Nη[1], Nξ[3]*Nη[2], Nξ[2]*Nη[3], Nξ[1]*Nη[2],
            Nξ[2]*Nη[2]
        )
    end

    GP = get_gauss_rule(Val{:line}, bReduced)
    nGP_1d = length(GP)
    
    N0  = Matrix{Vector{T}}(undef, nGP_1d, nGP_1d)
    Nx  = Matrix{Vector{T}}(undef, nGP_1d, nGP_1d)
    Ny  = Matrix{Vector{T}}(undef, nGP_1d, nGP_1d)
    wgt = Matrix{T}(undef, nGP_1d, nGP_1d)
    A   = zero(T)

    @inbounds for (jj, (η, wη)) in enumerate(GP), (ii, (ξ, wξ)) in enumerate(GP)
        
      N_dual = N(adiff.D1([ξ,η])...)
        p = sum(N_dual[a] * p0[a] for a in 1:9)
        
        # transposed Jacobian
        Jᵀ = SMatrix{2,2,T}(p[ii].g[jj] for jj in 1:2, ii in 1:2)
        grads_phys = Jᵀ \ hcat(adiff.grad.(N_dual)...)

        N0[ii, jj]  = adiff.val.(N_dual)
        Nx[ii, jj]  = grads_phys[1, :]
        Ny[ii, jj]  = grads_phys[2, :]
        
        val_w = detJ(Jᵀ) * wξ * wη
        wgt[ii, jj] = val_w
        A += val_w
    end

    C2DP(nodes, tuple(vec(N0)...), tuple(vec(Nx)...), tuple(vec(Ny)...), tuple(vec(wgt)...), A, mat, 2)
end

# ---------------------------------------------------------------------------
# 3D Phase Field Elements
# ---------------------------------------------------------------------------
#=
function Tet10P(nodes::Vector{<:Integer}, 
                p0::Vector{Vector{T}};
                mat::M=Materials.Hooke() )where {T<:Number,M<:Material}

  La = 0.585410196624968
  Lb = 0.138196601125010
  ξ  = [La, Lb, Lb, Lb] 
  A  = hcat([[1,p[1],p[2],p[3],p[1]^2,p[2]^2,p[3]^2,p[2]p[3],p[1]p[3],p[1]p[2]] 
             for p in p0]...)
  C  = inv(A)
  V  = det(A[1:4,1:4])/6

  nGP = 4
  nN  = length(nodes)
  N0  = Vector{Vector{T}}(undef, nGP)
  Nx  = Vector{Vector{T}}(undef, nGP)
  Ny  = Vector{Vector{T}}(undef, nGP)
  Nz  = Vector{Vector{T}}(undef, nGP)
  for ii in 1:4
    ξ      = circshift([La, Lb, Lb, Lb], ii)
    N0[ii] = p0[1]ξ[1]+p0[2]ξ[2]+p0[3]ξ[3]+p0[4]ξ[4]
    Nx[ii] = C[:,2]+2C[:,5]N0[ii][1]+C[:,9]N0[ii][3]+C[:,10]N0[ii][2] 
    Ny[ii] = C[:,3]+2C[:,6]N0[ii][2]+C[:,8]N0[ii][3]+C[:,10]N0[ii][1]
    Nz[ii] = C[:,4]+2C[:,7]N0[ii][3]+C[:,8]N0[ii][2]+C[:, 8]N0[ii][1] 
  end
  wgt = fill(0.25V, 4)

  ∇N = tuple(tuple(Nx...), tuple(Ny...), tuple(Nz...),)
  C3DP{nGP,M,T,nN,2}(nodes, tuple(N0...), ∇N, tuple(wgt...), V, mat)
end
=#

"""
    Tet10P(nodes, p0; mat=Materials.Hooke(), bReduced=false)

Constructs a 10-node quadratic phase-field tetrahedron.
"""
function tet_quadrature_11point()
    # 11-point rule, 5th degree precision
    a1, a2 = 0.25, 0.0714285714285714  # 1/4, 1/14
    b1 = 0.7857142857142857  # 11/14
    b2 = 0.3994035761667992
    b3 = 0.1005964238332008
    
    w1 = -0.01315555555555556  # -74/5625
    w2 = 0.007622222222222222   # 343/45000
    w3 = 0.02488888888888889    # 56/2250
    
    # 1 point at centroid
    gp1 = (SVector(0.25, 0.25, 0.25), w1)
    
    # 4 points near vertices
    gps_vert = [
        (SVector(a2, a2, a2), w2),
        (SVector(b1, a2, a2), w2),
        (SVector(a2, b1, a2), w2),
        (SVector(a2, a2, b1), w2),
    ]
    
    # 6 points on edges
    gps_edge = [
        (SVector(b2, b3, b3), w3),
        (SVector(b3, b2, b3), w3),
        (SVector(b3, b3, b2), w3),
        (SVector(b3, b2, b2), w3),
        (SVector(b2, b3, b2), w3),
        (SVector(b2, b2, b3), w3),
    ]
    
    return [gp1; gps_vert; gps_edge]
end
function tet_quadrature_15point()
    # 15-point rule, 5th degree precision
    a = 0.25
    b1, b2 = 0.0, 0.5
    c1 = 0.3333333333333333
    c2 = (1.0 - c1) / 3.0
    
    w1 = 0.030283678097089
    w2 = 0.006026785714286
    w3 = 0.011645249086029
    
    # 1 point at centroid
    gp1 = (SVector(a, a, a), w1)
    
    # 4 points at vertices (actually slightly inside)
    gps_vert = [
        (SVector(b1, c2, c2), w2),
        (SVector(c2, b1, c2), w2),
        (SVector(c2, c2, b1), w2),
        (SVector(c2, c2, c2), w2),
    ]
    
    # 4 points on faces
    a1 = 0.0665501535736643
    a2 = 0.4334498464263357
    w_face = 0.011645249086029
    
    gps_face = [
        (SVector(a1, a1, a2), w_face),
        (SVector(a1, a2, a1), w_face),
        (SVector(a2, a1, a1), w_face),
        (SVector(a1, a1, a1), w_face),
    ]
    
    # 6 points on edges
    e1 = 0.4592925882927231
    e2 = 0.0407074117072769
    w_edge = 0.010949141561386
    
    gps_edge = [
        (SVector(e1, e1, e2), w_edge),
        (SVector(e1, e2, e1), w_edge),
        (SVector(e2, e1, e1), w_edge),
        (SVector(e1, e2, e2), w_edge),
        (SVector(e2, e1, e2), w_edge),
        (SVector(e2, e2, e1), w_edge),
    ]
    
    return [gp1; gps_vert; gps_face; gps_edge]
end
function tet_quadrature_04point()
  a, b, w = 0.5854101966249685, 0.1381966011250105, 1.0/24.0
  return ((SVector(a, b, b), w), (SVector(b, a, b), w),
         (SVector(b, b, a), w), (SVector(b, b, b), w) )
end
tet_15pts_gemini = [(SVector(1/4, 1/4, 1/4), 0.030283678097089),
                    (SVector(1/3, 1/3, 1/3), 0.006026785714286),
                    (SVector(1/3, 1/3, 0.0), 0.006026785714286),
                    (SVector(1/3, 0.0, 1/3), 0.006026785714286),
                    (SVector(0.0, 1/3, 1/3), 0.006026785714286),
                    (SVector(0.727272727272727, 0.090909090909091, 0.090909090909091), 0.011645249086029),
                    (SVector(0.090909090909091, 0.727272727272727, 0.090909090909091), 0.011645249086029),
                    (SVector(0.090909090909091, 0.090909090909091, 0.727272727272727), 0.011645249086029),
                    (SVector(0.090909090909091, 0.090909090909091, 0.090909090909091), 0.011645249086029),
                    (SVector(0.433449846426336, 0.433449846426336, 0.066550153573664), 0.010949141561386),
                    (SVector(0.433449846426336, 0.066550153573664, 0.433449846426336), 0.010949141561386),
                    (SVector(0.433449846426336, 0.066550153573664, 0.066550153573664), 0.010949141561386),
                    (SVector(0.066550153573664, 0.433449846426336, 0.433449846426336), 0.010949141561386),
                    (SVector(0.066550153573664, 0.433449846426336, 0.066550153573664), 0.010949141561386),
                    (SVector(0.066550153573664, 0.066550153573664, 0.433449846426336), 0.010949141561386)
                   ]
"""
    Tet10P(nodes, p0; mat=Materials.Hooke(), bReduced=false)

Constructs a 10-node quadratic phase-field tetrahedron.
"""
function Tet10P(nodes::Vector{<:Integer}, 
                p0::Vector{Vector{T}};
                mat=Materials.Hooke(),
                bReduced::Bool=false) where T<:Number

  function N(ξ, η, ζ)
    λ = (1.0 - ξ - η - ζ, ξ, η, ζ)
        SVector(
            λ[1]*(2λ[1]-1), λ[2]*(2λ[2]-1), λ[3]*(2λ[3]-1), λ[4]*(2λ[4]-1), 
            4λ[1]λ[2], 4λ[2]λ[3], 4λ[1]λ[3], 4λ[1]λ[4], 4λ[2]λ[4], 4λ[3]λ[4]
        )
  end

  GPs = tet_quadrature_15point()
  nGP = length(GPs)

  N0  = Vector{Vector{T}}(undef, nGP)
  Nx  = Vector{Vector{T}}(undef, nGP)
  Ny  = Vector{Vector{T}}(undef, nGP)
  Nz  = Vector{Vector{T}}(undef, nGP)
  wgt = Vector{T}(undef, nGP)
  V   = zero(T)

  @inbounds for (ii, (coords, wii)) in enumerate(GPs)
    N_dual = N(adiff.D1(coords)...)
    p = sum(N_dual[a] * p0[a] for a in 1:10)

    # transposed Jacobian
    Jᵀ = SMatrix{3,3,T}(p[ii].g[jj] for jj in 1:3, ii in 1:3)
    grads_phys = Jᵀ \ hcat(adiff.grad.(N_dual)...)

    N0[ii]  = adiff.val.(N_dual)
    Nx[ii]  = grads_phys[1, :]
    Ny[ii]  = grads_phys[2, :]
    Nz[ii]  = grads_phys[3, :]
    wgt[ii] = abs(detJ(Jᵀ)) * wii
    V += wgt[ii]
  end

  C3DP(nodes, tuple(N0...), tuple(Nx...), tuple(Ny...), tuple(Nz...), tuple(wgt...), V, mat, 2)
end

##
# ---------------------------------------------------------------------------
# INDEX MAPS FOR HEX27 (USER'S CUSTOM ORDER, BUT I_MAP IS FLIPPED 1<->3)
# Correction for negative Jacobian due to X-axis inversion (X decreases as ξ increases)
# ---------------------------------------------------------------------------

# Indices 1-8 (Corners)
# Original I_MAP: [1, 3, 3, 1, 1, 3, 3, 1] 
const I_MAP_HEX27_CORR = [3, 1, 1, 3, 3, 1, 1, 3] 
const J_MAP_HEX27 = [1, 1, 3, 3, 1, 1, 3, 3] 
const K_MAP_HEX27 = [1, 1, 1, 1, 3, 3, 3, 3] 

# Indices 9-20 (Mid-Sides: 1-2, 1-4, 1-5, 2-3, 2-6, 3-4, 3-7, 4-8, 5-6, 5-8, 6-7, 7-8)
# Original I_MAP: [2, 1, 1, 3, 3, 2, 3, 1, 2, 1, 3, 2]
const I_MAP_MIDSIDES_CORR = [2, 3, 3, 1, 1, 2, 1, 3, 2, 3, 1, 2] 
const J_MAP_MIDSIDES = [1, 2, 1, 2, 1, 3, 3, 3, 1, 2, 2, 3]
const K_MAP_MIDSIDES = [1, 1, 2, 1, 2, 1, 2, 2, 3, 3, 3, 3]

# Indices 21-26 (Mid-Face Centers: -Z, -Y, -X, +X, +Y, +Z)
# Original I_MAP: [2, 2, 1, 3, 2, 2]
const I_MAP_FACE_CENTERS_CORR = [2, 2, 3, 1, 2, 2]
const J_MAP_FACE_CENTERS = [2, 1, 2, 2, 3, 2]
const K_MAP_FACE_CENTERS = [1, 2, 2, 2, 2, 3]

# Index 27 (Body Center) - No change since I=2 (center node)
const I_MAP_CENTER_CORR = [2] 
const J_MAP_CENTER = [2] 
const K_MAP_CENTER = [2]

# Final consolidated map arrays
const I_MAP_FINAL = vcat(I_MAP_HEX27_CORR, I_MAP_MIDSIDES_CORR, I_MAP_FACE_CENTERS_CORR, I_MAP_CENTER_CORR)
const J_MAP_FINAL = vcat(J_MAP_HEX27, J_MAP_MIDSIDES, J_MAP_FACE_CENTERS, J_MAP_CENTER)
const K_MAP_FINAL = vcat(K_MAP_HEX27, K_MAP_MIDSIDES, K_MAP_FACE_CENTERS, K_MAP_CENTER)

"""
    Hex27P(nodes, p0; mat=Materials.Hooke(), bReduced=false)

Constructs a 27-node quadratic phase-field hexahedron.
"""
function Hex27P(nodes::Vector{<:Integer}, 
                p0::Vector{Vector{T}};
                mat=Materials.Hooke(),
                bReduced::Bool=false) where T<:Number

  poly(ξ) = SVector(0.5*ξ*(ξ-1), (1-ξ^2), 0.5*ξ*(ξ+1))
  #=
  function N(ξ::T, η::T, ζ::T) where T <: Number
  Ux, Uy, Uz = poly(ξ), poly(η), poly(ζ)
  vals = MArray{Tuple{3,3,3},T}(undef)
  for i in 1:3, j in 1:3, k in 1:3
  vals[i,j,k] = Ux[i] * Uy[j] * Uz[k]
  end
  return SVector(vals)
  end
  =#
function N(ξ, η, ζ)
    # Assumes poly(x) returns N_(-1), N_0, N_(1) in that order
    Nξ, Nη, Nζ = poly(ξ), poly(η), poly(ζ) 
    vals = MVector{27, typeof(Nξ[1])}(undef)

    @inbounds for idx in 1:27
        i = I_MAP_FINAL[idx] # Index for Nξ (1, 2, or 3)
        j = J_MAP_FINAL[idx] # Index for Nη
        k = K_MAP_FINAL[idx] # Index for Nζ
        
        # N_node = N_i(ξ) * N_j(η) * N_k(ζ)
        vals[idx] = Nξ[i] * Nη[j] * Nζ[k]
    end
    
    return SVector(vals)
end

  GP = get_gauss_rule(Val{:line}, bReduced)
  nGP_1d = length(GP)

  N0  = Array{Vector{T}, 3}(undef, nGP_1d, nGP_1d, nGP_1d)
  Nx  = Array{Vector{T}, 3}(undef, nGP_1d, nGP_1d, nGP_1d)
  Ny  = Array{Vector{T}, 3}(undef, nGP_1d, nGP_1d, nGP_1d)
  Nz  = Array{Vector{T}, 3}(undef, nGP_1d, nGP_1d, nGP_1d)
  wgt = Array{T, 3}(undef, nGP_1d, nGP_1d, nGP_1d)
  V   = zero(T)

  @inbounds for 
    (k, (ζ, wζ)) in enumerate(GP), 
    (j, (η, wη)) in enumerate(GP), 
    (i, (ξ, wξ)) in enumerate(GP)

    N_dual = N(adiff.D1([ξ,η,ζ])...)
    p = sum(N_dual[a] * p0[a] for a in 1:27)

    # transposed Jacobian
    Jᵀ = SMatrix{3,3,T}(p[ii].g[jj] for jj in 1:3, ii in 1:3)
    grads_phys = Jᵀ \ hcat(adiff.grad.(N_dual)...)

    N0[i,j,k]  = adiff.val.(N_dual)
    Nx[i,j,k]  = grads_phys[1, :]
    Ny[i,j,k]  = grads_phys[2, :]
    Nz[i,j,k]  = grads_phys[3, :]
    wgt[i,j,k] = detJ(Jᵀ) * wξ * wη * wζ
    V += wgt[i,j,k]
  end

  C3DP(nodes, tuple(vec(N0)...), tuple(vec(Nx)...), tuple(vec(Ny)...), tuple(vec(Nz)...), 
       tuple(vec(wgt)...), V, mat, 2)
end


