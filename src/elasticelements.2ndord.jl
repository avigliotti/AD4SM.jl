# ===========================================================================
# elasticelements.2ndord.jl
# ---------------------------------------------------------------------------
# Second-order (Quadratic) Mechanical Elements
# ===========================================================================

#=
# Theoretical Background: Second-Order Mechanical Elements

## 1. Isoparametric Mapping & Jacobian
The mapping from reference coordinates ($\boldsymbol{\xi}$) to physical coordinates ($\mathbf{x}$) is:
$$ \mathbf{x}(\boldsymbol{\xi}) = \sum_{a} N_a(\boldsymbol{\xi}) \mathbf{x}_a $$
The Jacobian matrix of this transformation is $\mathbf{J} = \partial \mathbf{x} / \partial \boldsymbol{\xi}$.
Spatial gradients of shape functions are computed via the chain rule:
$$ \nabla_\mathbf{x} N = \mathbf{J}^{-T} \nabla_{\boldsymbol{\xi}} N $$
In the code, this is implemented as `J' \ ∇N_ref`.

## 2. Volumetric Locking & Reduced Integration
Full integration (e.g., $3\times3$ for Quad09) imposes strict volumetric constraints that can cause "locking" (stiff behavior/checkerboard pressure).
Reduced integration (e.g., $2\times2$) relaxes these constraints, providing accurate results for nearly incompressible materials or constrained boundaries.

## 3. Element Types
- **Lagrange (Quad09, Hex27)**: Full tensor product. High accuracy but prone to locking without reduced integration.
- **Serendipity (Quad08, Hex20)**: Boundary nodes only. More efficient, less prone to locking, but can suffer from hourglassing if under-integrated.
- **Simplices (Tria06, Tet10)**: Standard quadratic elements.
=#

# Quadrature for 1D line rule (used for tensor products)
function get_gauss_rule_1d(bReduced::Bool)
    T = Float64
    if bReduced # 2 points (Order 3 precision)
        val = 0.577350269189626
        return ((-val, T(1.0)), (val, T(1.0)))
    else # 3 points (Order 5 precision)
        v1, w1 = 0.774596669241483, 0.555555555555556
        v2, w2 = 0.0, 0.888888888888889
        return ((-v1, w1), (v2, w2), (v1, w1))
    end
end

# Tet Quadrature Rules (used for Tet10/Tet10P)
function tet_quadrature_15point()
    T = Float64
    # 15-point rule, 5th degree precision (Consolidated coordinates/weights)
    w1 = 0.030283678097089
    w2 = 0.006026785714286
    w3 = 0.011645249086029
    w4 = 0.010949141561386
    
    return (
      (SVector{3,T}(0.25, 0.25, 0.25), w1),
      (SVector{3,T}(1/3, 1/3, 1/3), w2),
      (SVector{3,T}(1/3, 1/3, 0.0), w2),
      (SVector{3,T}(1/3, 0.0, 1/3), w2),
      (SVector{3,T}(0.0, 1/3, 1/3), w2),
      (SVector{3,T}(0.727272727272727, 0.090909090909091, 0.090909090909091), w3),
      (SVector{3,T}(0.090909090909091, 0.727272727272727, 0.090909090909091), w3),
      (SVector{3,T}(0.090909090909091, 0.090909090909091, 0.727272727272727), w3),
      (SVector{3,T}(0.090909090909091, 0.090909090909091, 0.090909090909091), w3),
      (SVector{3,T}(0.433449846426336, 0.433449846426336, 0.066550153573664), w4),
      (SVector{3,T}(0.433449846426336, 0.066550153573664, 0.433449846426336), w4),
      (SVector{3,T}(0.433449846426336, 0.066550153573664, 0.066550153573664), w4),
      (SVector{3,T}(0.066550153573664, 0.433449846426336, 0.433449846426336), w4),
      (SVector{3,T}(0.066550153573664, 0.433449846426336, 0.066550153573664), w4),
      (SVector{3,T}(0.066550153573664, 0.066550153573664, 0.433449846426336), w4)
    )
end
function tet_quadrature_04point()
  T = Float64
  a, b, w = 0.5854101966249685, 0.1381966011250105, 1.0/24.0
  return (
    (SVector{3,T}(a, b, b), w), 
    (SVector{3,T}(b, a, b), w),
    (SVector{3,T}(b, b, a), w), 
    (SVector{3,T}(b, b, b), w) 
  )
end

function N_Quad08(ξ::T, η::T) where T
  # Nodal Coordinates for Serendipity Quad08 (used in N_Quad08)
  Q8_ξ_n = SVector{8,T}(-1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0)
  Q8_η_n = SVector{8,T}(-1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0)

  # Serendipity Shape Functions (8 nodes)
  Nd = MVector{8, T}(undef)
  for k in 1:4 
    Nd[k] = 0.25 * (1 + ξ*Q8_ξ_n[k]) * (1 + η*Q8_η_n[k]) * (ξ*Q8_ξ_n[k] + η*Q8_η_n[k] - 1)
  end
  for k in 5:8 
    if Q8_ξ_n[k] == 0
      Nd[k] = 0.5 * (1 - ξ^2) * (1 + η*Q8_η_n[k])
    else
      Nd[k] = 0.5 * (1 + ξ*Q8_ξ_n[k]) * (1 - η^2)
    end
  end
  return SVector(Nd)
end

# Hex27 Index Maps (Used in N_Hex27_Mapped)
const I_MAP_HEX27_CORR = [3, 1, 1, 3, 3, 1, 1, 3] 
const J_MAP_HEX27 = [1, 1, 3, 3, 1, 1, 3, 3] 
const K_MAP_HEX27 = [1, 1, 1, 1, 3, 3, 3, 3] 
const I_MAP_MIDSIDES_CORR = [2, 3, 3, 1, 1, 2, 1, 3, 2, 3, 1, 2] 
const J_MAP_MIDSIDES = [1, 2, 1, 2, 1, 3, 3, 3, 1, 2, 2, 3]
const K_MAP_MIDSIDES = [1, 1, 2, 1, 2, 1, 2, 2, 3, 3, 3, 3]
const I_MAP_FACE_CENTERS_CORR = [2, 2, 3, 1, 2, 2]
const J_MAP_FACE_CENTERS = [2, 1, 2, 2, 3, 2]
const K_MAP_FACE_CENTERS = [1, 2, 2, 2, 2, 3]
const I_MAP_CENTER_CORR = [2] 
const J_MAP_CENTER = [2] 
const K_MAP_CENTER = [2]
const I_MAP_FINAL = vcat(I_MAP_HEX27_CORR, I_MAP_MIDSIDES_CORR, I_MAP_FACE_CENTERS_CORR, I_MAP_CENTER_CORR)
const J_MAP_FINAL = vcat(J_MAP_HEX27, J_MAP_MIDSIDES, J_MAP_FACE_CENTERS, J_MAP_CENTER)
const K_MAP_FINAL = vcat(K_MAP_HEX27, K_MAP_MIDSIDES, K_MAP_FACE_CENTERS, K_MAP_CENTER)

# Helper function to generate N-D tensor product GPs from 1D scalar rules
function generate_tensor_GPs(GP_1d, dim::Int, T)
    coords_1d = [p[1] for p in GP_1d]
    weights_1d = [p[2] for p in GP_1d]
    
    if dim == 2
        GPs = Tuple{SVector{2,T}, T}[]
        for i in 1:length(GP_1d), j in 1:length(GP_1d)
            push!(GPs, (SVector(coords_1d[i], coords_1d[j]), weights_1d[i] * weights_1d[j]))
        end
        return tuple(GPs...)
    elseif dim == 3
        GPs = Tuple{SVector{3,T}, T}[]
        for i in 1:length(GP_1d), j in 1:length(GP_1d), k in 1:length(GP_1d)
            push!(GPs, (SVector(coords_1d[i], coords_1d[j], coords_1d[k]), weights_1d[i] * weights_1d[j] * weights_1d[k]))
        end
        return tuple(GPs...)
    end
    error("Unsupported dimension for tensor product.")
end

# ===========================================================================
# REFACTORED CONTINUOUS (MECHANICAL) ELEMENTS
# ===========================================================================

"""
    Tria06(nodes, p0; mat=Materials.Hooke(), bReduced=false)
Constructs a 6-node quadratic triangular element.
**Quadrature**: Uses a fixed 3-point rule (Strang).
"""
function Tria06(nodes::Vector{<:Integer}, 
                p0::Vector{<:AbstractVector{T}};
                mat=Materials.Hooke(),
                bReduced::Bool=false) where T<:Number

    # Shape functions: L1(2L1-1), 4L1L2, etc.
    function N(ξ, η) 
        λ = (1.0 - ξ - η, ξ, η) # Barycentric coordinates
        SVector(
            λ[1]*(2λ[1]-1), λ[2]*(2λ[2]-1), λ[3]*(2λ[3]-1), # Corners
            4λ[1]*λ[2],      4λ[2]*λ[3],      
            4λ[3]*λ[1]       # Midsides
        )
    end

    # 3-point rule (Degree 2) - Standard for quadratic triangles
    w = T(1.0/3.0)
    GPs = (
        (SVector{2,T}(1.0/6.0, 1.0/6.0), w),
        (SVector{2,T}(2.0/3.0, 1.0/6.0), w),
        (SVector{2,T}(1.0/6.0, 2.0/3.0), w),
    )

    ∇N, wgt, V = _calculate_mech_fields_2d(N, GPs, nodes, p0)

    # C2DE expects ∇N as two separate tuples: Nx and Ny
    C2DE(nodes, ∇N[1], ∇N[2], wgt, V, mat, 2)
end

"""
    Quad09(nodes, p0; mat=Materials.Hooke(), bReduced=false)
Constructs a 9-node quadratic Lagrangian quadrilateral.
"""
function Quad09(nodes::Vector{<:Integer}, 
                p0::Vector{<:AbstractVector{T}};
                mat=Materials.Hooke(),
                bReduced::Bool=false) where T<:Number

    # 1D Lagrange polynomials
    poly(ξ) = SVector(0.5*ξ*(ξ-1), (1-ξ^2), 0.5*ξ*(ξ+1))

    # Tensor product N(ξ,η)
    function N(ξ, η)
        Nξ, Nη = poly(ξ), poly(η)
        # Ordering: Corners(1-4), Midsides(5-8), Center(9)
        SVector(
            Nξ[1]*Nη[1], Nξ[3]*Nη[1], Nξ[3]*Nη[3], Nξ[1]*Nη[3],
            Nξ[2]*Nη[1], Nξ[3]*Nη[2], Nξ[2]*Nη[3], Nξ[1]*Nη[2],
            Nξ[2]*Nη[2]
        )
    end

    # Generate 2D Gaussian points from 1D rule
    GP_1d = get_gauss_rule_1d(bReduced)
    GPs = generate_tensor_GPs(GP_1d, 2, T)

    ∇N, wgt, V = _calculate_mech_fields_2d(N, GPs, nodes, p0)

    C2DE(nodes, ∇N[1], ∇N[2], wgt, V, mat, 2)
end

"""
    Quad08(nodes, p0; mat=Materials.Hooke(), bReduced=false)
Constructs an 8-node Serendipity quadratic quadrilateral.
"""
function Quad08(nodes::Vector{<:Integer}, 
                p0::Vector{<:AbstractVector{T}};
                mat=Materials.Hooke(),
                bReduced::Bool=false) where T<:Number

    # N_Quad08 is defined externally above
    
    # Generate 2D Gaussian points from 1D rule
    GP_1d = get_gauss_rule_1d(bReduced)
    GPs = generate_tensor_GPs(GP_1d, 2, T)

    ∇N, wgt, V = _calculate_mech_fields_2d(N_Quad08, GPs, nodes, p0)

    C2DE(nodes, ∇N[1], ∇N[2], wgt, V, mat, 2)
end

"""
    Tet10(nodes, p0; mat=Materials.Hooke(), bReduced=false)
Constructs a 10-node quadratic tetrahedron.
"""
function Tet10(nodes::Vector{<:Integer}, 
               p0::Vector{<:AbstractVector{T}};
               mat=Materials.Hooke(),
               bReduced::Bool=false) where T<:Number

    # Barycentric shape functions
    function N(ξ, η, ζ)
      λ = (1.0 - ξ - η - ζ, ξ, η, ζ)
      SVector(λ[1]*(2λ[1]-1),
              λ[2]*(2λ[2]-1), 
              λ[3]*(2λ[3]-1), 
              λ[4]*(2λ[4]-1), 
              4λ[1]*λ[2], 
              4λ[2]*λ[3], 
              4λ[3]*λ[1], 
              4λ[1]*λ[4], 
              4λ[2]*λ[4], 
              4λ[3]*λ[4])
    end

    # Quadrature: Use the standard 4-point rule (Keast, Degree 2)
    GPs = tet_quadrature_04point()

    ∇N, wgt, V = _calculate_mech_fields_3d(N, GPs, nodes, p0)

    # C3DE expects ∇N as three separate tuples: Nx, Ny, Nz
    C3DE(nodes, ∇N[1], ∇N[2], ∇N[3], wgt, V, mat, 2)
end

"""
    Hex27(nodes, p0; mat=Materials.Hooke(), bReduced=false)
Constructs a 27-node quadratic Lagrange hexahedron.
"""
function Hex27(nodes::Vector{<:Integer}, 
               p0::Vector{<:AbstractVector{T}};
               mat=Materials.Hooke(),
               bReduced::Bool=false) where T<:Number

    poly(ξ) = SVector(0.5*ξ*(ξ-1), (1-ξ^2), 0.5*ξ*(ξ+1))

    function N_Hex27_Mapped(ξ, η, ζ)
        # Nodal ordering is corrected via map arrays
        Nξ, Nη, Nζ = poly(ξ), poly(η), poly(ζ) 
        vals = MVector{27, typeof(Nξ[1])}(undef)

        @inbounds for idx in 1:27
            i = I_MAP_FINAL[idx] # Index for Nξ (1, 2, or 3)
            j = J_MAP_FINAL[idx] # Index for Nη
            k = K_MAP_FINAL[idx] # Index for Nζ
            vals[idx] = Nξ[i] * Nη[j] * Nζ[k]
        end
        
        return SVector(vals)
    end

    # Generate 3D Gaussian points from 1D rule
    GP_1d = get_gauss_rule_1d(bReduced)
    GPs = generate_tensor_GPs(GP_1d, 3, T)

    ∇N, wgt, V = _calculate_mech_fields_3d(N_Hex27_Mapped, GPs, nodes, p0)

    C3DE(nodes, ∇N[1], ∇N[2], ∇N[3], wgt, V, mat, 2)
end

