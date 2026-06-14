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

# ===========================================================================
# REFACTORED PHASE FIELD (P) ELEMENTS
# ===========================================================================

"""
    Tria06P(nodes, p0; mat=Materials.NoMat(), bReduced=false)
Constructs a 6-node quadratic triangular phase-field element.
"""
function Tria06P(nodes::Vector{<:Integer}, 
                 p0::Vector{<:AbstractVector{T}};
                 mat=Materials.NoMat(),
                 bReduced::Bool=false) where T<:Number

    # Shape functions (same as Tria06)
    function N(ξ, η) 
        λ = (1.0 - ξ - η, ξ, η)
        SVector(
            λ[1]*(2λ[1]-1), λ[2]*(2λ[2]-1), λ[3]*(2λ[3]-1), 
            4λ[1]*λ[2], 4λ[2]*λ[3], 4λ[3]*λ[1]
        )
    end

    # Quadrature (same as Tria06)
    w = T(1.0/3.0)
    GPs = (
        (SVector{2,T}(1.0/6.0, 1.0/6.0), w),
        (SVector{2,T}(2.0/3.0, 1.0/6.0), w),
        (SVector{2,T}(1.0/6.0, 2.0/3.0), w)
    )

    N0, ∇N, wgt, V = _calculate_pf_fields_2d(N, GPs, nodes, p0)

    C2DP(nodes, N0, ∇N[1], ∇N[2], wgt, V, mat, 2)
end

"""
    Quad09P(nodes, p0; mat=Materials.NoMat(), bReduced=false)
Constructs a 9-node quadratic phase-field quadrilateral.
"""
function Quad09P(nodes::Vector{<:Integer}, 
                 p0::Vector{<:AbstractVector{T}};
                 mat=Materials.NoMat(),
                 bReduced::Bool=false) where T<:Number

    poly(ξ) = SVector(0.5*ξ*(ξ-1), (1-ξ^2), 0.5*ξ*(ξ+1))
    
    function N(ξ, η)
      Nξ, Nη = poly(ξ), poly(η)
      SVector(Nξ[1]*Nη[1], Nξ[3]*Nη[1], Nξ[3]*Nη[3], Nξ[1]*Nη[3],
              Nξ[2]*Nη[1], Nξ[3]*Nη[2], Nξ[2]*Nη[3], Nξ[1]*Nη[2],
              Nξ[2]*Nη[2] )
    end

    # Generate 2D Gaussian points from 1D rule
    GP_1d = get_gauss_rule_1d(bReduced)
    GPs = generate_tensor_GPs(GP_1d, 2, T)

    N0, ∇N, wgt, V = _calculate_pf_fields_2d(N, GPs, nodes, p0)

    C2DP(nodes, N0, ∇N[1], ∇N[2], wgt, V, mat, 2)
end

"""
    Tet10P(nodes, p0; mat=Materials.NoMat(), bReduced=false)
Constructs a 10-node quadratic phase-field tetrahedron.
"""
function Tet10P(nodes::Vector{<:Integer}, 
                p0::Vector{<:AbstractVector{T}};
                mat=Materials.NoMat(),
                bReduced::Bool=false) where T<:Number

  function N(ξ, η, ζ)
    λ = (1.0 - ξ - η - ζ, ξ, η, ζ)
        SVector(
            λ[1]*(2λ[1]-1), λ[2]*(2λ[2]-1), λ[3]*(2λ[3]-1), λ[4]*(2λ[4]-1), 
            4λ[1]*λ[2], 4λ[2]*λ[3], 4λ[1]*λ[3], 4λ[1]*λ[4], 4λ[2]*λ[4], 4λ[3]*λ[4]
        )
  end

  # Use 4-point for reduced, 15-point for full integration
  GPs = bReduced ? tet_quadrature_04point() : tet_quadrature_15point()

  N0, ∇N, wgt, V = _calculate_pf_fields_3d(N, GPs, nodes, p0)

  C3DP(nodes, N0, ∇N[1], ∇N[2], ∇N[3], wgt, V, mat, 2)
end

"""
    Hex27P(nodes, p0; mat=Materials.NoMat(), bReduced=false)
Constructs a 27-node quadratic phase-field hexahedron.
"""
function Hex27P(nodes::Vector{<:Integer}, 
                p0::Vector{<:AbstractVector{T}};
                mat=Materials.NoMat(),
                bReduced::Bool=false) where T<:Number

    poly(ξ) = SVector(0.5*ξ*(ξ-1), (1-ξ^2), 0.5*ξ*(ξ+1))
    
    function N_Hex27_Mapped(ξ, η, ζ)
        # Nodal ordering is corrected via map arrays
        Nξ, Nη, Nζ = poly(ξ), poly(η), poly(ζ) 
        vals = MVector{27, typeof(Nξ[1])}(undef)

        @inbounds for idx in 1:27
            i = I_MAP_FINAL[idx]
            j = J_MAP_FINAL[idx]
            k = K_MAP_FINAL[idx]
            vals[idx] = Nξ[i] * Nη[j] * Nζ[k]
        end
        
        return SVector(vals)
    end

    # Generate 3D Gaussian points from 1D rule
    GP_1d = get_gauss_rule_1d(bReduced)
    GPs = generate_tensor_GPs(GP_1d, 3, T)

    N0, ∇N, wgt, V = _calculate_pf_fields_3d(N_Hex27_Mapped, GPs, nodes, p0)

    C3DP(nodes, N0, ∇N[1], ∇N[2], ∇N[3], wgt, V, mat, 2)
end
