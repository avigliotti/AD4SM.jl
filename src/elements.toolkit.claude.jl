# ===========================================================================
# Deformation gradient and displacement gradient
# ===========================================================================

"""
    get∇u(elem, u, ii)

Compute displacement gradient ∇u at Gauss point ii.
"""
@inline function get∇u(elem::CElem{D,P,M,<:Any,N}, u::AbstractArray{T}, ii::Int) where {D,P,M,T,N}
    ∇u = @MMatrix zeros(T, D, D)
    u_mat = SMatrix{D,N}(u[1:D, :])
    
    @inbounds for jj in 1:D, kk in 1:D
        ∇u[jj, kk] = elem.∇N[kk][ii] ⋅ u_mat[jj, :]
    end
    
    return SMatrix{D,D,T}(∇u)
end

"""
    get∇u(elem, u)

Compute volume-averaged displacement gradient.
"""
function get∇u(elem::CElem{D,P,M,T,N}, u::AbstractArray{T}) where {D,P,M,T,N}
    ∇u = @MMatrix zeros(T, D, D)
    
    @inbounds for ii in 1:P
        ∇u .+= elem.wgt[ii] * get∇u(elem, u, ii)
    end
    
    return SMatrix{D,D,T}(∇u / elem.V)
end

get∇u(elems::Array{<:CElem}, u::AbstractArray) =
    [get∇u(elem, u[:, elem.nodes]) for elem in elems]

"""
    getF(elem, u, ii)

Compute deformation gradient F = I + ∇u at Gauss point ii.
"""
@inline getF(elem::CElem, u::AbstractArray, ii::Int) = get∇u(elem, u, ii) + I

"""
    getF(elem, u)

Compute volume-averaged deformation gradient.
"""
@inline getF(elem::CElem, u::AbstractArray) = get∇u(elem, u) + I

getF(elems::Array{<:CElem}, u::AbstractArray) = 
    [getF(elem, u[:, elem.nodes]) for elem in elems]

# ===========================================================================
# Jacobian determinant
# ===========================================================================

"""
    detJ(F::SMatrix{2,2})

Compute determinant of 2×2 matrix (optimized).
"""
@inline detJ(F::SMatrix{2,2,T}) where {T} = F[1,1]F[2,2] - F[1,2]F[2,1]

"""
    detJ(F::SMatrix{3,3})

Compute determinant of 3×3 matrix (Sarrus rule, optimized).
"""
@inline function detJ(F::SMatrix{3,3,T}) where {T}
    # Sarrus rule: faster than cofactor expansion
    F[1,1]*(F[2,2]F[3,3] - F[2,3]F[3,2]) -
    F[1,2]*(F[2,1]F[3,3] - F[2,3]F[3,1]) +
    F[1,3]*(F[2,1]F[3,2] - F[2,2]F[3,1])
end

"""
    detJ(elem, u, ii)

Compute Jacobian at Gauss point ii.
"""
detJ(elem::CElem, u::AbstractArray, ii::Int) = detJ(getF(elem, u, ii))

"""
    detJ(elem, u)

Compute volume-averaged Jacobian.
"""
function detJ(elem::CElem{D,P,<:Any,<:Any,N}, u::AbstractArray{T}) where {D,P,T,N}
    u_mat = SMatrix{D,N,T}(u[1:D, :])
    J = zero(T)
    
    @inbounds for ii in 1:P
        J += elem.wgt[ii] * detJ(elem, u_mat, ii)
    end
    
    return J / elem.V
end

detJ(elems::Array{<:CElem}, u::AbstractArray) = 
    [detJ(elem, u[:, elem.nodes]) for elem in elems]

# ===========================================================================
# Current volume
# ===========================================================================

"""
    getV(elem, u)

Compute current element volume: V = ∫ J dV₀.
"""
@inline function getV(elem::CElem{D,P,M,T,N}, u::AbstractArray{T}) where {D,P,M,T,N}
    V = zero(T)
    
    @inbounds for ii in 1:P
        F = getF(elem, u, ii)
        V += elem.wgt[ii] * detJ(F)
    end
    
    return V
end

"""
    getV(elems, u)

Compute total current volume.
"""
getV(elems::AbstractArray{<:CElem}, u::AbstractArray) = 
    sum(getV(elem, u[:, elem.nodes]) for elem in elems)

# ===========================================================================
# Stress computation
# ===========================================================================

"""
    getσ(elem, u, ii)

Compute Cauchy stress at Gauss point ii: σ = (1/J)·P·Fᵀ.
"""
function getσ(elem::CElem{D,P,M,S,N} where {P,S,M}, 
              u::AbstractArray{T}, ii::Int) where {D,N,T}
    u_mat = SMatrix{D,N,T}(u[1:D, :])
    F = getF(elem, u_mat, ii)
    
    # Compute 1st PK stress via AD
    δϕ = getϕ(adiff.D1(F), elem.mat)
    P = reshape(adiff.grad(δϕ), size(F))
    
    # Convert to Cauchy
    J = detJ(F)
    return (P * F') / J
end

"""
    getσ(elem, u)

Compute volume-averaged Cauchy stress.
"""
function getσ(elem::CElem{D,P,M,S,N} where {S,M}, 
              u::AbstractArray{T}) where {P,D,N,T}
    u_mat = SMatrix{D,N,T}(u[1:D, :])
    σ = @MMatrix zeros(T, D, D)
    
    @inbounds for ii in 1:P
        σ .+= elem.wgt[ii] * getσ(elem, u_mat, ii)
    end
    
    return SMatrix{D,D}(σ / elem.V)
end

"""
    getσ(elems::Vector{<:C3D}, u)

Compute Voigt stress (xx,yy,zz,xy,xz,yz) for 3D elements.
"""
function getσ(elems::Vector{<:C3D}, u::Matrix{<:Real})
    nElems = length(elems)
    idx = [1, 5, 9, 2, 3, 6]  # Voigt indices
    σ = zeros(6, nElems)
    
    @inbounds for (ii, elem) in enumerate(elems)
        σ_full = getσ(elem, u[:, elem.nodes])
        σ[:, ii] .= σ_full[idx]
    end
    
    return σ
end

"""
    getσ(elems::Vector{<:C2D}, u)

Compute Voigt stress (xx,yy,xy) for 2D elements.
"""
function getσ(elems::Vector{<:C2D}, u::Matrix{<:Real})
    nElems = length(elems)
    idx = [1, 4, 2]  # Voigt indices for 2D
    σ = zeros(3, nElems)
    
    @inbounds for (ii, elem) in enumerate(elems)
        σ_full = getσ(elem, u[:, elem.nodes])
        σ[:, ii] .= σ_full[idx]
    end
    
    return σ
end

# ===========================================================================
# First Piola-Kirchhoff stress
# ===========================================================================

"""
    getP(elem, u, ii)

Compute 1st PK stress at Gauss point ii.
"""
function getP(elem::CElem{D,P,M,S,N} where {P,S,M}, 
              u::AbstractArray{T}, ii::Int) where {D,N,T}
    u_mat = SMatrix{D,N,T}(u[1:D, :])
    F = getF(elem, u_mat, ii)
    
    δϕ = getϕ(adiff.D1(F), elem.mat)
    P = reshape(adiff.grad(δϕ), size(F))
    
    return P'
end

"""
    getP(elem, u)

Compute volume-averaged 1st PK stress.
"""
function getP(elem::CElem{D,P,M,S,N} where {S,M}, 
              u::AbstractArray{T}) where {P,D,N,T}
    u_mat = SMatrix{D,N,T}(u[1:D, :])
    P_avg = @MMatrix zeros(T, D, D)
    
    @inbounds for ii in 1:P
        P_avg .+= elem.wgt[ii] * getP(elem, u_mat, ii)
    end
    
    return SMatrix{D,D}(P_avg / elem.V)
end

# ===========================================================================
# Scalar field evaluation
# ===========================================================================

"""
    get_d(elem, d, ii)

Evaluate scalar field d at Gauss point ii.
"""
@inline function get_d(elem::CPElem{D,P,M,T,N}, d::Vector{T}, ii::Int) where {D,P,M,T,N}
    return dot(elem.N[ii], SVector{N}(d[elem.nodes]))
end

"""
    get_∇d(elem, d, ii)

Compute gradient ∇d at Gauss point ii.
"""
@inline function get_∇d(elem::CPElem{D,P,M,T,N}, d::Vector{T}, ii::Int) where {D,P,M,T,N}
    ∇d = @MVector zeros(T, D)
    d_nodal = SVector{N}(d[elem.nodes])
    
    @inbounds for jj in 1:D
        ∇d[jj] = dot(elem.∇N[jj][ii], d_nodal)
    end
    
    return SVector{D,T}(∇d)
end

"""
    get_d_and_∇d(elem, d, ii)

Evaluate scalar field and gradient (d, ∇d) at Gauss point ii.
"""
@inline function get_d_and_∇d(elem::CPElem{D,P,M,T,N}, d::Vector{T}, ii::Int) where {D,P,M,T,N}
    d_nodal = SVector{N}(d[elem.nodes])
    d_val = dot(elem.N[ii], d_nodal)
    
    ∇d = @MVector zeros(T, D)
    @inbounds for jj in 1:D
        ∇d[jj] = dot(elem.∇N[jj][ii], d_nodal)
    end
    
    return d_val, SVector{D,T}(∇d)
end

# ===========================================================================
# Global assembly for Newton-Raphson
# ===========================================================================

"""
    makeϕrKt(Φ, elems, u)

Assemble global energy, residual, and tangent from element contributions.

Uses automatic differentiation (D2 objects) to extract ϕ, r, and Kt.
"""
function makeϕrKt(Φ::Vector{<:adiff.D2}, elems::Vector{<:AbstractContinuumElem}, u)
    N = length(u)
    
    # Count total non-zeros in tangent
    Nt = sum(length(ϕ.g.v) * length(ϕ.g.v) for ϕ in Φ)
    
    # Preallocate sparse matrix triplets
    II = zeros(Int, Nt)
    JJ = zeros(Int, Nt)
    Kt_vals = zeros(Nt)
    
    # Preallocate residual and energy
    r = zeros(N)
    ϕ_total = 0.0
    
    indxs = LinearIndices(u)
    pos = 1
    
    for (ii, elem) in enumerate(elems)
        # Global DOF indices for this element
        idxs = indxs[:, elem.nodes][:]
        n = length(idxs)
        
        # Accumulate energy and residual
        ϕ_total += adiff.val(Φ[ii])
        r[idxs] .+= adiff.grad(Φ[ii])
        
        # Accumulate tangent stiffness (outer product structure)
        nn = n * n
        rng = pos:pos+nn-1
        
        ones_n = ones(n)
        II[rng] .= idxs * ones_n'
        JJ[rng] .= ones_n * idxs'
        Kt_vals[rng] .= adiff.hess(Φ[ii])
        
        pos += nn
    end
    
    # Build sparse matrix and drop zeros
    Kt = dropzeros(sparse(II, JJ, Kt_vals, N, N))
    
    return ϕ_total, r, Kt
end

"""
    makeϕr(Φ, elems, u)

Assemble global energy and residual (first-order AD, no tangent).
"""
function makeϕr(Φ::Vector{<:adiff.Duals}, elems::Vector{<:AbstractContinuumElem}, u)
    indxs = LinearIndices(u)
    r = zeros(length(u))
    ϕ_total = 0.0
    
    for (ii, elem) in enumerate(elems)
        idxs = indxs[:, elem.nodes][:]
        ϕ_total += adiff.val(Φ[ii])
        r[idxs] .+= adiff.grad(Φ[ii])
    end
    
    return ϕ_total, r
end

# ===========================================================================
# Chain rule operators for AD
# ===========================================================================

"""
    ×(ϕ::adiff.D2, F::Array{adiff.D1})

Chain rule for second-order AD: ϕ(F(u)) → ϕ(u).

Propagates gradients and Hessians through deformation gradient.
"""
function ×(ϕ::adiff.D2{N,M,T}, F::AbstractArray{adiff.D1{P,T}}) where {N,M,P,T}
    val = ϕ.v
    grad = adiff.Grad(zeros(T, P))
    hess = adiff.Grad(zeros(T, (P+1)*P÷2))
    
    # First derivative: ∂ϕ/∂u = Σᵢ (∂ϕ/∂Fᵢ)(∂Fᵢ/∂u)
    @inbounds for ii in 1:N
        grad += ϕ.g[ii] * F[ii].g
    end
    
    # Second derivative: off-diagonal terms (i≠j)
    @inbounds for ii in 2:N, jj in 1:ii-1
        hess += ϕ.h[ii,jj] * (F[ii].g * F[jj].g + F[jj].g * F[ii].g)
    end
    
    # Second derivative: diagonal terms
    @inbounds for ii in 1:N
        hess += ϕ.h[ii,ii] * F[ii].g * F[ii].g
    end
    
    return adiff.D2(val, grad, hess)
end

"""
    ×(ϕ::adiff.D1, F::Array{adiff.D1})

Chain rule for first-order AD: ϕ(F(u)) → ϕ(u).
"""
function ×(ϕ::adiff.D1{N,T}, F::AbstractArray{adiff.D1{P,T}}) where {N,P,T}
    val = ϕ.v
    grad = adiff.Grad(zeros(T, P))
    
    @inbounds for ii in 1:N
        grad += ϕ.g[ii] * F[ii].g
    end
    
    return adiff.D1(val, grad)
end

# ===========================================================================
# Invariants of right Cauchy-Green tensor C = F'·F
# ===========================================================================

"""
    I₁(F)

First invariant: I₁ = tr(C) = tr(F'·F) = Σᵢⱼ Fᵢⱼ².
Optimized to avoid forming C explicitly.
"""
@inline I₁(F::Union{SMatrix{3,3}, SVector{9}}) = (F[1]F[1] + F[2]F[2] + F[3]F[3] +
                                                  F[4]F[4] + F[5]F[5] + F[6]F[6] +
                                                  F[7]F[7] + F[8]F[8] + F[9]F[9])

@inline I₁(F::Union{SMatrix{2,2}, SVector{4}}) = F[1]F[1] + F[2]F[2] + F[3]F[3] + F[4]F[4]

"""
    I₂(F)

Second invariant: I₂ = ½[(tr C)² - tr(C²)].
Uses direct formula from components of C.
"""
@inline function I₂(F::Union{SMatrix{3,3}, SVector{9}})
    C = F'F
    # I₂ = C₁₁C₂₂ + C₁₁C₃₃ + C₂₂C₃₃ - C₁₂² - C₁₃² - C₂₃²
    return C[1]C[5] + C[1]C[9] + C[5]C[9] - C[2]C[4] - C[3]C[7] - C[6]C[8]
end

@inline function I₂(F::Union{SMatrix{2,2}, SVector{4}})
    C = F' * F
    return C[1] * C[4] - C[2] * C[3]
end

"""
    Ī₁(F)

Deviatoric first invariant: Ī₁ = I₁/J^(2/3).
Measures pure distortion (shape change).
"""
@inline Ī₁(F::Union{SMatrix{3,3}, SVector{9}}) = I₁(F) / detJ(F)^(2/3)
@inline Ī₁(F::Union{SMatrix{2,2}, SVector{4}}) = I₁(F) / detJ(F)

"""
    Ī₂(F)

Deviatoric second invariant: Ī₂ = I₂/J^(4/3).
"""
@inline Ī₂(F::Union{SMatrix{3,3}, SVector{9}}) = I₂(F) / detJ(F)^(4/3)
@inline Ī₂(F::Union{SMatrix{2,2}, SVector{4}}) = I₂(F) / detJ(F)^2

# ===========================================================================
# Export convenience (if needed)
# ===========================================================================

# These functions are typically not exported but available via module qualification
# If you want them exported, add to the parent module's export list
