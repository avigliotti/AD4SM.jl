export makeϕrKt
# 
#
include("./elasticelements.1stord.jl")
include("./elasticelements.2ndord.jl")
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


