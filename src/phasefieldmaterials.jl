# CPElems{P,M,T,I}  = Union{C2DP{P,M,T,I}, C3DP{P,M,T,I}, CAS{P,M,T,I}}
## Phase field
#
struct PhaseField{M,AT}<:Material
  l0::Number 
  Gc::Number 
  mat::M
  n::Number
end

Material = Union{Material,PhaseField}
#
# constructor
#
PhaseField(l0::T,Gc::T,mat::M,n::Number) where {T<:Number,M} = PhaseField{M,:ATn}(l0,Gc,mat,n)

# phase field functions
# withuot history
function getϕ(F::Matrix{D}, d::U, ∇d::Vector{V}, mat::PhaseField{M,:ATn} where M) where {D,U,V}

  l0,Gc,n   = mat.l0,mat.Gc,mat.n
  I1,I1sq   = get1stinvariants(F, mat.mat)
  ν, Es     = mat.mat.ν, mat.mat.E
  λ         = Es*ν/(1+ν)/(1-2ν) 
  μ         = Es/2/(1+ν) 

  γ = d^n + l0^2*(∇d⋅∇d)
  ψ = if I1≥0
    (1-d)^2*(λ/2*I1^2 + μ*I1sq)
  else
    λ/2*I1^2 + (1-d)^2*μ*I1sq
  end

  ψ + Gc/2l0*γ
end
function getϕ(F::Matrix{D}, d::U, ∇d::Vector{V}, mat::PhaseField{NeoHooke,:ATn}) where {D<:Number,U<:Number,V<:Number}

  l0,Gc,n = mat.l0,mat.Gc,mat.n
  C1,K    = mat.mat.C1,mat.mat.K 

  J    =  F[1]F[5]F[9]-F[1]F[6]F[8]-
          F[2]F[4]F[9]+F[2]F[6]F[7]+
          F[3]F[4]F[8]-F[3]F[5]F[7]
  I1   =  F[1]^2+F[2]^2+F[3]^2+
          F[4]^2+F[5]^2+F[6]^2+
          F[7]^2+F[8]^2+F[9]^2

  γ   = d^n + l0^2*(∇d⋅∇d)
  ϕel = C1*(I1-3-2log(J)) + K*(J-1)^2
  ψ   = J≥1 ? (1-d)^2 * ϕel : ϕel 

  ψ + Gc/2l0*γ
end
# deviatoric/hydrostatic
function getϕ(F::Matrix{D}, d::U, ∇d::Vector{V}, mat::PhaseField{M,:DHn} where M) where {D,U,V}

  l0,Gc,n   = mat.l0,mat.Gc,mat.n
  I1,ϵd     = gethyddevdecomp(F, mat.mat)
  ν, Es     = mat.mat.ν, mat.mat.E
  λ, μ      = Es*ν/(1+ν)/(1-2ν), Es/2/(1+ν)  
  K         = λ+μ/3

  γ = d^n + l0^2*(∇d⋅∇d)
  ψ = if I1≥0
    (1-d)^2*(K/2*I1^2 + μ*ϵd)
  else
    K/2*I1^2 + (1-d)^2*μ*ϵd
  end

  ψ + Gc/2l0*γ
end
#
# with history
function getϕ(F::Matrix{D}, d::U, ∇d::Vector{V}, mat::PhaseField{M,:ATn} where M, ϕmax::Tuple{Number,Number}) where {D,U,V}

  l0,Gc,n   = mat.l0,mat.Gc,mat.n
  I1,I1sq   = get1stinvariants(F, mat.mat)
  ν, Es     = mat.mat.ν, mat.mat.E
  λ         = Es*ν/(1+ν)/(1-2ν) 
  μ         = Es/2/(1+ν) 

  γ = d^n + l0^2*(∇d⋅∇d)
  ψ = if I1≥0
    ϕmax = max(λ/2*I1^2 + μ*I1sq, ϕmax[1]), ϕmax[2]
    (1-d)^2*ϕmax[1]
  else
    ϕmax = ϕmax[1], max(μ*I1sq, ϕmax[2])
    λ/2*I1^2 + (1-d)^2*ϕmax[2]
  end

  ψ + Gc/2l0*γ, ϕmax
end
function getϕ(F::Matrix{D}, d::U, ∇d::Vector{V}, mat::PhaseField{NeoHooke,:ATn}, ϕmax::Number) where {D<:Number,U<:Number,V<:Number}

  l0,Gc,n = mat.l0,mat.Gc,mat.n
  C1,K    = mat.mat.C1,mat.mat.K 

  J    =  F[1]F[5]F[9]-F[1]F[6]F[8]-
          F[2]F[4]F[9]+F[2]F[6]F[7]+
          F[3]F[4]F[8]-F[3]F[5]F[7]
  I1   =  F[1]^2+F[2]^2+F[3]^2+
          F[4]^2+F[5]^2+F[6]^2+
          F[7]^2+F[8]^2+F[9]^2

  γ = d^n + l0^2*(∇d⋅∇d)
  ψ = if J≥1
    ϕmax = max(ϕmax, C1*(I1-3-2log(J)) + K*(J-1)^2)
    (1-d)^2 * ϕmax 
  else
    C1*(I1-3-2log(J)) + K*(J-1)^2
  end

  ψ + Gc/2l0*γ, ϕmax
end
# deviatoric/hydrostatic
function getϕ(F::Matrix{D}, d::U, ∇d::Vector{V}, mat::PhaseField{M,:DHn} where M, ϕmax::Tuple{Number,Number}) where {D,U,V}

  l0,Gc,n   = mat.l0,mat.Gc,mat.n
  I1,ϵd     = gethyddevdecomp(F, mat.mat)
  ν, Es     = mat.mat.ν, mat.mat.E
  λ, μ      = Es*ν/(1+ν)/(1-2ν), Es/2/(1+ν)  
  K         = λ+μ/3

  γ = d^n + l0^2*(∇d⋅∇d)
  ψ = if I1≥0
    ϕmax = max(K/2*I1^2 + μ*ϵd, ϕmax[1]), ϕmax[2]
    (1-d)^2*ϕmax[1]
  else
    ϕmax = ϕmax[1], max(μ*ϵd, ϕmax[2])
    λ/2*I1^2 + (1-d)^2*ϕmax[2]
  end

  ψ + Gc/2l0*γ, ϕmax
end
#
