__precompile__()

module adiff

using LinearAlgebra, StaticArrays
import Base: @propagate_inbounds, length, getindex, copy, convert, promote_rule
import Base: +, -, *, /, ^, inv, abs, sqrt, log, exp, zero, conj
import Base: >, <, ≥, ≤, ==
import Base: sin, cos, sinh, cosh, tanh, acos, min, max, eps
import LinearAlgebra: norm, dot, transpose, svdvals

# Macros
macro swap(x, y)
    quote
        local tmp = $(esc(x))
        $(esc(x)) = $(esc(y))
        $(esc(y)) = tmp
    end
end

# Kronecker delta function
δ(i, j, T)           = i == j ? one(T) : zero(T)

# Helper functions to generate tuple expressions
tupfy(f, N)          = :(@inbounds $(Expr(:tuple, [f(i) for i in 1:N]...)))
tupfy2(f, N)         = :(@inbounds $(Expr(:tuple, [f(i, j) for j in 1:N for i in 1:j]...)))

# Structures
struct Grad{N, T}
    v::NTuple{N, T}
end

struct D1{N, T} <: Number
    v::T
    g::Grad{N, T}
end

struct D2{N, M, T} <: Number
    v::T
    g::Grad{N, T}
    h::Grad{M, T}
end

Duals               = Union{D1, D2}

# Constructors
@inline @generated zero(::Type{Grad{N, T}}) where {N, T} = :(Grad($(tupfy(i -> zero(T), N))))
@inline zero(::Grad{N, T}) where {N, T} = zero(Grad{N, T})

Grad(v::T) where T<:Number               = Grad(tuple(v))
Grad(v::AbstractArray{T}) where T<:Number = Grad(tuple(v...))
D2(v::T) where T<:Number                 = D2(v, Grad(one(T)), Grad(zero(T)))
D1(v::T) where T<:Number                 = D1(v, Grad(one(T)))
D2{N, M, T}(v::Number) where {N, M, T}   = D2{N, M, T}(T(v), zero(Grad{N, T}), zero(Grad{M, T}))
D1{N, T}(v::Number) where {N, T}         = D1{N, T}(T(v), zero(Grad{N, T}))
D2(v::T, g::Grad{N, T}) where {N, T}     = D2(v, g, zero(Grad{(N+1)*N÷2, T}))

D2(x::AbstractArray{T}) where T<:Number = begin
    N    = length(x)
    grad = init(Grad{N, T})
    [D2(x, grad[ii], zero(Grad{(N+1)*N÷2, T})) for (ii, x) in enumerate(x)]
end

D1(x::AbstractArray{T}) where T<:Number = begin
    N    = length(x)
    grad = init(Grad{N, T})
    [D1(x, grad[ii]) for (ii, x) in enumerate(x)]
end

# Conversion functions
D1(x::D2)                              = D1(x.v, x.g)
D2(x::D1{N, T}) where {N, T}           = D2{N, (N+1)*N÷2, T}(x.v, x.g, zero(Grad{(N+1)*N÷2, T}))
convert(::Type{<:Real}, x::D1)         = x.v
convert(::Type{<:Real}, x::D2)         = x.v
length(::Grad{N}) where N              = N

# Promotion rules
promote_rule(::Type{D2{N, M, T}}, ::Type{D1{N}}) where {N, M, T} = D2{N, M, T}
promote_rule(::Type{D2{N, M, T}}, ::Type{<:Real}) where {N, M, T} = D2{N, M, T}
promote_rule(::Type{D1{N, T}}, ::Type{<:Real}) where {N, T} = D1{N, T}

# Indexing
@inline @propagate_inbounds getindex(x::Grad{N}, I...) where N = x.v[I...]
@inline @propagate_inbounds getindex(x::Grad{N}, I, J) where N = ((I > J) && @swap(I, J); x.v[(J-1)*J÷2+I])
@inline @generated init(::Type{Grad{N, T}}) where {N, T} = tupfy(j -> :(Grad($(tupfy(i -> δ(i, j, T), N)))), N)

# Arithmetic operations on Grad
@inline @generated +(x::Grad{N, T}, y::Grad{N, T}) where {N, T} = :(Grad{N, T}($(tupfy(i -> :(x[$i]+y[$i]), N))))
@inline @generated -(x::Grad{N, T}, y::Grad{N, T}) where {N, T} = :(Grad{N, T}($(tupfy(i -> :(x[$i]-y[$i]), N))))
@inline @generated -(y::Grad{N, T}) where {N, T}                = :(Grad{N, T}($(tupfy(i -> :( -y[$i]), N))))
@inline @generated *(x::Number, y::Grad{N, T}) where {N, T}     = :(Grad{N, T}($(tupfy(i -> :(x*y[$i]), N))))
@inline @generated *(y::Grad{N, T}, x::Number) where {N, T}     = :(Grad{N, T}($(tupfy(i -> :(x*y[$i]), N))))
@inline @generated /(y::Grad{N, T}, x::Number) where {N, T}     = :(Grad{N, T}($(tupfy(i -> :(y[$i]/x), N))))
@inline @generated *(x::Grad{N, T}, y::Grad{N, T}) where {N, T} = :(Grad{(N+1)*N÷2, T}($(tupfy2((i, j) -> :(x[$i]*y[$j]), N))))

# Relational operators for Duals
<(x::Duals, y::Number)                  = x.v < y
>(x::Duals, y::Number)                  = x.v > y
≤(x::Duals, y::Number)                  = x.v ≤ y
≥(x::Duals, y::Number)                  = x.v ≥ y
<(y::Number, x::Duals)                  = y < x.v
>(y::Number, x::Duals)                  = y > x.v
≤(y::Number, x::Duals)                  = y ≤ x.v
≥(y::Number, x::Duals)                  = y ≥ x.v
<(y::Duals, x::Duals)                   = y.v < x.v
>(y::Duals, x::Duals)                   = y.v > x.v
≤(y::Duals, x::Duals)                   = y.v ≤ x.v
≥(y::Duals, x::Duals)                   = y.v ≥ x.v

# Arithmetic operations for D1
@inline +(x::D1, y::D1)                 = D1(x.v+y.v, x.g+y.g)
@inline -(x::D1, y::D1)                 = D1(x.v-y.v, x.g-y.g)
@inline -(x::D1)                        = D1(-x.v, -x.g)
@inline *(x::D1, y::D1)                 = D1(x.v*y.v, x.v*y.g+y.v*x.g)
@inline inv(x::D1)                      = D1(1/x.v, (-1/x.v^2)*x.g)
@inline /(x::D1, y::D1)                 = x*inv(y)
@inline ^(x::T, n::Number) where T<:D1  = n == 0 ? one(T) : n == 1 ? x : D1(x.v^n, (n*x.v^(n-1))*x.g)
@inline ^(x::T, n::Integer) where T<:D1 = n == 0 ? one(T) : n == 1 ? x : D1(x.v^n, (n*x.v^(n-1))*x.g)
@inline log(x::D1)                      = D1(log(x.v), x.g/x.v)
@inline exp(x::D1)                      = D1(exp(x.v), exp(x.v)*x.g)
@inline sin(x::D1)                      = D1(sin(x.v), cos(x.v)*x.g)
@inline cos(x::D1)                      = D1(cos(x.v), -sin(x.v)*x.g)
@inline sinh(x::D1)                     = D1(sinh(x.v), cosh(x.v)*x.g)
@inline cosh(x::D1)                     = D1(cosh(x.v), sinh(x.v)*x.g)
@inline tanh(x::D1)                     = D1(tanh(x.v), (1-tanh(x.v)^2)*x.g)
@inline acos(x::D1)                     = D1(acos(x.v), -1/sqrt(1 - x^2)*x.g)
@inline sqrt(x::D1)                     = x^0.5
@inline abs(x::D1)                      = x.v ≥ 0 ? x : -x
@inline conj(x::D1{N, <:Real}) where N  = x
@inline norm(x::AbstractArray{<:D1})    = sqrt(dot(x, x))

# Arithmetic operations for D2
@inline +(x::D2, y::D2)                 = D2(x.v+y.v, x.g+y.g, x.h+y.h)
@inline -(x::D2, y::D2)                 = D2(x.v-y.v, x.g-y.g, x.h-y.h)
@inline -(x::D2)                        = D2(-x.v, -x.g, -x.h)
@inline *(x::D2, y::D2)                 = D2(x.v*y.v, x.v*y.g+y.v*x.g, x.v*y.h+y.v*x.h+x.g*y.g+y.g*x.g)
@inline inv(x::D2)                      = D2(1/x.v, (-1/x.v^2)*x.g, (2/x.v^3)*(x.g*x.g)-(1/x.v^2)*x.h)
@inline /(x::D2, y::D2)                 = x*inv(y)
@inline ^(x::T, n::Number) where T<:D2  = n == 0 ? one(T) : n == 1 ? x : D2(x.v^n, (n*x.v^(n-1))*x.g, (n*(n-1)*x.v^(n-2))*(x.g*x.g)+(n*x.v^(n-1))*x.h)
@inline ^(x::T, n::Integer) where T<:D2 = n == 0 ? one(T) : n == 1 ? x : D2(x.v^n, (n*x.v^(n-1))*x.g, (n*(n-1)*x.v^(n-2))*(x.g*x.g)+(n*x.v^(n-1))*x.h)
@inline log(x::D2)                      = D2(log(x.v),  x.g/x.v, -(x.g*x.g)/x.v^2+x.h/x.v)
@inline exp(x::D2)                      = D2(exp(x.v),  exp(x.v)*x.g, exp(x.v)*(x.g*x.g)+exp(x.v)*x.h)
@inline sin(x::D2)                      = D2(sin(x.v),  cos(x.v)*x.g, -sin(x.v)*(x.g*x.g)+cos(x.v)*x.h)
@inline cos(x::D2)                      = D2(cos(x.v), -sin(x.v)*x.g, -cos(x.v)*(x.g*x.g)-sin(x.v)*x.h)
@inline sinh(x::D2)                     = D2(sinh(x.v), cosh(x.v)*x.g, sinh(x.v)*(x.g*x.g)+cosh(x.v)*x.h)
@inline cosh(x::D2)                     = D2(cosh(x.v), sinh(x.v)*x.g, cosh(x.v)*(x.g*x.g)+sinh(x.v)*x.h)
@inline tanh(x::D2)                     = D2(tanh(x.v), (1-tanh(x.v)^2)*x.g, 2*(tanh(x.v)^2-1)*tanh(x.v)*(x.g*x.g)+(1-tanh(x.v)^2)*x.h)
@inline acos(x::D2)                     = D2(acos(x.v), -1/sqrt(1 - x^2)*x.g, -x/(1 - x^2)^(3/2)*(x.g*x.g)-1/sqrt(1 - x^2)*x.h)
@inline sqrt(x::D2)                     = x^0.5
@inline abs(x::D2)                      = x.v ≥ 0 ? x : -x
@inline conj(x::D2{N, M, <:Real}) where {N, M} = x
@inline norm(x::AbstractArray{<:D2})    = sqrt(dot(x, x))

# Data retrieving methods
@inline D1eval(f, x)                    = f(D1(x))
@inline D2eval(f, x)                    = f(D2(x))
@inline val(x::Duals)                   = x.v
@inline val(U::AbstractArray{Duals})    = [u.v for u in U]
@inline grad(x::Real)                   = 0
@inline grad(x::Duals)                  = [g for g in x.g.v]
@inline hess(x::D1{N, T}) where {N, T}  = zeros(T, N, N)
@inline hess(x::D2{N, M, T}) where {N, M, T} = [x.h[i, j] for i in 1:N, j in 1:N]


@inline min(x::Duals, y::Duals)   = x.v < y.v ? x : y
@inline max(x::Duals, y::Duals)   = x.v > y.v ? x : y
@inline eps(x::T) where T<:Duals  = T(eps(x.v))

function svdvals(F::SMatrix{3,3,T}; tol=1e-24, ϵη=1e-30) where T <: Duals
  # Smooth positive part: ~ max(x,0), but analytic
  @inline smoothplus(x, ϵ) = (x + sqrt(x*x + ϵ*ϵ)) / 2

  # C = F'F, expanded explicitly
  f11,f12,f13 = F[1,1],F[1,2],F[1,3]
  f21,f22,f23 = F[2,1],F[2,2],F[2,3]
  f31,f32,f33 = F[3,1],F[3,2],F[3,3]

  c11 = f11*f11 + f21*f21 + f31*f31
  c22 = f12*f12 + f22*f22 + f32*f32
  c33 = f13*f13 + f23*f23 + f33*f33
  c12 = f11*f12 + f21*f22 + f31*f32
  c13 = f11*f13 + f21*f23 + f31*f33
  c23 = f12*f13 + f22*f23 + f32*f33

  # mean eigenvalue
  m = (c11 + c22 + c33) / 3

  # deviatoric part B = C - m I
  b11 = c11 - m
  b22 = c22 - m
  b33 = c33 - m
  b12 = c12
  b13 = c13
  b23 = c23

  # p² = tr(B²)/6
  trB2 = b11*b11 + b22*b22 + b33*b33 + 2*(b12*b12 + b13*b13 + b23*b23)
  p2 = trB2 / 6

  η1 = m
  η2 = m
  η3 = m

  if p2 > tol
    p = sqrt(p2)
    invp = inv(p)

    # R = B/p
    r11 = b11*invp
    r22 = b22*invp
    r33 = b33*invp
    r12 = b12*invp
    r13 = b13*invp
    r23 = b23*invp

    detR = r11*(r22*r33 - r23*r23) -
    r12*(r12*r33 - r13*r23) +
    r13*(r12*r23 - r13*r22)

    # exact theory gives r in [-1,1]
    r = detR / 2

    # numerical safeguard
    # r = min(one(T), max(-one(T), r))
    r = min(one(T)-eps(r.v), max(-one(T)+eps(r.v), r))

    ϕ = acos(r) / 3

    two_p = 2p
    η1 = m + two_p*cos(ϕ)
    η2 = m + two_p*cos(ϕ + 2T(pi)/3)
    η3 = m + two_p*cos(ϕ + 4T(pi)/3)
  end

  # protect sqrt from tiny negative roundoff
  η1 = smoothplus(η1, T(ϵη))
  η2 = smoothplus(η2, T(ϵη))
  η3 = smoothplus(η3, T(ϵη))

  return @SVector [sqrt(η1), sqrt(η2), sqrt(η3)]
end

end
