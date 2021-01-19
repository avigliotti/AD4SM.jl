__precompile__()

module adiff

using LinearAlgebra
import Base: length, getindex, copy, convert, promote_rule
import Base: +, -, *, /, ^, inv, abs, sqrt, log, exp, zero, sin, cos
import LinearAlgebra: norm, dot, svd, transpose

#  macros 
macro swap(x,y)
  quote
    local tmp = $(esc(x))
    $(esc(x)) = $(esc(y))
    $(esc(y)) = tmp
  end
end
δ(i,j)         = i==j ? Float64(1) : Float64(0)
tupfy(f,N)     = :(@inbounds $(Expr(:tuple, [f(i) for i in 1:N]...)))
tupfy2(f,N)    = :(@inbounds $(Expr(:tuple, [f(i,j) for j in 1:N for i in 1:j]...)))
# structures 
struct Grad{N}
  v::NTuple{N,Float64}
end
struct D1{N} <: Number
  v::Float64
  g::Grad{N}
end
struct D2{N,M} <: Number
  v::Float64
  g::Grad{N}
  h::Grad{M}
end
# constructors
Grad(V...)                          = V .|> Float64 |> Grad 
Grad(V::Array)                      = (V...,) .|> Float64 |> Grad
D2(v::Real)                         = D2(Float64(v),Grad(1.0))
D2{N}(v::Real)    where N           = D2(Float64(v),zero(Grad{N}),zero(Grad{(N+1)N/2}))
D2{N,M}(v::Real)  where {N,M}       = D2(Float64(v),zero(Grad{N}),zero(Grad{(N+1)N/2}))
D2(v::Float64, g::Grad{N}) where N  = D2(v, g, zero(Grad{(N+1)N/2}))
D2(x::Array{<:Real}) = begin
  N     = length(x)
  grad  = init(Grad{N})
  [D2(Float64(x[i]), grad[i], zero(Grad{(N+1)N/2})) for i=1:N]
end
# D1
D1(v::Real)                   = D1(Float64(v),Grad(1.0))
D1{N}(v::Real)    where N     = D1(Float64(v),zero(Grad{N}))
D1(x::Array{<:Real}) = begin
  N     = length(x)
  grad  = init(Grad{N})
  return [D1(Float64(x), grad[i]) for (i,x) in enumerate(x)]
end
D1(x::D2)                     = D1(x.v, x.g)
D2{N,M}(x::D1{N}) where {N,M} = D2(x.v, x.g, zero(Grad{(N+1)N/2})) 
# conversion
# convert(::Type{D2{N,M}}, x::Type{<:Real})     where {N,M} = D2{N,M}(x)
# convert(::Type{D1{N}},   x::Type{<:Real})     where  N    = D1{N}(x)
# promotion
promote_rule(::Type{D2{N,M}}, ::Type{D1{N}})  where {N,M} = D2{N,M}
promote_rule(::Type{D2{N,M}}, ::Type{<:Real}) where {N,M} = D2{N,M}
promote_rule(::Type{D1{N}},   ::Type{<:Real}) where N     = D1{N}
#
@inline Base.@propagate_inbounds getindex(x::Grad{N}, I...) where N = x.v[I...]
@inline Base.@propagate_inbounds getindex(x::Grad{N}, I,J)  where N = ((I>J) && @swap(I,J); x.v[(J-1)J/2+I])
@inline length(x::D2{N,M}) where {N,M} = N
@inline length(x::Grad{N}) where N     = N
@inline @generated zero(::Type{Grad{N}})        where N = :(Grad($(tupfy(i->Float64(0),N))))
@inline @generated init(::Type{Grad{N}})        where N = tupfy(j->:(Grad($(tupfy(i->δ(i,j),N)))),N)
@inline @generated +(x::Grad{N}, y::Grad{N})    where N = :(Grad{N}($(tupfy(i->:(x[$i]+y[$i]),N))))
@inline @generated -(x::Grad{N}, y::Grad{N})    where N = :(Grad{N}($(tupfy(i->:(x[$i]-y[$i]),N))))
@inline @generated -(y::Grad{N})                where N = :(Grad{N}($(tupfy(i->:(-y[$i]),N))))
@inline @generated *(x::Real, y::Grad{N})       where N = :(Grad{N}($(tupfy(i->:(x*y[$i]),N))))
@inline @generated *(y::Grad{N}, x::Real)       where N = :(Grad{N}($(tupfy(i->:(x*y[$i]),N))))
@inline @generated /(y::Grad{N}, x::Real)       where N = :(Grad{N}($(tupfy(i->:(y[$i]/x),N))))
@inline @generated *(x::Grad{N}, y::Grad{N})    where N = :(Grad($(tupfy2((i,j)->:(x[$i]y[$j]),N))))
@inline zero(::Grad{N})              where N = zero(Grad{N})
@inline transpose(x::Grad{N})        where N = x
@inline transpose(x::D1{N})          where N = x
@inline transpose(x::D2{N,M})   where {N,M}  = x
#
# D1 operators,  this can be improved
# dual operators
S = Union{Int64, Float64}
@inline +(x::D1, y::D1)    = D1(x.v+y.v, x.g+y.g)
@inline -(x::D1, y::D1)    = D1(x.v-y.v, x.g-y.g)
@inline -(x::D1)           = D1(-x.v, -x.g)
@inline *(x::D1, y::D1)    = D1(x.v*y.v, x.v*y.g+y.v*x.g)
@inline inv(x::D1)         = D1(1/x.v, (-1/x.v^2)*x.g)
# @inline inv(x::D1)         = D1(inv(x.v), -inv(x.v^2)*x.g)
# @inline /(x::D1, y::D1)    = x*(1/y)
@inline /(x::D1, y::D1)    = x*inv(y)
@inline ^(x::D1, n::S)     = D1(x.v^n, (n*x.v^(n-1))*x.g)
@inline log(x::D1)         = D1(log(x.v),  x.g/x.v)
@inline exp(x::D1)         = D2(exp(x.v), exp(x.v)*x.g)
@inline sin(x::D1)         = D1(sin(x.v),  cos(x.v)*x.g)
@inline cos(x::D1)         = D1(cos(x.v), -sin(x.v)*x.g)
@inline sqrt(x::D1)        = x^0.5
@inline abs(x::D1)         = x.v ≥ 0 ? x : -x
@inline dot(x::Array{D1}, y::Array{D1}) = sum(x.*y)
@inline norm(x::Array{adiff.D1}) = sqrt(dot(x,x))
# data retrieving methods
@inline D1eval(f, x)           = f(D1(x))
@inline Real(x::D1)            = x.v
@inline val(x::D1)             = x.v
@inline grad(x::Real)          = 0  
@inline grad(x::D1{N}) where N = [x.g[i]   for i in 1:N]
@inline hess(x::D1{N}) where N = zeros(N,N)
#
#
# D2 operators
#
# dual operators
S = Union{Int64, Float64}
@inline +(x::D2, y::D2)    = D2(x.v+y.v, x.g+y.g, x.h+y.h)
@inline -(x::D2, y::D2)    = D2(x.v-y.v, x.g-y.g, x.h-y.h)
@inline -(x::D2)           = D2(-x.v, -x.g, -x.h)
@inline *(x::D2, y::D2)    = D2(x.v*y.v, x.v*y.g+y.v*x.g, x.v*y.h+y.v*x.h+x.g*y.g+y.g*x.g)
@inline inv(x::D2)         = D2(1/x.v, (-1/x.v^2)*x.g, (2/x.v^3)*(x.g*x.g) - (1/x.v^2)*x.h)
# @inline inv(x::D2)         = D2(inv(x.v), (-inv(x.v^2))*x.g, (2inv(x.v^3))*(x.g*x.g) - inv(x.v^2)*x.h)
# @inline /(x::D2, y::D2)    = x*(1/y)
@inline /(x::D2, y::D2)    = x*inv(y)
@inline ^(x::D2, n::S)     = D2(x.v^n, (n*x.v^(n-1))*x.g, (n*(n-1)*x.v^(n-2))*(x.g*x.g)+(n*x.v^(n-1))*x.h)
@inline log(x::D2)         = D2(log(x.v),  x.g/x.v,      -(x.g*x.g)/x.v^2 + x.h/x.v) 
@inline exp(x::D2)         = D2(exp(x.v), exp(x.v)*x.g, exp(x.v)*(x.g*x.g) + exp(x.v)*x.h)
@inline sin(x::D2)         = D2(sin(x.v),  cos(x.v)*x.g, -sin(x.v)*(x.g*x.g) + cos(x.v)*x.h) 
@inline cos(x::D2)         = D2(cos(x.v), -sin(x.v)*x.g, -cos(x.v)*(x.g*x.g) - sin(x.v)*x.h) 
@inline sqrt(x::D2)        = x^0.5
@inline abs(x::D2)         = x.v ≥ 0 ? x : -x
@inline dot(x::Array{D2}, y::Array{D2}) = sum(x.*y)
@inline norm(x::Array{adiff.D2{N,M}} where {N,M}) = sqrt(dot(x,x))
# data retrieving methods
@inline D2eval(f::F, x::T) where {F,T} = f(D2(x))
@inline Real(x::D2)                    = x.v
@inline val(x::D2)                     = x.v
@inline grad(x::D2{N,M})   where{N,M}  = [x.g[i]   for i in 1:N]
@inline hess(x::D2{N,M})   where{N,M}  = [x.h[i,j] for i in 1:N, j in 1:N]
end
