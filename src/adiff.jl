__precompile__()

module adiff

using LinearAlgebra
import Base: length, getindex, copy
import Base: +, -, *, /, ^, abs, sqrt, log, zero, sin, cos
import LinearAlgebra: norm, dot, svd

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
struct Hess{N}
  v::NTuple{N,Float64}
end
struct D1{N} <: Real
  v::Float64
  g::Grad{N}
end
struct D2{N,M} <: Real
  v::Float64
  g::Grad{N}
  h::Hess{M}
end
Ders{N} = Union{Grad{N}, Hess{N}}
# constructors
Grad(V...)                    = V .|> Float64 |> Grad 
Hess(V...)                    = V .|> Float64 |> Hess 
Grad(V::Array)                = (V...,) .|> Float64 |> Grad
Hess(V::Array)                = (V...,) .|> Float64 |> Hess
D2(v::Real)                   = D2(Float64(v),Grad(1.0))
D2{N}(v::Real)    where N     = D2(Float64(v),zero(Grad{N}),zero(Hess{N}))
D2{N,M}(v::Real)  where {N,M} = D2(Float64(v),zero(Grad{N}),zero(Hess{N}))
D2(v::Float64, g::Grad{N}) where N = D2(v, g, zero(Hess{N}))
D2(x::Array{<:Real}) = begin
  N     = length(x)
  grad  = init(Grad{N})
  hess  = init(Hess{N})
  return [D2(Float64(x), grad[i], hess[i]) for (i,x) in enumerate(x)]
end
# D1
D1(v::Real)                   = D1(Float64(v),Grad(1.0))
D1{N}(v::Real)    where N     = D1(Float64(v),zero(Grad{N}))
# D1(v::Float64, g::Grad{N}) where N = D1(v, g)
D1(x::Array{<:Real}) = begin
  N     = length(x)
  grad  = init(Grad{N})
  return [D1(Float64(x), grad[i]) for (i,x) in enumerate(x)]
end
#
@inline Base.@propagate_inbounds getindex(x::Grad{N}, I...) where N = x.v[I...]
@inline Base.@propagate_inbounds getindex(x::Hess{N}, I...) where N = x.v[I...]
@inline Base.@propagate_inbounds getindex(x::Hess{N}, I,J)  where N = ((I>J) && @swap(I,J); x.v[(J-1)J/2+I])
@inline length(x::D2{N,M}) where {N,M} = N
@inline length(x::Grad{N}) where N     = N
@inline length(x::Hess{N}) where N     = N
@inline @generated zero(::Type{Grad{N}})             where N = :(Grad($(tupfy(i->Float64(0),N))))
@inline @generated zero(::Type{Hess{N}})             where N = :(Hess($(tupfy(i->Float64(0),(N+1)N/2))))
@inline @generated init(::Type{Grad{N}})             where N = tupfy(j->:(Grad($(tupfy(i->δ(i,j),N)))),N)
@inline @generated init(::Type{Hess{N}})             where N = tupfy(i->zero(Hess{N}),N)
@inline @generated +(x::T, y::T)    where T<:Ders{N} where N = :(T($(tupfy(i->:(x[$i]+y[$i]),N))))
@inline @generated -(x::T, y::T)    where T<:Ders{N} where N = :(T($(tupfy(i->:(x[$i]-y[$i]),N))))
@inline @generated -(y::T)          where T<:Ders{N} where N = :(T($(tupfy(i->:(-y[$i]),N))))
@inline @generated *(x::Real, y::T) where T<:Ders{N} where N = :(T($(tupfy(i->:(x*y[$i]),N))))
@inline @generated *(y::T, x::Real) where T<:Ders{N} where N = :(T($(tupfy(i->:(x*y[$i]),N))))
@inline @generated /(y::T, x::Real) where T<:Ders{N} where N = :(T($(tupfy(i->:(y[$i]/x),N))))
@inline @generated *(x::Grad{N}, y::Grad{N})         where N = :(Hess($(tupfy2((i,j)->:(x[$i]y[$j]),N))))
# D1 operators,  this can be improved
# scale/shift operators
@inline +(x::Real, y::D1)  = D1(x+y.v, y.g)
@inline +(y::D1, x::Real)  = D1(x+y.v, y.g)
@inline -(x::Real, y::D1)  = D1(x-y.v, -y.g)
@inline -(y::D1, x::Real)  = D1(y.v-x, y.g)
@inline *(x::Real, y::D1)  = D1(x*y.v, x*y.g)
@inline *(y::D1, x::Real)  = D1(x*y.v, x*y.g)
@inline /(y::D1, x::Real)  = D1(y.v/x, y.g/x)
@inline /(x::Real, y::D1)  = D1(x/y.v, (-x/y.v^2)*y.g)
# dual operators
S = Union{Int64, Float64}
@inline +(x::D1, y::D1)    = D1(x.v+y.v, x.g+y.g)
@inline -(x::D1, y::D1)    = D1(x.v-y.v, x.g-y.g)
@inline -(x::D1)           = D1(-x.v, -x.g)
@inline *(x::D1, y::D1)    = D1(x.v*y.v, x.v*y.g+y.v*x.g)
@inline /(x::D1, y::D1)    = x*(1/y)
@inline ^(x::D1, n::S)     = D1(x.v^n, (n*x.v^(n-1))*x.g)
@inline log(x::D1)         = D1(log(x.v),  x.g/x.v)
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
# scale/shift operators
@inline +(x::Real, y::D2)  = D2(x+y.v, y.g,   y.h)
@inline +(y::D2, x::Real)  = D2(x+y.v, y.g,   y.h)
@inline -(x::Real, y::D2)  = D2(x-y.v, -y.g,  -y.h)
@inline -(y::D2, x::Real)  = D2(y.v-x, y.g,   y.h)
@inline *(x::Real, y::D2)  = D2(x*y.v, x*y.g, x*y.h)
@inline *(y::D2, x::Real)  = D2(x*y.v, x*y.g, x*y.h)
@inline /(y::D2, x::Real)  = D2(y.v/x, y.g/x, y.h/x)
@inline /(x::Real, y::D2)  = D2(x/y.v, (-x/y.v^2)*y.g, (2x/y.v^3)*(y.g*y.g) - (x/y.v^2)*y.h)
# dual operators
S = Union{Int64, Float64}
@inline +(x::D2, y::D2)    = D2(x.v+y.v, x.g+y.g, x.h+y.h)
@inline -(x::D2, y::D2)    = D2(x.v-y.v, x.g-y.g, x.h-y.h)
@inline -(x::D2)           = D2(-x.v, -x.g, -x.h)
@inline *(x::D2, y::D2)    = D2(x.v*y.v, x.v*y.g+y.v*x.g, x.v*y.h+y.v*x.h+x.g*y.g+y.g*x.g)
@inline /(x::D2, y::D2)    = x*(1/y)
@inline ^(x::D2, n::S)     = D2(x.v^n, (n*x.v^(n-1))*x.g, (n*(n-1)*x.v^(n-2))*(x.g*x.g)+(n*x.v^(n-1))*x.h)
@inline log(x::D2)         = D2(log(x.v),  x.g/x.v,      -(x.g*x.g)/x.v^2 + x.h/x.v) 
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
