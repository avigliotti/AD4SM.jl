__precompile__()

# new_adiff branch

module adiff

using LinearAlgebra
import Base: length, getindex, copy, convert, promote_rule
import Base: +, -, *, /, ^, inv, abs, sqrt, log, exp, zero
import Base: >, <, ≥, ≤, ==
import Base: sin, cos, sinh, cosh, tanh
import LinearAlgebra: norm, dot, svd, transpose, svdvals

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
struct D1{N}   <: Number
  v::Float64
  g::Grad{N}
end
struct D2{N,M} <: Number
  v::Float64
  g::Grad{N}
  h::Grad{M}
end
Duals = Union{D1, D2}
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
  [D2(Float64(x), grad[i], zero(Grad{(N+1)N/2})) for (i,x) in enumerate(x)]
end
D1(v::Real)                   = D1(Float64(v),Grad(1.0))
D1{N}(v::Real)    where N     = D1(Float64(v),zero(Grad{N}))
D1(x::Array{<:Real}) = begin
  N     = length(x)
  grad  = init(Grad{N})
  return [D1(Float64(x), grad[i]) for (i,x) in enumerate(x)]
end
D1(x::D2)                     = D1(x.v, x.g)
D2{N,M}(x::D1{N}) where {N,M} = D2(x.v, x.g, zero(Grad{(N+1)N/2})) 
# promotion
promote_rule(::Type{D2{N,M}}, ::Type{D1{N}})  where {N,M} = D2{N,M}
promote_rule(::Type{D2{N,M}}, ::Type{<:Real}) where {N,M} = D2{N,M}
promote_rule(::Type{D1{N}},   ::Type{<:Real}) where N     = D1{N}
#
@inline Base.@propagate_inbounds getindex(x::Grad{N}, I...) where N = x.v[I...]
@inline Base.@propagate_inbounds getindex(x::Grad{N}, I,J)  where N = ((I>J) && @swap(I,J); x.v[(J-1)J/2+I])
@inline @generated zero(::Type{Grad{N}})        where N = :(Grad($(tupfy(i->Float64(0),N))))
@inline @generated init(::Type{Grad{N}})        where N = tupfy(j->:(Grad($(tupfy(i->δ(i,j),N)))),N)
@inline @generated +(x::Grad{N}, y::Grad{N})    where N = :(Grad{N}($(tupfy(i->:(x[$i]+y[$i]),N))))
@inline @generated -(x::Grad{N}, y::Grad{N})    where N = :(Grad{N}($(tupfy(i->:(x[$i]-y[$i]),N))))
@inline @generated -(y::Grad{N})                where N = :(Grad{N}($(tupfy(i->:(-y[$i]),N))))
@inline @generated *(x::Real, y::Grad{N})       where N = :(Grad{N}($(tupfy(i->:(x*y[$i]),N))))
@inline @generated *(y::Grad{N}, x::Real)       where N = :(Grad{N}($(tupfy(i->:(x*y[$i]),N))))
@inline @generated /(y::Grad{N}, x::Real)       where N = :(Grad{N}($(tupfy(i->:(y[$i]/x),N))))
@inline @generated *(x::Grad{N}, y::Grad{N})    where N = :(Grad($(tupfy2((i,j)->:(x[$i]y[$j]),N))))
@inline length(x::D2{N,M})    where {N,M} = N
@inline length(x::Grad{N})    where N     = N
@inline zero(::Grad{N})       where N     = zero(Grad{N})
@inline transpose(x::Grad{N}) where N     = x
@inline transpose(x::D1{N})   where N     = x
@inline transpose(x::D2{N,M}) where {N,M} = x
#
# relational operators
#
S = Union{Int64, Float64}
<(x::Duals,y::S) = x.v<y
>(x::Duals,y::S) = x.v>y
≤(x::Duals,y::S) = x.v≤y
≥(x::Duals,y::S) = x.v≥y
<(y::S,x::Duals) = y<x.v
>(y::S,x::Duals) = y>x.v
≤(y::S,x::Duals) = y≤x.v
≥(y::S,x::Duals) = y≥x.v
#
# D1 operators,  this can be improved
# 
@inline +(x::D1, y::D1)         = D1(x.v+y.v, x.g+y.g)
@inline -(x::D1, y::D1)         = D1(x.v-y.v, x.g-y.g)
@inline -(x::D1)                = D1(-x.v, -x.g)
@inline *(x::D1, y::D1)         = D1(x.v*y.v, x.v*y.g+y.v*x.g)
@inline inv(x::D1)              = D1(1/x.v, (-1/x.v^2)*x.g)
@inline /(x::D1, y::D1)         = x*inv(y)
@inline ^(x::D1, n::S)          = D1(x.v^n, (n*x.v^(n-1))*x.g)
@inline log(x::D1)              = D1(log(x.v), x.g/x.v)
@inline exp(x::D1)              = D1(exp(x.v), exp(x.v)*x.g)
@inline sin(x::D1)              = D1(sin(x.v), cos(x.v)*x.g)
@inline cos(x::D1)              = D1(cos(x.v), -sin(x.v)*x.g)
# @inline tanh(x::D1)             = D1(tanh(x.v), (1-tanh(x.v)^2)*x.g)
@inline sinh(x::D1)             = (1-exp(-2x))/2exp(-x)
@inline cosh(x::D1)             = (1+exp(-2x))/2exp(-x)
@inline tanh(x::D1)             = (exp(2x)-1)/(exp(2x)+1)
@inline sqrt(x::D1)             = x^0.5
@inline abs(x::D1)              = x.v ≥ 0 ? x : -x
@inline dot(x::Array{D1}, y::Array{D1}) = sum(x.*y)
@inline norm(x::Array{D1})      = sqrt(dot(x,x))
# data retrieving methods
@inline D1eval(f, x)            = f(D1(x))
@inline Real(x::D1)             = Real(x.v)
@inline val(x::D1)              = x.v
@inline grad(x::Real)           = 0  
@inline grad(x::D1{N}) where N  = [x.g[i]   for i in 1:N]
@inline hess(x::D1{N}) where N  = zeros(N,N)
@inline val(U::Array{D1{N}})   where N = [u.v for u in U]
@inline grad(U::Array{D1{N}})  where N = [[u.g[ii] for u in U] for ii=1:N]
@inline hess(U::Array{D1{N}})  where N = [[u.h[ii] for u in U] for ii=1:M]
# D2 operators
@inline +(x::D2, y::D2)    = D2(x.v+y.v, x.g+y.g, x.h+y.h)
@inline -(x::D2, y::D2)    = D2(x.v-y.v, x.g-y.g, x.h-y.h)
@inline -(x::D2)           = D2(-x.v, -x.g, -x.h)
@inline *(x::D2, y::D2)    = D2(x.v*y.v, x.v*y.g+y.v*x.g, x.v*y.h+y.v*x.h+x.g*y.g+y.g*x.g)
@inline inv(x::D2)         = D2(1/x.v, (-1/x.v^2)*x.g, (2/x.v^3)*(x.g*x.g) - (1/x.v^2)*x.h)
@inline /(x::D2, y::D2)    = x*inv(y)
@inline ^(x::D2, n::S)     = D2(x.v^n, (n*x.v^(n-1))*x.g, (n*(n-1)*x.v^(n-2))*(x.g*x.g)+(n*x.v^(n-1))*x.h)
@inline log(x::D2)         = D2(log(x.v), x.g/x.v,      -(x.g*x.g)/x.v^2 + x.h/x.v) 
@inline exp(x::D2)         = D2(exp(x.v), exp(x.v)*x.g, exp(x.v)*(x.g*x.g) + exp(x.v)*x.h)
@inline sin(x::D2)         = D2(sin(x.v), cos(x.v)*x.g, -sin(x.v)*(x.g*x.g) + cos(x.v)*x.h) 
# @inline tanh(x::D2)        = D2(tanh(x.v), (1-tanh(x.v)^2)*x.g, 2(tanh(x.v)^2-1)*tanh(x.v)*(x.g*x.g)+ (1-tanh(x.v)^2)*x.h)
@inline sinh(x::D2)        = (1-exp(-2x))/2exp(-x)
@inline cosh(x::D2)        = (1+exp(-2x))/2exp(-x)
@inline tanh(x::D2)        = (exp(2x)-1)/(exp(2x)+1)
@inline cos(x::D2)         = D2(cos(x.v), -sin(x.v)*x.g, -cos(x.v)*(x.g*x.g) - sin(x.v)*x.h) 
@inline sqrt(x::D2)        = x^0.5
@inline abs(x::D2)         = x.v ≥ 0 ? x : -x
@inline dot(x::Array{D2{N,M}}, y::Array{D2{N,M}}) where {N,M} = sum(x.*y)
@inline norm(x::Array{D2{N,M}} where {N,M})  = sqrt(dot(x,x))
#
# data retrieving methods
@inline D2eval(f::F, x::T)       where {F,T} = f(D2(x))
@inline Real(x::D2)                          = Real(x.v)
@inline val(x::D2)                           = x.v
@inline grad(x::D2{N,M})         where{N,M}  = [x.g[i]   for i in 1:N]
@inline hess(x::D2{N,M})         where{N,M}  = [x.h[i,j] for i in 1:N, j in 1:N]

@inline val(U::Array{D2{N,M}})   where {N,M} = [u.v for u in U]
@inline grad(U::Array{D2{N,M}})  where {N,M} = [[u.g[ii] for u in U] for ii=1:N]
@inline hess(U::Array{D2{N,M}})  where {N,M} = [[u.h[ii] for u in U] for ii=1:M]

@inline val(A::Symmetric{D2{N,M}, Array{D2{N,M},2}})  where {N,M} = [u.v for u in A]
@inline grad(U::Symmetric{D2{N,M}, Array{D2{N,M},2}}) where {N,M} = [[u.g[ii] for u in U] for ii=1:N]
@inline hess(U::Symmetric{D2{N,M}, Array{D2{N,M},2}}) where {N,M} = [[u.h[ii] for u in U] for ii=1:M]

#function svdvals(B::Symmetric{D2{N,M},Array{D2{N,M},2}} where {N,M};
#                 ϵ=1e-9)
function svdvals(B::Array{D2{N,M},2} where {N,M};
                 ϵ=1e-9)
  # svdB = svd(Symmetric(val.(B)))
  svdB = svd(val(B))
  U, S = svdB.U, svdB.S
  N    = length(S)  

  dσ(i,h,k) = U[h,i]U[k,i]
  function d2σ(i,h,k,l,m)
    y = 0
    for j = 1:N
      ΔS = S[i]-S[j]
      (abs(ΔS)>ϵ) &&
      (y += (dσ(i,k,m)dσ(j,h,l)+dσ(i,k,l)dσ(j,h,m))/ΔS)
      # (y += U[k,i]*U[h,j]*(U[l,j]*U[m,i]+U[l,i]*U[m,j])/ΔS)
    end
    y
  end

  σv = S
  σg = [ begin
          x = zero(B[1].g)
          for h=1:N, k=1:N
            x += dσ(i,h,k)*B[h,k].g
          end
          x
        end for i in 1:N]
  σh = [ begin
          x = zero(B[1].h)
          for h=1:N, k=1:N
            for l=1:N, m=1:N
              x += d2σ(i,h,k,l,m)*(B[h,k].g*B[l,m].g)
            end
            x += dσ(i,h,k)*B[h,k].h
          end
          x
        end for i in 1:N]

  return [D2(σv[i], σg[i], σh[i]) for i in 1:N]
end

end
