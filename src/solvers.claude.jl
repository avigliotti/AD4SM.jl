module Solvers

using LinearAlgebra, Printf
using Distributed, SparseArrays
 
using ..adiff, ..Materials, ..Elements 

import ..Elements.makeϕrKt

export makeϕrKt, ConstEq, SolverParams, solve, solvestep!, setp

# Global worker count
p = Int64(nworkers())
setp(x) = global p = Int64(x)

# Constraint equation structure
struct ConstEq
  func
  iDoFs::Vector{Int64}
  D::Type
end
ConstEq(func, iDoFs) = ConstEq(func, iDoFs, adiff.D2)

# Solver parameters
struct SolverParams
  dTolu::Float64
  dTole::Float64
  dNoise::Float64
  maxiter::Int
  becho::Bool
  bpredict::Bool
  maxupdt::Float64
end

function SolverParams(;
  dTol = 1e-5,
  dTolu = dTol,
  dTole = 1e2 * dTol,
  dNoise = 1e-12,
  maxiter = 11,
  becho = false,
  bpredict = true,
  maxupdt = NaN)
  
  SolverParams(dTolu, dTole, dNoise, maxiter, becho, bpredict, maxupdt)
end

# Abstract problem state
abstract type AbstractProblemState{N} end

# Unconstrained problem state (only Dirichlet/Neumann BCs)
mutable struct UnconstrainedState{N} <: AbstractProblemState{N}
  u::Array{Float64,N}
  f::Array{Float64,N}
  bfree::BitArray{N}
  ifree::Vector{Int64}
  icnst::Vector{Int64}
end

function UnconstrainedState(u::Array{Float64,N}, f::Array{Float64,N}, 
                            bfree::BitArray{N}) where N
  ifree = findall(vec(bfree))
  icnst = findall(vec(.!bfree))
  UnconstrainedState{N}(u, f, bfree, ifree, icnst)
end

# Constrained problem state (includes Lagrange multipliers)
mutable struct ConstrainedState{N} <: AbstractProblemState{N}
  u::Array{Float64,N}
  f::Array{Float64,N}
  λ::Vector{Float64}
  eqns::Vector{ConstEq}
  bfree::BitArray{N}
  ifree::Vector{Int64}
  icnst::Vector{Int64}
end

function ConstrainedState(u::Array{Float64,N}, f::Array{Float64,N}, 
                          bfree::BitArray{N}, eqns::Vector{ConstEq},
                          λ::Vector{Float64}) where N
  ifree = findall(vec(bfree))
  icnst = findall(vec(.!bfree))
  ConstrainedState{N}(u, f, λ, eqns, bfree, ifree, icnst)
end

# Factory function to create appropriate state type
function create_state(u, f, bfree, eqns, λ)
  isempty(eqns) ? UnconstrainedState(u, f, bfree) : 
                  ConstrainedState(u, f, bfree, eqns, λ)
end

# Utility: split work among processors
function split_work(N::Int64, np::Int64)
  n = div(N, np)
  nEls = fill(n, np)
  nEls[1:(N - np*n)] .+= 1
  [range(sum(nEls[1:i-1]) + 1, length=nEls[i]) for i in 1:np]
end

# Parallel constraint equation evaluation
function makeϕrKt(eqns::Vector{ConstEq}, u::Vector{Float64}, λ::Vector{Float64})
  nEqs = length(eqns)
  
  if p == 1 || nEqs <= p
    return makeϕrKt(eqns, u, λ, 1:nEqs)
  end
  
  nDoFs = length(u)
  chunks = split_work(nEqs, Elements.p)
  futures = [@spawn makeϕrKt(eqns, u, λ, chunk) for chunk in chunks]
  
  veqs = zeros(nEqs)
  reqs = spzeros(nDoFs, nEqs)
  Keqs = spzeros(nDoFs, nDoFs)
  
  for fut in futures
    v, r, K = fetch(fut)
    veqs .+= v
    reqs .+= r
    Keqs .+= K
  end
  
  (veqs, reqs, Keqs)
end

# Serial constraint equation evaluation
function makeϕrKt(eqns::Vector{ConstEq}, u::Vector{Float64}, λ::Vector{Float64}, chunk)
  nEqs = length(eqns)
  nDoFs = length(u)
  
  veqs = zeros(nEqs)
  reqs = spzeros(nDoFs, nEqs)
  Keqs = spzeros(nDoFs, nDoFs)
  
  for i ∈ chunk 
    eqn = eqns[i]
    iDoFs = eqn.iDoFs
    ϕ = eqn.func(eqn.D(u[iDoFs]))
    veqs[i] = adiff.val(ϕ)
    reqs[iDoFs, i] = adiff.grad(ϕ)
    Keqs[iDoFs, iDoFs] += λ[i] * adiff.hess(ϕ)
  end
  
  (veqs, reqs, Keqs)  
end

# Time formatting
secs2hms(secs) = @sprintf("%02i:%02i:%02i", divrem(secs, 3600)..., divrem(secs % 3600, 60)[2])

# Main solver interface
function solve(elems, u;
               N = 11,
               LF = range(1e-4, stop=1, length=N), 
               eqns = ConstEq[],
               λ = zeros(length(eqns)),
               ifree = isnan.(u),
               fe = zeros(size(u)),
               becho = false,
               bechoi = false,
               ballus = true,
               kwargs...)

  params = SolverParams(; becho=bechoi, kwargs...)
  N = length(LF)
  t0 = time_ns()
  allus = ballus ? [] : nothing
  
  # Create BitArray for free DoFs
  bfree = BitArray(ifree)
  bcnst = .!bfree
  
  # Initialize state (create once, update in place)
  unew = copy(u)
  unew[bfree] .= 0
  fnew = copy(fe)
  λnew = copy(λ)
  state = create_state(unew, fnew, bfree, eqns, λnew)
  
  uold = zeros(size(u))

  for (i, lf) in enumerate(LF)
    # Update boundary conditions and forces
    state.u[bcnst] .= u[bcnst] .* lf
    state.f[bfree] .= fe[bfree] .* lf
    
    # Store backup for failure recovery
    lastu = copy(state.u)
    lastλ = has_constraints(state) ? copy(state.λ) : nothing
    
    T = @elapsed (failed, normr, iter) = try_solve_step!(elems, uold, state, params)
    
    if failed 
      @printf("\n!! failed at LF: %.3f, with normr/dTol: %.3e\n", lf, normr/params.dTolu)
      state.u .= lastu
      has_constraints(state) && (state.λ .= lastλ)
      break
    end
    
    uold .= state.u
    ballus && push_result!(allus, state)
    
    becho && @printf("step %3i/%i, LF=%.3f, done in %2i iter, after %.2f sec., ETA %s\n",
                     i, N, lf, iter, T, secs2hms(T*(N-i)))
    bechoi && println()
    becho && flush(stdout)
  end
  
  becho && @printf("completed in %s\n", secs2hms((time_ns()-t0) ÷ 1e9))
  flush(stdout)

  ballus ? allus : state.u
end

# Helper functions for type checking
has_constraints(::UnconstrainedState) = false
has_constraints(::ConstrainedState) = true

# Result storage dispatch
push_result!(allus, state::UnconstrainedState) = push!(allus, (copy(state.u), copy(state.f)))
push_result!(allus, state::ConstrainedState)   = push!(allus, (copy(state.u), copy(state.f), copy(state.λ)))

# Wrapped solve step with error handling
function try_solve_step!(elems, uold, state::AbstractProblemState, params::SolverParams)
  try 
    solvestep!(elems, uold, state, params)
  catch e
    @warn sprint(showerror, e) * "\n" * 
          sprint((io,v) -> show(io, "text/plain", v), stacktrace(catch_backtrace()))
    (true, Inf, 0)
  end
end

# Unconstrained problem solver - dispatch on UnconstrainedState
function solvestep!(elems, uold, state::UnconstrainedState, params::SolverParams)
  nfree = length(state.ifree)
  ncnst = length(state.icnst)
  
  # Convert arrays to vectors for linear algebra operations
  uvec = vec(state.u)
  fvec = vec(state.f)
  uoldvec = vec(uold)
  
  # Predictor
  if params.bpredict
    Δucnst = uvec[state.icnst] - uoldvec[state.icnst]
    Φ, fi, Kt = eval_elements(elems, uold)
    
    res = fi[state.ifree] - fvec[state.ifree] - Kt[state.ifree, state.icnst] * Δucnst
    updt = lu(Kt[state.ifree, state.ifree]) \ res
    uvec[state.ifree] .= uoldvec[state.ifree] .+ updt
    normupdt = maximum(abs.(updt))
    
    params.becho && @printf("predictor: normupdt=%.2e\n", normupdt)
  else
    uvec[state.ifree] .= uoldvec[state.ifree]
    updt = zeros(nfree)
  end
  
  # Corrector
  iter = 0
  normru = NaN
  
  while iter < params.maxiter
    Φ, fi, Kt = eval_elements(elems, state.u)
    res = fvec[state.ifree] - fi[state.ifree]
    
    norm0 = ncnst > 0 ? norm(fi[state.icnst]) / ncnst : 0
    normru = norm0 > params.dTolu ? norm(res) / nfree / norm0 : norm(res) / nfree
    
    if normru < params.dTolu
      fvec[:] .= fi
      params.becho && @printf("iter: %2i, norm0: %.2e, normru: %.2e\n", iter, norm0, normru)
      return (false, normru, iter)
    end
    
    updt = lu(Kt[state.ifree, state.ifree]) \ res
    normupdt = clamp_update!(updt, params.maxupdt)
    uvec[state.ifree] .+= updt
    
    params.becho && @printf("iter: %2i, norm0: %.2e, normru: %.2e, normupdt: %.2e\n", 
                            iter, norm0, normru, normupdt)
    
    iter += 1
  end
  
  (true, normru, iter)
end

# Constrained problem solver - dispatch on ConstrainedState
function solvestep!(elems, uold, state::ConstrainedState, params::SolverParams)
  nfree = length(state.ifree)
  ncnst = length(state.icnst)
  nEqs = length(state.eqns)
  nDoFs = nfree + nEqs
  
  iius = 1:nfree
  iieqs = nfree .+ (1:nEqs)
  
  # Convert arrays to vectors for linear algebra operations
  uvec = vec(state.u)
  fvec = vec(state.f)
  uoldvec = vec(uold)
  
  H = spzeros(nDoFs, nDoFs)
  
  # Predictor
  if params.bpredict
    Δucnst = uvec[state.icnst] - uoldvec[state.icnst]
    Φ, fi, Kt = eval_elements(elems, uold)
    vEqs, rEqs, KEqs = makeϕrKt(state.eqns, uoldvec, state.λ)
    
    resu = fi[state.ifree] - fvec[state.ifree] - rEqs[state.ifree, :] * state.λ - 
           (Kt[state.ifree, state.icnst] - KEqs[state.ifree, state.icnst]) * Δucnst
    rese = -vEqs
    res = vcat(resu, rese)
    
    assemble_hessian!(H, Kt, KEqs, rEqs, state.ifree, iius, iieqs, nDoFs, params.dNoise)
    updt = lu(H) \ res
    
    uvec[state.ifree] .= uoldvec[state.ifree] .+ updt[iius]
    state.λ .-= updt[iieqs]
    normupdt = maximum(abs.(updt[iius]))
    
    params.becho && @printf("predictor: normupdt=%.2e\n", normupdt)
  else
    uvec[state.ifree] .= uoldvec[state.ifree]
    updt = zeros(nDoFs)
  end
  
  # Corrector
  iter = 0
  normru = normre = NaN
  
  while iter < params.maxiter
    Φ, fi, Kt = eval_elements(elems, state.u)
    vEqs, rEqs, KEqs = makeϕrKt(state.eqns, uvec, state.λ)
    
    resu = fi[state.ifree] - fvec[state.ifree] - rEqs[state.ifree, :] * state.λ
    rese = -vEqs
    res = -vcat(resu, rese)
    
    norm0 = ncnst > 0 ? norm(fi[state.icnst]) / ncnst : 0
    normru = norm0 > params.dTolu ? norm(res) / nfree / norm0 : norm(res) / nfree
    normre = maximum(abs.(rese))
    
    if normru < params.dTolu && normre < params.dTole
      fvec[:] .= fi - rEqs * state.λ
      params.becho && @printf("iter: %2i, norm0: %.2e, normru: %.2e, normre: %.2e\n", 
                              iter, norm0, normru, normre)
      return (false, normru, iter)
    end
    
    assemble_hessian!(H, Kt, KEqs, rEqs, state.ifree, iius, iieqs, nDoFs, params.dNoise)
    updt = lu(H) \ res
    normupdt = clamp_update!(updt[iius], params.maxupdt)
    
    uvec[state.ifree] .+= updt[iius]
    state.λ .+= updt[iieqs]
    
    params.becho && @printf("iter: %2i, norm0: %.2e, normru: %.2e, normre: %.2e, normupdt: %.2e\n", 
                            iter, norm0, normru, normre, normupdt)
    
    iter += 1
  end
  
  (true, normru, iter)
end

# Helper: evaluate element contributions
function eval_elements(elems, u)
  δϕ = [getϕ(elem, adiff.D2(u[:, elem.nodes])) for elem in elems]
  makeϕrKt(δϕ, elems, u)
end

# Helper: assemble Hessian for constrained problem
function assemble_hessian!(H, Kt, KEqs, rEqs, ifree, iius, iieqs, nDoFs, dNoise)
  H[iius, iius] = Kt[ifree, ifree] - KEqs[ifree, ifree]
  H[iius, iieqs] = -rEqs[ifree, :]
  H[iieqs, iius] = transpose(H[iius, iieqs])
  H .+= spdiagm(0 => dNoise * randn(nDoFs))
end

# Helper: clamp update magnitude
function clamp_update!(updt, maxupdt)
  normupdt = maximum(abs.(updt))
  if !isnan(maxupdt) && normupdt > maxupdt
    updt .*= maxupdt / normupdt
    return maxupdt
  end
  normupdt
end

end
