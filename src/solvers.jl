module Solvers

using LinearAlgebra, Printf
using Distributed, SparseArrays
 
using ..adiff, ..Materials, ..Elements 

import ..Elements.makeϕrKt

export makeϕrKt, ConstEq, solve, solvestep!

p = Int64(nworkers())
function setp(x)
  global p = Int64(x)
end
# structure for constraint eqs
struct ConstEq
  func
  iDoFs::Array{Int64}
  D::Type
end
ConstEq(func, iDoFs) = ConstEq(func, iDoFs, adiff.D2)
#
function makeϕrKt(eqns::Array{ConstEq}, u::Array{Float64}, λ::Array{Float64})

  function split(N::Int64, p::Int64)
    n    = Int64(floor(N/p))
    nEls = ones(Int64, p)*n
    nEls[1:N-p*n] .+= 1

    slice = [ range(sum(nEls[1:ii-1])+1, length=nEls[ii])
             for ii in 1:p]
  end

  nEqs   = length(eqns)
  if p==1 || nEqs<=p
    (veqs, reqs, Keqs) = makeϕrKt(eqns, u, λ, 1:nEqs)
  else
    nDoFs  = length(u)
    veqs   = zeros(nEqs)
    reqs   = spzeros(nDoFs, nEqs)
    Keqs   = spzeros(nDoFs, nDoFs)
    chunks = split(nEqs, Elements.p)
    procs  = [@spawn makeϕrKt(eqns, u, λ, chunk)  for chunk in chunks]

    for ii in 1:p 
      retval  = fetch(procs[ii])
      veqs .+= retval[1]
      reqs .+= retval[2]
      Keqs .+= retval[3]
    end
  end
  (veqs, reqs, Keqs)
end
function makeϕrKt(eqns::Array{ConstEq}, u::Array{Float64}, λ::Array{Float64}, chunk)

  nEqs  = length(eqns)
  nDoFs = length(u)

  veqs  = zeros(nEqs)
  reqs  = spzeros(nDoFs, nEqs)
  Keqs  = spzeros(nDoFs, nDoFs)

  for ii ∈ chunk 
    eqn               =  eqns[ii]
    iDoFs             =  eqn.iDoFs
    ϕ                 =  eqn.func(eqn.D(u[iDoFs]))
    veqs[ii]          =  adiff.val(ϕ)
    reqs[iDoFs,ii]    =  adiff.grad(ϕ)
    Keqs[iDoFs,iDoFs] += λ[ii]adiff.hess(ϕ)
  end
  (veqs, reqs, Keqs)  
end

"""
    secs2hms(secs)

Convert seconds to HH:MM:SS formatted string.

# Arguments
- `secs`: Time in seconds

# Returns
Formatted time string
"""
function secs2hms(secs)
  h, r = divrem(secs, 3600)
  m, s = divrem(r, 60)
  @sprintf("%02i:%02i:%02i",h, m, s)
end
#
# solvers 
#
function solve(elems, u;
               N          = 11,
               LF         = range(1e-4, stop=1, length=N), 
               eqns       = [],
               λ          = zeros(length(eqns)),
               ifree      = isnan.(u),
               fe         = zeros(size(u)),
               becho      = false,
               dTol       = 1e-5,
               dTolu      = dTol,
               dTole      = 1e2dTol,
               dNoise     = 1e-12,
               maxiter    = 11,
               bechoi     = false,
               ballus     = true,
               bpredict   = true,
               maxupdt    = NaN)

  N     = length(LF)
  t0    = Base.time_ns()
  beqns = length(eqns)>0
  if ballus;    allu = [];  end

  fnew  = copy(fe)
  icnst = .!ifree
  unew  = copy(u)
  unew[ifree] .= 0
  uold  = zeros(size(unew))
  λnew  = copy(λ)

  for (ii,LF) in enumerate(LF)
    unew[icnst] .= u[icnst]*LF 
    fnew[ifree] .= fe[ifree]*LF 
    lastu       = copy(unew)
    lastλ       = copy(λnew)
    T           = @elapsed (bfailed, normr, iter) = 
    try 
      solvestep!(elems, uold, unew, ifree, 
                 eqns      = eqns,
                 λ         = λnew,
                 fe        = fnew, 
                 dTole     = dTole,
                 dTolu     = dTolu,
                 dNoise    = dNoise,
                 maxiter   = maxiter,
                 becho     = bechoi,
                 bpredict  = bpredict,
                 maxupdt   = maxupdt)
    catch e
      error_msg = sprint(showerror, e)
      st        = sprint((io,v) -> show(io, "text/plain", v), 
                         stacktrace(catch_backtrace()))
      @warn "\n\n$error_msg\n\n$st" 
      (true, Inf, 0)
    end
    if bfailed 
      @printf("\n!! failed at LF: %.3f, with normr/dTol: %.3e\n", LF, normr/dTol)
      unew = lastu
      λnew = lastλ
      break
    else
      uold[:] .= unew[:]
      if ballus
        if beqns
          push!(allu, (copy(unew), copy(fnew), copy(λnew)))
        else
          push!(allu, (copy(unew), copy(fnew)))
        end
      end
      becho  && @printf("step %3i/%i, LF=%.3f, done in %2i iter, after %.2f sec., ETA %s\n",
                           ii,N,LF,iter,T,secs2hms(T*(N-ii)))
      bechoi && println()
    end
    becho && flush(stdout)
  end
  becho && @printf("completed in %s \n",secs2hms((time_ns()-t0)÷1e9)); flush(stdout)

  ballus ? allu : unew
end  
function solvestep!(elems, uold, unew, bfreeu;
                    fe        = zeros(length(unew)),
                    eqns      = [],
                    λ         = zeros(length(eqns)),
                    dTol      = 1e-5,
                    dTolu     = dTol,
                    dTole     = 1e2dTol,
                    dNoise    = 1e-12,
                    maxiter   = 11,
                    becho     = false,
                    bpredict  = true,
                    maxupdt   = NaN)

  ifreeu    = findall(bfreeu[:])
  icnstu    = findall(.!bfreeu[:])

  nEqs      = length(eqns)
  nfreeu    = length(ifreeu)
  ncnstu    = length(icnstu)
  nDoFs     = nfreeu + nEqs
  iius      = 1:nfreeu
  iieqs     = nfreeu .+ (1:nEqs)

  bdone     = false
  bfailed   = false
  iter      = 0
  normupdt  = 0
  normre    = NaN
  oldupdt   = zeros(nDoFs)
  updt      = zeros(nDoFs)
  if nEqs != 0 
    H       = spzeros(nDoFs,nDoFs)
  end

  # predictor step
  if bpredict
    deltat = @elapsed begin
      Δucnst    = unew[icnstu]-uold[icnstu]
      # (Φ,fi,Kt) = makeϕrKt(elems, uold)
      Φ,fi,Kt = let 
        δϕ = [ getϕ(elem, adiff.D2(uold[:,elem.nodes])) for elem in elems ]
        makeϕrKt(δϕ, elems, uold)
      end

      if nEqs == 0
        res              = fi[ifreeu]-fe[ifreeu]
        res[:]         .-= Kt[ifreeu,icnstu]*Δucnst
        # updt[:]          = qr(Kt[ifreeu,ifreeu])\res
        updt[:]          = lu(Kt[ifreeu,ifreeu])\res
        unew[ifreeu]    .= uold[ifreeu] .+ updt 
        normupdt         = maximum(abs.(updt))
      else
        (vEqs,rEqs,KEqs) = makeϕrKt(eqns, uold, λ)
        resu             = fi[ifreeu]-fe[ifreeu]-rEqs[ifreeu,:]*λ
        resu[:]        .-= (Kt[ifreeu,icnstu]-KEqs[ifreeu,icnstu])*Δucnst
        rese             = -vEqs
        res              = vcat(resu, rese)

        H[iius,iius]     = Kt[ifreeu,ifreeu]-KEqs[ifreeu,ifreeu]
        H[iius,iieqs]    = -rEqs[ifreeu,:]
        H[iieqs,iius]    = transpose(H[iius,iieqs])
        # H[iieqs,iieqs]   = spdiagm(0=>dNoise*randn(nEqs))
        H               += spdiagm(0=>dNoise*randn(nDoFs))
        # updt[:]          = qr(H)\res
        updt[:]          = lu(H)\res
        unew[ifreeu]    .= uold[ifreeu] .+ updt[iius]
        λ              .-= updt[iieqs]
        normupdt         = maximum(abs.(updt[iius]))
      end
    end
    becho && @printf("predictor step done in %.2f sec., ", deltat)
    becho && @printf("with normupdt: %.2e, starting corrector loop\n", normupdt); flush(stdout)
  else
    unew[ifreeu] .= uold[ifreeu]
  end

  # corrector loop
  while !bdone & !bfailed 
    global normru
    oldupdt   = copy(updt)
    tic       = Base.time_ns()
    # (Φ,fi,Kt) = makeϕrKt(elems, unew)
    Φ,fi,Kt = let 
      δϕ = [ getϕ(elem, adiff.D2(unew[:,elem.nodes])) for elem in elems ]
      makeϕrKt(δϕ, elems, unew)
    end

    if nEqs == 0
      res    = fe[ifreeu]-fi[ifreeu]      
      norm0  = ncnstu > 0     ? norm(fi[icnstu])/ncnstu : 0
      normru = norm0  > dTolu ? norm(res)/nfreeu/norm0  : norm(res)/nfreeu
      bdone  = (normru<dTolu)
    else
      (vEqs,rEqs,KEqs) = makeϕrKt(eqns, unew, λ)
      resu   = fi[ifreeu]-fe[ifreeu]-rEqs[ifreeu,:]*λ
      rese   = -vEqs
      res    = -vcat(resu, rese)
      norm0  = ncnstu > 0     ? norm(fi[icnstu])/ncnstu : 0
      normru = norm0  > dTolu ? norm(res)/nfreeu/norm0  : norm(res)/nfreeu
      normre = maximum(abs.(rese))
      bdone  = (normru<dTolu) && (normre<dTole)
    end

    if bdone
      fe[:]   = nEqs==0 ? fi[:] : fi[:]-rEqs*λ
    elseif iter < maxiter
      if nEqs == 0
        # updt[:]         = qr(Kt[ifreeu,ifreeu])\res
        updt[:]         = lu(Kt[ifreeu,ifreeu])\res
        normupdt        = maximum(abs.(updt))
        if !isnan(maxupdt)
          if normupdt > maxupdt  
            updt      .*= (maxupdt/normupdt)
            normupdt    = maximum(abs.(updt))
          end
        end
        unew[ifreeu]  .+= updt
      else
        H[iius,iius]    = Kt[ifreeu,ifreeu]-KEqs[ifreeu,ifreeu]
        H[iius,iieqs]   = -rEqs[ifreeu,:]
        H[iieqs,iius]   = transpose(H[iius,iieqs])
        #H[iieqs,iieqs]  = spdiagm(0=>dNoise*randn(nEqs))
        H               += spdiagm(0=>dNoise*randn(nDoFs))
        # updt[:]         = qr(H)\res
        # updt[:]         = cholesky(Symmetric(H))\res
        updt[:]         = lu(H)\res
        normupdt        = maximum(abs.(updt[iius]))
        if !isnan(maxupdt)
          if normupdt > maxupdt  
            updt      .*= (maxupdt/normupdt)
            normupdt    = maximum(abs.(updt[iius]))
          end
        end
        unew[ifreeu]  .+= updt[iius]
        λ             .+= updt[iieqs]
      end              
    else
      bfailed = true
    end    

    if becho 
      if (bdone | bfailed) 
        @printf("iter: %2i, norm0: %.2e, normru: %.2e, normre: %.2e\n", 
                iter, norm0, normru, normre)
      else
        @printf("iter: %2i, norm0: %.2e, normru: %.2e, normre: %.2e, normupdt: %.2e, α: %6.3f\n", 
                iter, norm0, normru, normre, normupdt, 
                oldupdt⋅updt/norm(updt)/norm(oldupdt))
      end
      flush(stdout)
    end
    iter  += 1
  end

  (bfailed, normru, iter)
end

end
