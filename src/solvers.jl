module Solvers

using LinearAlgebra, Printf
using Distributed, SparseArrays
using ProgressMeter, Dates#, StatsBase
#
# using .adiff, .Materials, .Elements 
using ..adiff, ..Materials, ..Elements 

import ..Materials.getϕ

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
# elastic energy evaluation functions for elements
function getϕ(elem::Elements.Rod,  u::Matrix{<:Number})

  l   = norm(elem.r0+u[:,2]-u[:,1])
  F11 = l/elem.l0
  elem.A*elem.l0*getϕ(F11, elem.mat)    

end
function getϕ(elem::Elements.Beam, u::Matrix{<:Number})

  L, r0, t, w = elem.L, elem.r0, elem.t, elem.w
  T    = [r0[1] r0[2]; -r0[2] r0[1]]
  u0   = vcat(T*u[1:2,1], u[3,1], T*u[1:2,2], u[3,2])
  u0_x = (u0[4]-u0[1])/L

  ϕ  = 0
  for (r,wr) in elem.lgwx
    x, dx     = r*L, wr*L

    v0_x  = (-6x/L^2 + 6x^2/L^3)u0[2] +
    (1 - 4x/L + 3x^2/L^2)u0[3] +
    (6x/L^2 - 6x^2/L^3)u0[5] +
    (-2x/L + 3x^2/L^2)u0[6]

    v0_xx = (-6/L^2 + 12x/L^3)u0[2] + 
    (-4/L + 6x/L^2)u0[3] + 
    (6/L^2 - 12x/L^3)u0[5] + 
    (-2/L + 6x/L^2)u0[6]

    for (s,ws) in elem.lgwy
      y, dy  = s*elem.t, ws*elem.t

      dV   = dx*dy*elem.w
      C11 = (1+u0_x-v0_xx*y)^2 + v0_x^2
      ϕ   += getϕ(C11, elem.mat)*dV
    end
  end
  return ϕ
end
function getϕ(elem::Elements.CElems{P}, u::Array{U,2})  where {U, P}
  ϕ = zero(U)
  for ii=1:P
    F  = getF(elem,u,ii)
    ϕ += elem.wgt[ii]getϕ(F,elem.mat)
  end 
  ϕ
end
getϕu(elem::Elements.C2D, u::Matrix) = getϕ(elem, adiff.D2(u))
getϕu(elem::Elements.CAS, u::Matrix) = getϕ(elem, adiff.D2(u))
function getϕu(elem::Elements.C3D{P}, u0::Matrix{T})  where {P,T}

  u, v, w = u0[1:3:end], u0[2:3:end], u0[3:3:end]
  N       = lastindex(u0)  
  wgt     = elem.wgt
  val     = zero(T)
  grad    = zeros(T,N)
  hess    = zeros(T,(N+1)N÷2)
  δF      = zeros(T,N,9)

  @inbounds for ii=1:P
    Nx,Ny,Nz    = elem.Nx[ii],elem.Ny[ii],elem.Nz[ii]
    δF[1:3:N,1] = δF[2:3:N,2] = δF[3:3:N,3] = Nx
    δF[1:3:N,4] = δF[2:3:N,5] = δF[3:3:N,6] = Ny
    δF[1:3:N,7] = δF[2:3:N,8] = δF[3:3:N,9] = Nz

    F    = [Nx⋅u Ny⋅u Nz⋅u;
            Nx⋅v Ny⋅v Nz⋅v;
            Nx⋅w Ny⋅w Nz⋅w ] + I
    ϕ    = getϕ(adiff.D2(F), elem.mat)::adiff.D2{9, 45, T}
    val += wgt[ii]ϕ.v
    @inbounds for jj=1:9,i1=1:N
      grad[i1] += wgt[ii]*ϕ.g[jj]*δF[i1,jj]
      @inbounds  for kk=1:9,i2=1:i1
        hess[(i1-1)i1÷2+i2] += wgt[ii]*ϕ.h[jj,kk]*δF[i1,jj]*δF[i2,kk]
      end   
    end
  end

  adiff.D2(val, adiff.Grad(grad), adiff.Grad(hess))
end
#=
function getϕ(elem::T where T<:CElems, u::Matrix{<:Number})
  M = length(elem.wgt)
  if isa(u[1], adiff.D2) 
    ϕ = sum([begin
               F = getF(elem,u,ii)
               ϕ = getϕ(adiff.D2(getfield.(F,:v)),elem.mat)
               elem.wgt[ii]cross(ϕ,F)
             end  for ii in 1:M])
  else
    ϕ = sum([elem.wgt[ii]getϕ(getF(elem,u,ii), elem.mat) for ii in 1:M])
  end 
end
function cross(ϕ, F)
  N = length(F)
  g = sum([ϕ.g[ii]F[ii].g for ii in 1:N])
  h = sum([0.5ϕ.h[ii,jj]*(F[ii].g*F[jj].g+F[jj].g*F[ii].g) for jj=1:N for ii=1:N])
  adiff.D2(ϕ.v, g, h) 
end
=#
# elastic energy evaluation functions for models (Array of elements)
function getϕ(elems::Vector{<:Elements.CElems}, 
              u::Array{T,2}) where T
  nElems = length(elems)
  N      = length(u[:,elems[1].nodes])
  M      = (N+1)N÷2

  Φ = Vector{adiff.D2{N,M,T}}(undef, nElems)
  Threads.@threads for ii=1:nElems
    Φ[ii] = getϕu(elems[ii], u[:,elems[ii].nodes])
  end
  makeϕrKt(Φ, elems, u)
end 
function makeϕrKt(Φ::Array{adiff.D2{N,M,T}, 1} where {N,M}, elems, u) where T
  N  = length(u) 
  Nt = 0
  for ϕ in Φ
    Nt += length(ϕ.g.v)*length(ϕ.g.v)
  end

  I  = zeros(T, Nt)
  J  = zeros(T, Nt)  
  Kt = zeros(T, Nt)  
  r  = zeros(T, N)
  ϕ  = 0
  indxs  = LinearIndices(u)

  N1 = 1
  for (ii,elem) in enumerate(elems)    
    idxii     = indxs[:, elem.nodes][:]    
    ϕ        += adiff.val(Φ[ii]) 
    r[idxii] += adiff.grad(Φ[ii]) 
    nii       = length(idxii)
    Nii       = nii*nii
    oneii     = ones(nii)
    idd       = N1:N1+Nii-1
    I[idd]    = idxii * transpose(oneii)
    J[idd]    = oneii * transpose(idxii)
    Kt[idd]   = adiff.hess(Φ[ii])
    N1       += Nii
  end

  ϕ, r, sparse(I,J,Kt,N,N)
end
function getϕ(eqns::Array{ConstEq}, u::Array{Float64}, λ::Array{Float64})

  nEqs   = length(eqns)
  if p==1 || nEqs<=p
    (veqs, reqs, Keqs) = getϕ(eqns, u, λ, 1:nEqs)
  else
    nDoFs  = length(u)
    veqs   = zeros(nEqs)
    reqs   = spzeros(nDoFs, nEqs)
    Keqs   = spzeros(nDoFs, nDoFs)
    chunks = split(nEqs, Elements.p)
    procs  = [@spawn getϕ(eqns, u, λ, chunk)  for chunk in chunks]

    for ii in 1:p 
      retval  = fetch(procs[ii])
      veqs .+= retval[1]
      reqs .+= retval[2]
      Keqs .+= retval[3]
    end
  end
  (veqs, reqs, Keqs)
end
function getϕ(eqns::Array{ConstEq}, u::Array{Float64}, λ::Array{Float64}, chunk)

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
#=
function renumber!(nodes, elems)

  unodes   = sort(unique(vcat([elem.nodes for elem in elems]...)))
  el_nodes = hcat([elem.nodes for elem in elems]...)
  shift    = maximum(unodes) + 1

  el_nodes .+= shift
  for (ii, nodeid) in enumerate(unodes)
    nodeid += shift
    el_nodes[el_nodes .== nodeid] .= ii
  end
  nodes = nodes[unodes]

  for (ii,elem) in enumerate(elems)
    elem.nodes[:] .= el_nodes[:,ii]
  end
  (nodes, elems)
end
=#
function split(N::Int64, p::Int64)
  n    = Int64(floor(N/p))
  nEls = ones(Int64, p)*n
  nEls[1:N-p*n] .+= 1

  slice = [ range(sum(nEls[1:ii-1])+1, length=nEls[ii])
           for ii in 1:p]
end
# solvers 
function solve(elems, u;
               N          = 11,
               LF         = range(1e-4, stop=1, length=N), 
               eqns       = [],
               λ          = zeros(length(eqns)),
               ifree      = isnan.(u),
               fe         = zeros(size(u)),
               bprogress  = false,
               becho      = false,
               dTol       = 1e-5,
               dTolu      = dTol,
               dTole      = 1e2dTol,
               dNoise     = 1e-12,
               maxiter    = 11,
               bechoi     = false,
               bprogressi = false,
               ballus     = true,
               bpredict   = true,
               maxupdt    = NaN)

  N     = length(LF)
  t0    = Base.time_ns()
  beqns = length(eqns)>0
  if bprogress; p    = ProgressMeter.Progress(length(LF)); end
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
                 bprogress = bprogressi,
                 becho     = bechoi,
                 bpredict  = bpredict,
                 maxupdt   = maxupdt)
    catch
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
      bprogress && ProgressMeter.next!(p)
      becho     && @printf("step %3i/%i, LF=%.3f, done in %2i iter, after %.2f sec.\n",
                           ii,N,LF,iter,T)
    end
    becho && flush(stdout)
  end
  becho && @printf("completed in %s\n",(Base.time_ns()-t0)÷1e9|>
                   Dates.Second|>Dates.CompoundPeriod|>Dates.canonicalize)
  becho && flush(stdout)

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
                    bprogress = false,
                    bpredict  = true,
                    maxupdt   = NaN)

  if bprogress
    p = ProgressMeter.ProgressThresh(dTolu)
  end

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
      (Φ,fi,Kt) = getϕ(elems, uold)
      if nEqs == 0
        res              = fi[ifreeu]-fe[ifreeu]
        res[:]         .-= Kt[ifreeu,icnstu]*Δucnst
        # updt[:]          = qr(Kt[ifreeu,ifreeu])\res
        updt[:]          = lu(Kt[ifreeu,ifreeu])\res
        unew[ifreeu]    .= uold[ifreeu] .+ updt 
        normupdt         = maximum(abs.(updt))
      else
        (vEqs,rEqs,KEqs) = getϕ(eqns, uold, λ)
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
    becho && @printf("\npredictor step done in %.2f sec., ", deltat)
    becho && @printf("with normupdt: %.2e, starting corrector loop\n", normupdt); flush(stdout)
  else
    unew[ifreeu] .= uold[ifreeu]
  end
  # corrector loop
  while !bdone & !bfailed 
    global normru
    oldupdt = copy(updt)
    tic     = Base.time_ns()
    (Φ,fi,Kt) = getϕ(elems, unew)

    if nEqs == 0
      res    = fe[ifreeu]-fi[ifreeu]      
      norm0  = ncnstu > 0     ? norm(fi[icnstu])/ncnstu : 0
      normru = norm0  > dTolu ? norm(res)/nfreeu/norm0  : norm(res)/nfreeu
      bdone  = (normru<dTolu)
    else
      (vEqs,rEqs,KEqs) = getϕ(eqns, unew, λ)
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
    bprogress && ProgressMeter.update!(p, normru)
    if becho 
      if (bdone | bfailed) 
        @printf("iter: %2i, norm0: %.2e, normru: %.2e, normre: %.2e, eltime: %.2f sec.\n", 
                iter, norm0, normru, normre, Int64(Base.time_ns()-tic)/1e9)
      else
        @printf("iter: %2i, norm0: %.2e, normru: %.2e, normre: %.2e, normupdt: %.2e, α: %6.3f, eltime: %.2f sec.\n", 
                iter, norm0, normru, normre, normupdt, 
                oldupdt⋅updt/norm(updt)/norm(oldupdt), Int64(Base.time_ns()-tic)/1e9)
      end
      flush(stdout)
    end
    iter  += 1
  end

  (bfailed, normru, iter)
end

end
