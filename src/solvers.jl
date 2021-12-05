module Solvers

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
#
#
# elastic energy evaluation functions for models (Array of elements)
function getϕ(elems::Array, u; T=Threads.nthreads())

  nDoFs  = length(u)
  nElems = length(elems)
  indxes = LinearIndices(u)

  Φ = zeros(size(elems))
  r = zeros(nDoFs)
  C = [spzeros(nDoFs, nDoFs) for ii = 1:T]
  Threads.@threads for kk = 1:T
    for ii = kk:T:nElems
      elem           =  elems[ii]
      nodes          =  elem.nodes
      iDoFs          =  indxes[:,nodes][:]
      ϕ              =  Elements.getϕ(elem, adiff.D2(u[:,nodes]))
      Φ[ii]          =  adiff.val(ϕ)
      r[iDoFs]       += adiff.grad(ϕ)
      C[kk][iDoFs,iDoFs] += adiff.hess(ϕ)
    end
  end

  (Φ, r, sum(C))
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
        # updt[:]          = Kt[ifreeu,ifreeu]\res
        updt[:]          = qr(Kt[ifreeu,ifreeu])\res
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
        H[iieqs,iieqs]   = spdiagm(0=>dNoise*randn(nEqs))
        updt[:]          = qr(H)\res
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
        updt[:]         = qr(Kt[ifreeu,ifreeu])\res
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
        H[iieqs,iieqs]  = spdiagm(0=>dNoise*randn(nEqs))
        updt[:]         = qr(H)\res
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
