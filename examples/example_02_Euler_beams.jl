
using PyPlot, MAT
;

using AD4SM
;

mean(x) = sum(x)/length(x)

#=  
makeHexaModels is a helper function to create a N1xN2 
hexagonal lattice with side length L0
=#
function makeHexaModels(;N1 = 5, N2=5, L0=1, dTol=1e-6)

  # unit cell and periodic directions
  a1 = sqrt(3)L0.*(cos(π/6), sin(π/6))
  a2 = sqrt(3)L0.*(cos(π/6), -sin(π/6))
  p0 = [[cos(θ)L0, sin(θ)L0] for θ = range(0, stop=5π/3, length=6)]

  # replicate the nodes
  nodes = []
  for jj in  range(1, stop=N2)
    for ii in range(0, stop=N1-1)
      Δ = a1.*floor(ii/2) .+ a2.*ceil(ii/2) .+ (jj-1).*(a1.-a2)
      for item in p0
        push!(nodes, [item[1]+Δ[1], item[2]+Δ[2]])
      end
    end
  end

  #remove duplicated nodes
  global jj = 1
  while jj < length(nodes)
    rmidx = []
    for (ii, p0) in enumerate(nodes[jj+1:end])
      dd = sqrt((p0[1]-nodes[jj][1])^2 + (p0[2]-nodes[jj][2])^2) 
      (dd <= dTol) && push!(rmidx, ii)        
    end
    deleteat!(nodes, jj.+rmidx)
    jj+=1    
  end

  # add beams
  jj    = 1
  beams = []
  while jj < length(nodes)
    for (ii, p0) in enumerate(nodes[jj+1:end])
      dd = sqrt((p0[1]-nodes[jj][1])^2 + (p0[2]-nodes[jj][2])^2) 
      (abs(dd-L0) <= dTol) && push!(beams, [jj, jj+ii])
    end
    jj+=1    
  end

  return (nodes, beams)
end
;

@time (nodes, beams) = makeHexaModels(N1 = 21, N2 = 21, L0 = 1.2)

nNodes   = length(nodes)
t, w, Es = .1, .1, 5
A        = t*w
# elems    = [Elements.Beam(beam, nodes[beam], t, w, mat=Materials.Hooke(Es, 0.3)) for beam in beams]
elems    = [Elements.Beam(beam, nodes[beam], t, w, mat=Materials.Hooke1D(Es)) for beam in beams]
;

minx = minimum([node[1] for node in nodes])
maxx = maximum([node[1] for node in nodes])
miny = minimum([node[2] for node in nodes])
maxy = maximum([node[2] for node in nodes])


idbtm = findall(abs.([node[2] for node in nodes].-miny) .<1e-6)
idtop = findall(abs.([node[2] for node in nodes].-maxy) .<1e-6)

LY    = maxy-miny
;

ax = getproperty(figure(),:add_subplot)(1,1,1)
for item in beams
    n1 = nodes[item[1]]
    n2 = nodes[item[2]]
    PyPlot.plot(
        [n1[1], n2[1]], 
        [n1[2], n2[2]],
        markersize=1, color=:b)
end
getproperty(ax, :set_aspect)("equal")
title("undeformed model")
;

u     = fill!(zeros(3, nNodes), NaN)
ΔY    = 0.75LY

u[:,idbtm] .= 0.0
u[:,idtop] .= 0.0
u[2,idtop] .= ΔY
;

bfreeu = isnan.(u)
uold   = zeros(size(u))
unew   = zeros(size(u))
unew[.!bfreeu] .= u[.!bfreeu]*1e-4

fe    = zeros(length(unew))
;

allus = Solvers.solve(elems, u, N=20, bprogress=false, bechoi=false, becho=true, ballus=true)
;

unew = allus[end][1]
ax   = getproperty(figure(),:add_subplot)(1,1,1)

for beam in beams
    n1, n2 = nodes[beam]
    u1 = unew[:,beam[1]]
    u2 = unew[:,beam[2]]    
    
    PyPlot.plot(
        [n1[1], n2[1]], 
        [n1[2], n2[2]],
        markersize=1, color=:r, alpha=0.25)
    PyPlot.plot(
        [n1[1]+u1[1], n2[1]+u2[1]], 
        [n1[2]+u1[2], n2[2]+u2[2]],
        markersize=1, color=:b)
end
getproperty(ax, :set_aspect)("equal")
title("deformed  model")
;

idd = LinearIndices(u)[2,idtop]

RY  = [sum(u[2][idd]) for u in allus];
Δu  = [mean(u[1][idd]) for u in allus]

PyPlot.plot(Δu/LY, RY/Es/A)
;

matwrite("HexaLattice.mat", Dict(
  "Ry"     => RY,
  "nodes"  => hcat(nodes[:]...)|> transpose |> collect, 
  "beams"  => hcat(beams[:]...)|> transpose |> collect, 
  "LY"     => LY,
  "DeltaY" => Δu, 
  "u"      => unew,
  "Es"     => Float64(Es),
  "A"      => A))
