module AD4SM

include("adiff.jl")
include("materials.jl")
include("elements.jl")
include("solvers.jl")

using  .adiff, .Solvers, .Materials, .Elements
export adiff, Materials, Elements, Solvers

VER = "0.0.7"

end # module
