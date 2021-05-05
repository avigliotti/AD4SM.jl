module AD4SM


include("adiff.jl")
include("materials.jl")
include("elements.jl")

# using .adiff, .Materials, .Elements 
export adiff, Materials, Elements 

VER = "0.0.2"
export VER

end # module
