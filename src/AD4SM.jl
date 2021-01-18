module AD4SM

greet() = print("Hello World!")

# include("adiff.jl")
include("materials.jl")
include("elements.jl")

# using .adiff, .Materials, .Elements 
export adiff, Materials, Elements 

end # module
