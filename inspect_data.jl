using JLD2

d = load(data-bosip/SIRProblem/standard_1_data.jld2)
println(Keys: )
key = first(keys(d))
println(First key: )
data = d[key]
println(Data type: )
if isdefined(Base, :fieldnames)
    try
        println(Fields: )
    catch e
        println(Cannot get fields: )
    end
end

# Try different ways to access it
for (k, v) in d
    println(nKey: , Type: )
    if typeof(v) <: AbstractVector
        println( Length: )
        if length(v) > 0
            println( First elem type: )
        end
    end
end
