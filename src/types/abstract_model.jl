
"""
Subtypes of `AbstractModel` represent surrogate models for SBI.

Each subtype of `AbstractModel` should implement:
- `construct_model(::AbstractModel, ::AbstractProblem) -> ::SurrogateModel`
"""
abstract type AbstractModel end

"""
    construct_model(::AbstractModel, ::AbstractProblem) -> ::SurrogateModel

Construct an instance of the `SurrogateModel` for the given `AbstractProblem`
according to the given `AbstractModel`.
"""
function construct_model end
