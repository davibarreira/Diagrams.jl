module Diagrams

using LinearAlgebra
using Plots
using GeometryBasics
using RecursiveArrayTools

export plotvec!
export envelope

@doc """
plotvec!(v::Vector{<:Real})

Auxiliary function to plot multiple vectors starting
from the origin.
For a list of vectors [v1,v2,v3], we have that
`plotvec!([v1,v2,v3])` plots each vector starting from the origin.
"""
function plotvec!(v::Vector{<:Vector})
    va = VectorOfArray(v)
    zvec = zeros(size(v)[1], 2)[:, 1]
    quiver!(zvec, zvec, quiver=(va[1, :], va[2, :]))
end
function plotvec!(v::Vector{<:Real})
    plotvec!([v])
end


@doc """
support(g::AbstractGeometry, v::Vector)

This is the support function. Given a vector,
it returns the supremum of the inner product
with another vector `x` inside the set ``S``.
In this case, the set is replaced by a geometric
shape `g`.

```math
\\text{support}_S(v):=\\text{sup}_{x \\in S} \\langle x, v \\rangle
```
"""
support(g::AbstractGeometry, v::Vector) = mapreduce(p -> dot(p, v), max, coordinates(g))

@doc """
envelope(g::AbstractGeometry, v::Vector)

The envelope of a geometric shape 

```math
\\text{envelope}_S(v):=\\frac{\\text{sup}_{x \\in S} \\langle x, v \\rangle}{||v||}
```
"""
envelope(t::AbstractGeometry, v::Vector) = support(t, normalize(v))

# @doc"""
# maxgrade(A::MultiVector)

# Returns the maximum grade of a multivector A
# with non-null value, i.e. ``\\langle A \\rangle_{max}``
# """
# function maxgrade(A::MultiVector)
#     grade(A,basegrade(algebra(A), findlast(x -> !(x ≈ 0), vector(A))))
# end


end
