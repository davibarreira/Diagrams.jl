module Diagrams

using LinearAlgebra
using Plots
using GeometryBasics
using RecursiveArrayTools
using CliffordAlgebras

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

"""
maxgrade(A::MultiVector)

Returns the maximum grade ``k`` of a multivector A
with non-null value, i.e. ``\\langle A \\rangle_{max}``
is equal to `grade(A,maxgrade(A))`.
"""
function maxgrade(A::MultiVector)
    k = findlast(x -> !(x ≈ 0), vector(A))
    if isnothing(k)
        return 0
    end
    return basegrade(algebra(A), k)
end

"""
mingrade(A::MultiVector)

Returns the minimum grade ``k`` of a multivector A
with non-null value, or returns 0.
"""
function mingrade(A::MultiVector)
    k = findfirst(x -> !(x ≈ 0), vector(A))
    if isnothing(k)
        return 0
    end
    return basegrade(algebra(A), k)
end


# Another possible implementation
# """
# mingrade(A::MultiVector)

# Returns the grade of a blade A. If A
# is a multivector (linear combination of blades of different grades),
# than it returns the smallest non-null grade.
# """
# function mingrade(A::MultiVector)
#     g = findfirst(x->isgrade(A,x),0:sum(signature(algebra(A)))) - 1
#     if isnothing(g)
#         return 0
#     end
#     return g
# end

"""
△(A::MultiVector,B::MultiVector)

Symmetric differences.

"""
function △(A::MultiVector,B::MultiVector)
    C = A*B
    return grade(C, maxgrade(C))
end

grademeet(A::MultiVector,B::MultiVector) = (mingrade(A) + mingrade(B) - maxgrade(△(A,B)))/2

"""
getgradesdict(cl::CliffordAlgebra)

Returns a dictionary where a list of multivector basis
are given for each grade.

"""
function getgradesdict(cl::CliffordAlgebra)
    gradesdict = Dict(g =>[] for g in 0:sum(signature(cl)))
    ngrades = sum(signature(cl))
    for g in 0:ngrades
        for (i,p) in enumerate(propertynames(cl))
            if i <= sum(binomial.(dimcl,0:g))
                push!(gradesdict[g],p)
            end
        end
    end

    return gradesdict
end

"""
Returns a list of base multivector for a given
grade `k`. Note that grades go from ``0`` to ``n``, 
where ``n`` is the dimension, given by the sum of
the signature, e.g. ``R^{3,1,1}`` has
``n = 5``.
"""
function basesfromgrade(cl::CliffordAlgebra, k::Int)
    ngrades = sum(signature(cl))
    i = sum(binomial.(ngrades,0:k-1))+1
    j = sum(binomial.(ngrades,0:k))
    basesymbol.(Ref(cl),i:j)
end

end
