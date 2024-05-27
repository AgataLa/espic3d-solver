module Particles
using StaticArrays
export Species

Base.@kwdef struct Species
    q :: Float64
    m :: Float64
    x :: Array{Array{Float64, 1},1} = []
    v :: Array{Array{Float64, 1},1} = []
end

end