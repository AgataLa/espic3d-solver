# N = 1000
# const abc = zeros(Float64, N,N,N)

# function operation1(a)
#     for i in eachindex(a)
#         a[i] += 10
#     end
# end

# function operation2()
#     for i in eachindex(abc)
#         abc[i] += 10
#     end
# end


# using BenchmarkTools

# @btime operation1(abc) # 332.800 μs (0 allocations: 0 bytes)  338.700 μs (0 allocations: 0 bytes) 646.124 ms (0 allocations: 0 bytes) 651.692 ms (0 allocations: 0 bytes)
# @btime operation1($abc) # 335.800 μs (0 allocations: 0 bytes)

# @btime operation2() # 340.700 μs (0 allocations: 0 bytes) 337.400 μs (0 allocations: 0 bytes) 684.969 ms (0 allocations: 0 bytes) 645.480 ms (0 allocations: 0 bytes) 641.956 ms (0 allocations: 0 bytes)

# @code_llvm operation1(abc)
# @code_llvm operation2()


include("pic3d.jl")
using .PIC3D
using PyPlot
using LinearAlgebra
using PyCall
anim =  pyimport("matplotlib.animation");
using BenchmarkTools

function generate_particles!(NP, d)
    num = d / (2*NP / (PIC3D.XL*PIC3D.YL*PIC3D.ZL))
    println(num)
    electrons = PIC3D.Species(q=PIC3D.q_e*num, m=PIC3D.m_e*num)
    ions = PIC3D.Species(q=-PIC3D.q_e*num, m=PIC3D.m_e*num*2000)
    @inbounds for i = 1:NP
        x = [rand()*PIC3D.XL, rand()*PIC3D.YL, rand()*PIC3D.ZL]
        v = [5e5, 0, 0]
        push!(electrons.x, [x...])
        push!(electrons.v, [v...])
        push!(ions.x, [x...])
        push!(ions.v, [v...])
        
        x = [rand()*PIC3D.XL, rand()*PIC3D.YL, rand()*PIC3D.ZL]
        v = [-5e5, 0, 0]
        push!(electrons.x, [x...])
        push!(electrons.v, [v...])
        push!(ions.x, [x...])
        push!(ions.v, [v...])
    end

    return electrons, ions
end

NP = 2048 #2048
d = 4.5e9
electrons, ions = generate_particles!(NP, d)

@btime PIC3D.timestep_multigrid!(electrons, ions, 1)