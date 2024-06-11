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


function poly_2_eval(p)
    ex = 0
    for e in p
        ex *= 2
        ex+=e
    end
    ex
end


function swap_array(nn)
    
end

swap_array(8)

poly_2_eval(digits(Bool, 3, base=2, pad=3))

function poly_2_eval(p)
    ex = 0
    for e in p
        ex *= 2
        ex+=e
    end
    ex
end

function swap_index(index, nbits)
    return poly_2_eval(digits(Bool, index, base=2, pad=nbits))
end

function FFT1D(data::Array{Float64}, isign)
    n=size(data, 1) << 1
    nbits = ndigits(size(data,1)-1, base=2)
    for i in 1:div(size(data,1)-1, 2)
        pair = swap_index(i, nbits)
        temp = data[i]
        data[i] = data[pair]
        data[pair] = temp
    end

    mmax = 2
    while n > mmax
        istep = mmax << 1
        theta = isign*(2π/mmax)
        wtemp = sin(0.5*theta)
        wpr = -2.0 * wtemp^2
        wpi = sin(theta)
        wr = 1.0
        wi = 0.0
        m = 1
        while m < mmax
            i = m
            while i <= n
                j=i+mmax
                tempr=wr*data[j]-wi*data[j+1]
                tempi=wr*data[j+1]+wi*data[j]
                data[j]=data[i]-tempr
                data[j+1]=data[i+1]-tempi
                data[i] += tempr
                data[i+1] += tempi

                i += istep
            end
            wr=(wtemp=wr)*wpr-wi*wpi+wr
            wi=wi*wpr+wtemp*wpi+wi

            m += 2
        end
        mmax=istep;
    end
end


x = rand(Float64, 16)

FFT1D(x, 1)