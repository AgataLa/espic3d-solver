include("src/pic3d.jl")
using .PIC3D
using LinearAlgebra

filename = "results.txt"

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
        push!(ions.v, [0.0, 0.0, 0.0])
        
        x = [rand()*PIC3D.XL, rand()*PIC3D.YL, rand()*PIC3D.ZL] #[rand()*PIC3D.XL, rand()*PIC3D.YL, rand()*PIC3D.ZL]
        v = [-5e5, 0, 0]
        push!(electrons.x, [x...])
        push!(electrons.v, [v...])
        push!(ions.x, [x...])
        push!(ions.v, [0.0, 0.0, 0.0])
    end

    return electrons, ions
end


NP = 2048
d = 4.5e9
electrons, ions = generate_particles!(NP, d)

Δt = 1e-9
timesteps = 1000

t1 = time();
for i = 1:timesteps
    PIC3D.timestep_FFT!(1.0, electrons, ions)
    if i % 100 == 0
        elapsed_time = time() - t1
        open(filename, "w") do file
            write(file, "FFT: $(i)      $(elapsed_time)\n")
        end
        t1 = time();
    end
end


# include("src/pic3d.jl")
# using .PIC3D
# electrons, ions = generate_particles!(NP, d)
# for i = 1:timesteps
#     PIC3D.timestep!(1.0, electrons, ions)
# end


# include("src/pic3d.jl")
# using .PIC3D
# electrons, ions = generate_particles!(NP, d)
# for i = 1:timesteps
#     PIC3D.timestep_multigrid!(1.0, electrons, ions)
# end
