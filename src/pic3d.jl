module PIC3D
include("particles.jl")
include("potential_solvers.jl")
using .Particles
using StaticArrays
using LinearAlgebra
using Random
using Statistics

const ε_0 = 8.854187818814e-12                      # vacuum permittivity [F/m]
const c   = 299_792_458.0                           # speed of light [m/s]
const c²  = c*c
const q_e = -1.602176634e-19                        # charge of electron [C = A*s]
const m_e = 9.1093837139e-31                        # mass of electron [kg]

const XL  = 0.05*32                                 # grid length along X (rows), iterator = i
const YL  = 0.05*32                                 # grid length along Y (columns), iterator = j
const ZL  = 0.05*32                                 # grid length along Z (pages), iterator = k
const Δx  = 0.05
const Δy  = 0.05
const Δz  = 0.05
const Δt  = 1.0e-9                                  # 1 ns
const NX  = round(Int, XL/Δx) + 1                   # number of nodes along X
const NY  = round(Int, YL/Δy) + 1
const NZ  = round(Int, ZL/Δz) + 1

const Ex    = zeros(Float64, NX-1, NY-1, NZ-1)
const Ey    = zeros(Float64, NX-1, NY-1, NZ-1)
const Ez    = zeros(Float64, NX-1, NY-1, NZ-1)
const ρ     = zeros(Float64, NX, NY, NZ)            # for charge and charge density in grid nodes
const ϕ     = zeros(Float64, NX-1, NY-1, NZ-1)      # for electric potential

# multigrid
const R1    = zeros(Float64, NX-1, NY-1, NZ-1)
const EPS1  = zeros(Float64, NX-1, NY-1, NZ-1)
const R2    = zeros(Float64, ceil(Int, (NX-1)/2), ceil(Int, (NY-1)/2), ceil(Int, (NZ-1)/2))
const EPS2  = zeros(Float64, ceil(Int, (NX-1)/2), ceil(Int, (NY-1)/2), ceil(Int, (NZ-1)/2))
const R4    = zeros(Float64, ceil(Int, (NX-1)/4), ceil(Int, (NY-1)/4), ceil(Int, (NZ-1)/4))
const EPS4  = zeros(Float64, ceil(Int, (NX-1)/4), ceil(Int, (NY-1)/4), ceil(Int, (NZ-1)/4))
const R8    = zeros(Float64, ceil(Int, (NX-1)/8), ceil(Int, (NY-1)/8), ceil(Int, (NZ-1)/8))
const EPS8  = zeros(Float64, ceil(Int, (NX-1)/8), ceil(Int, (NY-1)/8), ceil(Int, (NZ-1)/8))

#PCG
const APCG = spzeros(length(ϕ), length(ϕ))
const bPCG = zeros(Float64, length(ϕ))

const E_ex = [0.0, 0.0, 0.0]
const B_ex = [0.0, 0.0, 0.0]
Random.seed!(15)

# FFT
function compute_K2()
    K2 = zeros(Float64, (NX-1),(NY-1),(NZ-1))

    n = NX-1
    if n % 2 == 0
        kx = [0:n÷2-1; -n÷2:-1] .* 2π ./ XL
    else
        kx = [0:(n-1)÷2; -(n-1)÷2:-1] .* 2π ./ XL
    end
    n = NY-1
    if n % 2 == 0
        ky = [0:n÷2-1; -n÷2:-1] .* 2π ./ YL
    else
        ky = [0:(n-1)÷2; -(n-1)÷2:-1] .* 2π ./ YL
    end
    n = NZ-1
    if n % 2 == 0
        kz = [0:n÷2-1; -n÷2:-1] .* 2π ./ ZL
    else
        kz = [0:(n-1)÷2; -(n-1)÷2:-1] .* 2π ./ ZL
    end

    for k in 1:(NZ-1), j in 1:(NY-1), i in 1:(NX-1)
        if kx[i] != 0
            K2[i,j,k] += kx[i]^2*((sin(Δx*kx[i]/2))/(Δx*kx[i]/2))^2
        end
        if ky[j] != 0
            K2[i,j,k] += ky[j]^2*((sin(Δy*ky[j]/2))/(Δy*ky[j]/2))^2
        end
        if kz[k] != 0
            K2[i,j,k] += kz[k]^2*((sin(Δz*kz[k]/2))/(Δz*kz[k]/2))^2
        end
    end

    return K2
end

const K2 = compute_K2()
const ρ̂ = zeros(ComplexF64, NX-1, NY-1, NZ-1)
const ϕ̂ = similar(ρ̂)


function cell_for_particle(x)                   # left upper front node of cell
    i = floor(Int, x[1] / Δx) + 1
    j = floor(Int, x[2] / Δy) + 1
    k = floor(Int, x[3] / Δz) + 1
    return (i,j,k)
end

function get_weights(i, j, k, x)
    wx = (x[1] - (i-1)*Δx) / Δx
    wy = (x[2] - (j-1)*Δy) / Δy
    wz = (x[3] - (k-1)*Δz) / Δz
    return (wx,wy,wz)
end

function charge_deposition!(sp::Species)
    @inbounds for p in eachindex(sp.x)
        i, j, k = cell_for_particle(sp.x[p])
        wx, wy, wz = get_weights(i, j, k, sp.x[p])
        ρ[i,   j,   k]   += (1-wx) * (1-wy) * (1-wz) * sp.q
        ρ[i+1, j,   k]   += wx     * (1-wy) * (1-wz) * sp.q
        ρ[i+1, j+1, k]   += wx     * wy     * (1-wz) * sp.q
        ρ[i+1, j+1, k+1] += wx     * wy     * wz     * sp.q
        ρ[i,   j+1, k]   += (1-wx) * wy     * (1-wz) * sp.q
        ρ[i,   j+1, k+1] += (1-wx) * wy     * wz     * sp.q
        ρ[i,   j,   k+1] += (1-wx) * (1-wy) * wz     * sp.q
        ρ[i+1, j,   k+1] += wx     * (1-wy) * wz     * sp.q  
    end

    @inbounds for k = 1:(NZ-1), j = 1:(NY-1)
        ρ[1,j,k] += ρ[NX,j,k]
    end
    @inbounds for j = 1:(NY-1), i = 1:(NX-1)
        ρ[i,j,1] += ρ[i,j,NZ]
    end
    @inbounds for k = 1:(NZ-1), i = 1:(NX-1)
        ρ[i,1,k] += ρ[i,NY,k]
    end

    # @inbounds for k = 1:(NZ-1), j = 1:(NY-1)
    #     ρ[NX,j,k] = ρ[1,j,k]
    # end
    # @inbounds for j = 1:(NY-1), i = 1:(NX-1)
    #     ρ[i,j,NZ] = ρ[i,j,1]
    # end
    # @inbounds for k = 1:(NZ-1), i = 1:(NX-1)
    #     ρ[i,NY,k] = ρ[i,1,k]
    # end
end

function compute_charge_density!()
    @inbounds for i in eachindex(ρ)
        ρ[i] /= Δx*Δy*Δz*ε_0
    end
    ρ̅ = mean(ρ[1:NX-1, 1:NY-1, 1:NZ-1])
    @inbounds for i in eachindex(ρ)
        ρ[i] -= ρ̅
    end
end


function compute_electric_field!()
    @inbounds for k = 1:size(Ex)[3], j = 1:size(Ex)[2], i = 1:size(Ex)[1]
        Ex[i,j,k] = (pbc(ϕ,i-1,j,k)- pbc(ϕ,i+1,j,k))/(2Δx)
        Ey[i,j,k] = (pbc(ϕ,i,j-1,k)- pbc(ϕ,i,j+1,k))/(2Δy)
        Ez[i,j,k] = (pbc(ϕ,i,j,k-1) - pbc(ϕ,i,j,k+1))/(2Δz)
    end
end


function interpolate_E_to_particle!(x, e)
    i, j, k = cell_for_particle(x)
    wx, wy, wz = get_weights(i, j, k, x)
    e[1] = sum_electric_field(Ex, wx, wy, wz, i, j, k)
    e[2] = sum_electric_field(Ey, wx, wy, wz, i, j, k)
    e[3] = sum_electric_field(Ez, wx, wy, wz, i, j, k)
    
    e .+= E_ex
end


function sum_electric_field(E, wx, wy, wz, i, j, k)
    e  = (1-wx) * (1-wy) * (1-wz) * pbc(E,i,j,k)
    e += wx     * (1-wy) * (1-wz) * pbc(E,i+1,j,k)
    e += wx     * wy     * (1-wz) * pbc(E,i+1,j+1,k)
    e += wx     * wy     * wz     * pbc(E,i+1,j+1,k+1)
    e += (1-wx) * wy     * (1-wz) * pbc(E,i,j+1,k)
    e += (1-wx) * wy     * wz     * pbc(E,i,j+1,k+1)
    e += (1-wx) * (1-wy) * wz     * pbc(E,i,j,k+1)
    e += wx     * (1-wy) * wz     * pbc(E,i+1,j,k+1)
    return e
end


function periodic_boundary_for_particle!(x)
    if x[1] < 0
        x[1] += (NX-1)*Δx
    elseif x[1] > (NX-1)*Δx
        x[1] -= (NX-1)*Δx
    end
    if x[2] < 0
        x[2] += (NY-1)*Δy
    elseif x[2] > (NY-1)*Δy
        x[2] -= (NY-1)*Δy
    end
    if x[3] < 0
        x[3] += (NZ-1)*Δz
    elseif x[3] > (NZ-1)*Δz
        x[3] -= (NZ-1)*Δz
    end      
end

const u⁻ = @MVector zeros(3)
const u′ = @MVector zeros(3)
const u⁺ = @MVector zeros(3)
const e = @MVector zeros(3)
const t = @MVector zeros(3)
const s = @MVector zeros(3)

function boris_pusher!(sp::Species, factor)
    for p in eachindex(sp.x)
        Q = (sp.q*Δt*factor)/(sp.m*2)
        interpolate_E_to_particle!(sp.x[p], e)
        @. u⁻ = sp.v[p] + Q*e

        u² = dot(u⁻, u⁻)       
        γ  = sqrt(1. + u²/c²)
        @. t  = (Q/γ) * B_ex
        t² = dot(t, t)
        s .= 2.0t ./ (1. + t²)

        u′ .= u⁻ .+ cross(u⁻, t)
        u⁺ .= u⁻ .+ cross(u′, s)
        
        @. sp.v[p] = u⁺ + e * Q

        @. sp.x[p] += sp.v[p]*Δt*factor

        periodic_boundary_for_particle!(sp.x[p])
    end
end

# function cross_product!(a, b)
#     cp[1] = a[2]*b[3] - a[3]*b[2]
#     cp[2] = a[3]*b[1] - a[1]*b[3]
#     cp[3] = a[1]*b[2] - a[2]*b[1]
# end

function bit_reversal_permutation!(x, dim)
    N = size(x, dim)
    j = 1
    @inbounds for i in 1:N
        if j > i
            if dim == 1
                temp       = x[i, :, :]
                x[i, :, :] = x[j, :, :]
                x[j, :, :] = temp
            elseif dim == 2
                temp       = x[:, i, :]
                x[:, i, :] = x[:, j, :]
                x[:, j, :] = temp
            elseif dim == 3
                temp       = x[:, :, i]
                x[:, :, i] = x[:, :, j]
                x[:, :, j] = temp
            end
        end
        m = N >> 1
        while m >= 1 && j > m
            j -= m
            m >>= 1
        end
        j += m
    end
end

function fft1d_dim!(x, dim, isign)
    
    N = size(x, dim)

    step = 2
    while step <= N
        half_step = step ÷ 2
        w_m = exp(-2im* isign * π / step)

        if dim == 1
            @inbounds for k in 1:step:N, l in 1:size(x, 2), m in 1:size(x, 3)
                w = 1.0
                @inbounds for n in 0:(half_step-1)
                    u = x[k + n, l, m]
                    t = w * x[k + n + half_step, l, m]
                    x[k + n, l, m] = u + t
                    x[k + n + half_step, l, m] = u - t
                    w *= w_m
                end
            end
        elseif dim == 2
            @inbounds for k in 1:size(x, 1), l in 1:step:N, m in 1:size(x, 3)
                w = 1.0
                @inbounds for n in 0:(half_step-1)
                    u = x[k, l + n, m]
                    t = w * x[k, l + n + half_step, m]
                    x[k, l + n, m] = u + t
                    x[k, l + n + half_step, m] = u - t
                    w *= w_m
                end
            end
        elseif dim == 3
            @inbounds for k in 1:size(x, 1), l in 1:size(x, 2), m in 1:step:N
                w = 1.0
                @inbounds for n in 0:(half_step-1)
                    u = x[k, l, m + n]
                    t = w * x[k, l, m + n + half_step]
                    x[k, l, m + n] = u + t
                    x[k, l, m + n + half_step] = u - t
                    w *= w_m
                end
            end
        end

        step *= 2
    end

    if isign == -1
        x .= x ./ N
    end
    
    return x
end

function fft3d(x, isign) #zrobić żeby w tablicy po kolei były przechowywane real i im
    ρfft .= convert(Array{ComplexF64, 3}, x)
    for dim in 1:3
        bit_reversal_permutation!(ρfft, dim)
        fft1d_dim!(ρfft, dim, isign)
    end
    return ρfft
end


const ρfft = zeros(ComplexF64, size(ϕ))


function compute_potential_FFT!()
    # FFT gęstości ładunku
    ρ̂ .= fft3d(ρ[1:(NX-1),1:(NY-1),1:(NZ-1)], 1)

    @inbounds for i in 1:size(ϕ̂, 1), j in 1:size(ϕ̂, 2), k in 1:size(ϕ̂, 3)
        if K2[i,j,k] != 0
            ϕ̂[i, j, k] = ρ̂[i, j, k] / K2[i,j,k]
        else
            ϕ̂[i, j, k] = 0.0 # na pewno 0 ?
        end
    end
    
    ϕ .= real.(fft3d(ϕ̂, -1))
end


function clear!()
    fill!(ρ, 0)
    fill!(ϕ, 0)
    fill!(Ex, 0)
    fill!(Ey, 0)
    fill!(Ez, 0)
end

function timestep!(sp::Species, time_factor)
    clear!()
    charge_deposition!(sp)
    compute_charge_density!()
    compute_potential!()
    compute_electric_field!()
    boris_pusher!(sp, time_factor)
end

function timestep!(sp_e::Species, sp_i::Species, time_factor=1.0)
    clear!()
    charge_deposition!(sp_e)
    charge_deposition!(sp_i)
    compute_charge_density!()
    compute_potential!()
    compute_electric_field!()
    boris_pusher!(sp_e, time_factor)
    boris_pusher!(sp_i, time_factor)
end

function timestep_multigrid!(sp::Species, time_factor=1.0)
    clear!()
    charge_deposition!(sp)
    compute_charge_density!()
    compute_potential_multigrid!()
    compute_electric_field!()
    boris_pusher!(sp, time_factor)
end

function timestep_multigrid!(sp_e::Species, sp_i::Species, time_factor=1.0)
    clear!()
    charge_deposition!(sp_e)
    charge_deposition!(sp_i)
    compute_charge_density!()
    compute_potential_multigrid!()
    compute_electric_field!()
    boris_pusher!(sp_e, time_factor)
    boris_pusher!(sp_i, time_factor)
end

function timestep_PCG!(sp::Species, time_factor=1.0)
    clear!()
    charge_deposition!(sp)
    compute_charge_density!()
    compute_potential_PCG!()
    compute_electric_field!()
    boris_pusher!(sp, time_factor)
end

function timestep_PCG!(sp_e::Species, sp_i::Species, time_factor=1.0)
    clear!()
    charge_deposition!(sp_e)
    charge_deposition!(sp_i)
    compute_charge_density!()
    compute_potential_PCG!()
    compute_electric_field!()
    boris_pusher!(sp_e, time_factor)
    boris_pusher!(sp_i, time_factor)
end

function timestep_FFT!(sp_e::Species, sp_i::Species, time_factor=1.0)
    clear!()
    charge_deposition!(sp_e)
    charge_deposition!(sp_i)
    compute_charge_density!()
    compute_potential_FFT!()
    #println(ϕ[1:3,1:3,1])
    compute_electric_field!()
    boris_pusher!(sp_e, time_factor)
    boris_pusher!(sp_i, time_factor)
end
end