module PIC3D
include("particles.jl")
include("potential_solvers.jl")
using .Particles
using StaticArrays
using LinearAlgebra
using Random

const ε_0 = 8.854187818814e-12                   # vacuum permittivity [F/m]
const c   = 299_792_458.0                       # speed of light [m/s]
const c²  = c*c
const q_e = -1.602176634e-19                    # charge of electron [C = A*s]
const m_e = 9.1093837139e-31                    # mass of electron [kg]

const XL  = 1.0                                 # grid length along X (rows), iterator = i
const YL  = 1.0                                 # grid length along Y (columns), iterator = j
const ZL  = 1.0                                 # grid length along Z (pages), iterator = k
const Δx  = 0.05
const Δy  = 0.05
const Δz  = 0.05
const Δt  = 1.0e-9                              # 1 ns
const NX  = round(Int, XL/Δx) + 1               # number of nodes along X
const NY  = round(Int, YL/Δy) + 1
const NZ  = round(Int, ZL/Δz) + 1

const Ex    = zeros(Float64, NX-1, NY-1, NZ-1)
const Ey    = zeros(Float64, NX-1, NY-1, NZ-1)
const Ez    = zeros(Float64, NX-1, NY-1, NZ-1)
const ρ     = zeros(Float64, NX, NY, NZ)        # for charge and charge density in grid nodes
const ϕ     = zeros(Float64, NX-1, NY-1, NZ-1)  # for electric potential

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

    @inbounds for k = 1:(NZ-1), j = 1:(NY-1)    # wszędzie -1 bo te węzły i tak nie będą wykorzystane przy liczeniu potencjału
        ρ[1,j,k] += ρ[NX,j,k]
    end
    @inbounds for j = 1:(NY-1), i = 1:(NX-1)
        ρ[i,j,1] += ρ[i,j,NZ]
    end
    @inbounds for k = 1:(NZ-1), i = 1:(NX-1)
        ρ[i,1,k] += ρ[i,NY,k]
    end
end

function compute_charge_density!()
    @inbounds for i in eachindex(ρ)
        ρ[i] /= Δx*Δy*Δz*ε_0
    end
end


function compute_electric_field!()
    @inbounds for k = 1:size(Ex)[3], j = 1:size(Ex)[2], i = 1:size(Ex)[1]
        Ex[i,j,k] = (pbc(ϕ,i-1,j,k)- pbc(ϕ,i,j,k))/(2Δx)
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

const u⁻ = zeros(Float64, 3)
const u′ = zeros(Float64, 3)
const u⁺ = zeros(Float64, 3)
const e = zeros(Float64, 3)
const t = zeros(Float64, 3)
const s = zeros(Float64, 3)

function boris_pusher!(sp::Species, factor)
    for p in eachindex(sp.x)
        Q = (sp.q*Δt*factor)/(sp.m*2)
        interpolate_E_to_particle!(sp.x[p], e)
        u⁻ .= sp.v[p] + Q*e

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
end