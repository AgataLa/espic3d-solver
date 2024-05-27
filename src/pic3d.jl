module PIC3D
include("particles.jl")
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


const max_it    = 10000                         # Gauss-Seidel for ϕ
const ω         = 1.4                           # SOR
const tolerance = 1e-8                          # L2 tolerance

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

    #println(ρ)
end

function compute_charge_density!()
    @inbounds for i in eachindex(ρ)
        ρ[i] /= Δx*Δy*Δz
    end
end

function periodic_boundary_conditions(prv, nxt, N)
    if prv == 0
        prv = N
    end
    if nxt > N
        nxt = 1
    end
    return (prv, nxt)
end

function pbc(A, i, j, k)
    return A[(mod1(i, size(A)[1])), (mod1(j, size(A)[2])), (mod1(k, size(A)[3]))]
end


function compute_potential!()
    L2 = 0.0
    conv = false

    @inbounds for m = 1:max_it
        @inbounds for k = 1:size(ϕ)[3], j = 1:size(ϕ)[2], i = 1:size(ϕ)[1]
            prvi, nxti = periodic_boundary_conditions(i-1, i+1, size(ϕ)[1])
            prvj, nxtj = periodic_boundary_conditions(j-1, j+1, size(ϕ)[2])
            prvk, nxtk = periodic_boundary_conditions(k-1, k+1, size(ϕ)[3]) 
            ϕ_new = ((ϕ[prvi,j,k] + ϕ[nxti,j,k])/(Δx^2) + (ϕ[i,prvj,k] + ϕ[i,nxtj,k])/(Δy^2) + (ϕ[i,j,prvk] + ϕ[i,j,nxtk])/(Δz^2) + ρ[i,j,k]/ε_0) / (2/(Δx^2) + 2/(Δy^2) + 2/(Δz^2))

            ϕ[i,j,k] = ϕ[i,j,k] + ω*(ϕ_new - ϕ[i,j,k])
        end

        if m % 25 == 0
            sum = 0
            @inbounds for k = 1:(NZ-1), j = 1:(NY-1), i = 1:(NX-1)
                prvi, nxti = periodic_boundary_conditions(i-1, i+1, NX)
                prvj, nxtj = periodic_boundary_conditions(j-1, j+1, NY)
                prvk, nxtk = periodic_boundary_conditions(k-1, k+1, NZ)
                r = -ϕ[i,j,k] + (ρ[i,j,k]/ε_0 + (ϕ[prvi,j,k] + ϕ[nxti,j,k])/(Δy^2) + (ϕ[i,prvj,k] + ϕ[i,nxtj,k])/(Δx^2) + (ϕ[i,j,prvk] + ϕ[i,j,nxtk])/(Δz^2)) / (2/(Δx^2) + 2/(Δy^2) + 2/(Δz^2))
                sum += r^2
            end
            L2 = sqrt(sum / ((NX-1)*(NY-1)*(NZ-1)))
            if L2 < tolerance
                conv = true
                println("Converged after $(m) iterations, L2 = ", L2)
                break
            end
        end
    end

    if conv == false
        println("GS failed to converge, L2 = ", L2)
    end
end

function gauss_seidel(A, dx, dy, dz)
    @inbounds for m = 1:fine_its
        @inbounds for k = 1:(NZ-1), j = 1:(NY-1), i = 1:(NX-1)
            prvi, nxti = periodic_boundary_conditions(i-1, i+1, NX)
            prvj, nxtj = periodic_boundary_conditions(j-1, j+1, NY)
            prvk, nxtk = periodic_boundary_conditions(k-1, k+1, NZ) 
            ϕ_new = ((ϕ[prvi,j,k] + ϕ[nxti,j,k])/(Δx^2) + (ϕ[i,prvj,k] + ϕ[i,nxtj,k])/(Δy^2) + (ϕ[i,j,prvk] + ϕ[i,j,nxtk])/(Δz^2) + ρ[i,j,k]/ε_0) / (2/(Δx^2) + 2/(Δy^2) + 2/(Δz^2))

            ϕ[i,j,k] = ϕ[i,j,k] + ω*(ϕ_new - ϕ[i,j,k])
        end 
    end
end

function compute_potential_multigrid!()
    L2 = 0.0
    conv = false

    fine_its = 5
    h2_its = 10
    h4_its = 20

    Δx2 = 2Δx
    Δy2 = 2Δy
    Δz2 = 2Δz
    Δx4 = 4Δx
    Δy4 = 4Δy
    Δz4 = 4Δz

    R1 = zeros(Float64, NX-1, NY-1, NZ-1)
    EPS1 = zeros(Float64, NX-1, NY-1, NZ-1)
    R2 = zeros(Float64, ceil(Int, (NX-1)/2), ceil(Int, (NY-1)/2), ceil(Int, (NZ-1)/2))
    EPS2 = zeros(Float64, ceil(Int, (NX-1)/2), ceil(Int, (NY-1)/2), ceil(Int, (NZ-1)/2))
    R4 = zeros(Float64, ceil(Int, (NX-1)/4), ceil(Int, (NY-1)/4), ceil(Int, (NZ-1)/4))
    EPS4 = zeros(Float64, ceil(Int, (NX-1)/4), ceil(Int, (NY-1)/4), ceil(Int, (NZ-1)/4))

    for it = 1:max_it
        #1 fine mesh iterations
        @inbounds for m = 1:fine_its
            @inbounds for k = 1:(NZ-1), j = 1:(NY-1), i = 1:(NX-1)
                prvi, nxti = periodic_boundary_conditions(i-1, i+1, NX)
                prvj, nxtj = periodic_boundary_conditions(j-1, j+1, NY)
                prvk, nxtk = periodic_boundary_conditions(k-1, k+1, NZ) 
                ϕ_new = ((ϕ[prvi,j,k] + ϕ[nxti,j,k])/(Δx^2) + (ϕ[i,prvj,k] + ϕ[i,nxtj,k])/(Δy^2) + (ϕ[i,j,prvk] + ϕ[i,j,nxtk])/(Δz^2) + ρ[i,j,k]/ε_0) / (2/(Δx^2) + 2/(Δy^2) + 2/(Δz^2))

                ϕ[i,j,k] = ϕ[i,j,k] + ω*(ϕ_new - ϕ[i,j,k])
            end 
        end

        #1 fine mesh residuum and conv check
        sum_L2 = 0
        @inbounds for k = 1:(NZ-1), j = 1:(NY-1), i = 1:(NX-1)
            prvi, nxti = periodic_boundary_conditions(i-1, i+1, NX)
            prvj, nxtj = periodic_boundary_conditions(j-1, j+1, NY)
            prvk, nxtk = periodic_boundary_conditions(k-1, k+1, NZ)
            R1[i,j,k] = -ϕ[i,j,k] + (ρ[i,j,k]/ε_0 + (ϕ[prvi,j,k] + ϕ[nxti,j,k])/(Δy^2) + (ϕ[i,prvj,k] + ϕ[i,nxtj,k])/(Δx^2) + (ϕ[i,j,prvk] + ϕ[i,j,nxtk])/(Δz^2)) / (2/(Δx^2) + 2/(Δy^2) + 2/(Δz^2))
            sum_L2 += R1[i,j,k]^2
        end
        L2 = sqrt(sum_L2 / ((NX-1)*(NY-1)*(NZ-1)))
        if L2 < tolerance
            conv = true
            break
        end


        # 2h mesh restriction
        @inbounds for k = 1:2:size(R1)[3], j = 1:2:size(R1)[2], i = 1:2:size(R1)[1]
            prvi, nxti = pbc_for_residuum(i-1, i+1, size(R1)[3])
            prvj, nxtj = pbc_for_residuum(j-1, j+1, size(R1)[2])
            prvk, nxtk = pbc_for_residuum(k-1, k+1, size(R1)[1])
            R2[ceil(Int,i/2),ceil(Int,j/2),ceil(Int,k/2)] = (6*R1[i,j,k] + R1[prvi,j,k] + R1[nxti,j,k] + R1[i,prvj,k] + R1[i,nxtj,k]+ R1[i,j,prvk] + R1[i,j,nxtk]) / 12.0
        end

        # 4h mesh restriction
        @inbounds for k = 1:2:size(R2)[3], j = 1:2:size(R2)[2], i = 1:2:size(R2)[1]
            prvi, nxti = pbc_for_residuum(i-1, i+1, size(R2)[1])
            prvj, nxtj = pbc_for_residuum(j-1, j+1, size(R2)[2])
            prvk, nxtk = pbc_for_residuum(k-1, k+1, size(R2)[3])
            R4[ceil(Int,i/2),ceil(Int,j/2),ceil(Int,k/2)] = (6*R2[i,j,k] + R2[prvi,j,k] + R2[nxti,j,k] + R2[i,prvj,k] + R2[i,nxtj,k]+ R2[i,j,prvk] + R2[i,j,nxtk]) / 12.0
        end

        #4h mesh iterations
        @inbounds for m = 1:h4_its
            @inbounds for k = 1:size(R4)[3], j = 1:size(R4)[2], i = 1:size(R4)[1]
                prvi, nxti = pbc_for_residuum(i-1, i+1, size(R4)[1])
                prvj, nxtj = pbc_for_residuum(j-1, j+1, size(R4)[2])
                prvk, nxtk = pbc_for_residuum(k-1, k+1, size(R4)[3]) 

                new_eps = (R4[i,j,k] + (EPS4[prvi,j,k] + EPS4[nxti,j,k])/(Δx4^2) + (EPS4[i,prvj,k] + EPS4[i,nxtj,k])/(Δy4^2) + (EPS4[i,j,prvk] + EPS4[i,j,nxtk])/(Δz4^2)) / (2/(Δx4^2) + 2/(Δy4^2) + 2/(Δz4^2))

                EPS4[i,j,k] = EPS4[i,j,k] + ω*(new_eps - EPS4[i,j,k])
            end 
        end

        # interpolation from 4h to 2h mesh
        @inbounds for k = 1:size(R2)[3], j = 1:size(R2)[2], i = 1:size(R2)[1]
            ir4 = ceil(Int,i/2)
            jr4 = ceil(Int,j/2)
            kr4 = ceil(Int,k/2)
            prvi, nxti = pbc_for_residuum(ir4-1, ir4+1, size(R4)[1])
            prvj, nxtj = pbc_for_residuum(jr4-1, jr4+1, size(R4)[2])
            prvk, nxtk = pbc_for_residuum(kr4-1, kr4+1, size(R4)[3])
            
            if i % 2 == 1 && j % 2 == 1 && k % 2 == 1
                EPS2[i,j,k] = EPS4[ir4,jr4,kr4]
            elseif i % 2 == 0 && j % 2 == 1 && k % 2 == 1
                EPS2[i,j,k] = 0.5*(EPS4[ir4,jr4,kr4] + EPS4[nxti,jr4,kr4])
            elseif i % 2 == 1 && j % 2 == 0 && k % 2 == 1
                EPS2[i,j,k] = 0.5*(EPS4[ir4,jr4,kr4] + EPS4[ir4,nxtj,kr4])
            elseif i % 2 == 1 && j % 2 == 1 && k % 2 == 0
                EPS2[i,j,k] = 0.5*(EPS4[ir4,jr4,kr4] + EPS4[ir4,jr4,nxtk])
            elseif i % 2 == 1 && j % 2 == 0 && k % 2 == 0
                EPS2[i,j,k] = 0.25*(EPS4[ir4,jr4,kr4] + EPS4[ir4,nxtj,kr4] + EPS4[ir4,jr4,nxtk] + EPS4[ir4,nxtj,nxtk])
            elseif i % 2 == 0 && j % 2 == 1 && k % 2 == 0
                EPS2[i,j,k] = 0.25*(EPS4[ir4,jr4,kr4] + EPS4[nxti,jr4,kr4] + EPS4[ir4,jr4,nxtk] + EPS4[nxti,jr4,nxtk])
            elseif i % 2 == 0 && j % 2 == 0 && k % 2 == 1
                EPS2[i,j,k] = 0.25*(EPS4[ir4,jr4,kr4] + EPS4[ir4,nxtj,kr4] + EPS4[nxti,jr4,kr4] + EPS4[nxti,nxtj,kr4])
            else
                EPS2[i,j,k] = 0.125*(EPS4[ir4,jr4,kr4] + EPS4[nxti,jr4,kr4] + EPS4[ir4,nxtj,kr4] + EPS4[ir4,jr4,nxtk] +
                                    EPS4[nxti,nxtj,kr4] + EPS4[nxti,jr4,nxtk] + EPS4[ir4,nxtj,nxtk] + EPS4[nxti,nxtj,nxtk])
            end
            
        end
        
        #2h mesh iterations
        @inbounds for m = 1:h2_its
            @inbounds for k = 1:size(R2)[3], j = 1:size(R2)[2], i = 1:size(R2)[1]
                prvi, nxti = pbc_for_residuum(i-1, i+1, size(R2)[1])
                prvj, nxtj = pbc_for_residuum(j-1, j+1, size(R2)[2])
                prvk, nxtk = pbc_for_residuum(k-1, k+1, size(R2)[3]) 

                new_eps = (R2[i,j,k] + (EPS2[prvi,j,k] + EPS2[nxti,j,k])/(Δx2^2) + (EPS2[i,prvj,k] + EPS2[i,nxtj,k])/(Δy2^2) + (EPS2[i,j,prvk] + EPS2[i,j,nxtk])/(Δz2^2)) / (2/(Δx2^2) + 2/(Δy2^2) + 2/(Δz2^2))

                EPS2[i,j,k] = EPS2[i,j,k] + ω*(new_eps - EPS2[i,j,k])
            end 
        end

        # interpolation from 2h to fine mesh
        @inbounds for k = 1:size(R1)[3], j = 1:size(R1)[2], i = 1:size(R1)[1]
            ir2 = ceil(Int,i/2)
            jr2 = ceil(Int,j/2)
            kr2 = ceil(Int,k/2)
            prvi, nxti = pbc_for_residuum(ir2-1, ir2+1, size(R2)[1])
            prvj, nxtj = pbc_for_residuum(jr2-1, jr2+1, size(R2)[2])
            prvk, nxtk = pbc_for_residuum(kr2-1, kr2+1, size(R2)[3])
            
            if i % 2 == 1 && j % 2 == 1 && k % 2 == 1
                EPS1[i,j,k] = EPS2[ir2,jr2,kr2]
            elseif i % 2 == 0 && j % 2 == 1 && k % 2 == 1
                EPS1[i,j,k] = 0.5*(EPS2[ir2,jr2,kr2] + EPS2[nxti,jr2,kr2])
            elseif i % 2 == 1 && j % 2 == 0 && k % 2 == 1
                EPS1[i,j,k] = 0.5*(EPS2[ir2,jr2,kr2] + EPS2[ir2,nxtj,kr2])
            elseif i % 2 == 1 && j % 2 == 1 && k % 2 == 0
                EPS1[i,j,k] = 0.5*(EPS2[ir2,jr2,kr2] + EPS2[ir2,jr2,nxtk])
            elseif i % 2 == 1 && j % 2 == 0 && k % 2 == 0
                EPS1[i,j,k] = 0.25*(EPS2[ir2,jr2,kr2] + EPS2[ir2,nxtj,kr2] + EPS2[ir2,jr2,nxtk] + EPS2[ir2,nxtj,nxtk])
            elseif i % 2 == 0 && j % 2 == 1 && k % 2 == 0
                EPS1[i,j,k] = 0.25*(EPS2[ir2,jr2,kr2] + EPS2[nxti,jr2,kr2] + EPS2[ir2,jr2,nxtk] + EPS2[nxti,jr2,nxtk])
            elseif i % 2 == 0 && j % 2 == 0 && k % 2 == 1
                EPS1[i,j,k] = 0.25*(EPS2[ir2,jr2,kr2] + EPS2[ir2,nxtj,kr2] + EPS2[nxti,jr2,kr2] + EPS2[nxti,nxtj,kr2])
            else
                EPS1[i,j,k] = 0.125*(EPS2[ir2,jr2,kr2] + EPS2[nxti,jr2,kr2] + EPS2[ir2,nxtj,kr2] + EPS2[ir2,jr2,nxtk] +
                                    EPS2[nxti,nxtj,kr2] + EPS2[nxti,jr2,nxtk] + EPS2[ir2,nxtj,nxtk] + EPS2[nxti,nxtj,nxtk])
            end
            
        end


        # update fine mesh potential
        @inbounds for k = 1:(NZ-1), j = 1:(NY-1), i = 1:(NX-1)
            ϕ[i,j,k] -= EPS1[i,j,k]
        end
    end

    if conv == false
        println("GS failed to converge, L2 = ", L2)
    end
end


function compute_electric_field!()
    @inbounds for k = 1:(NZ-1), j = 1:(NY-1), i = 1:(NX-1)
        prvi, nxti = periodic_boundary_conditions(i-1, i+1, NX)
        prvj, nxtj = periodic_boundary_conditions(j-1, j+1, NY)
        prvk, nxtk = periodic_boundary_conditions(k-1, k+1, NZ) 
        Ex[i,j,k] = (ϕ[prvi,j,k]- ϕ[nxti,j,k])/(2Δx)
        Ey[i,j,k] = (ϕ[i,prvj,k]- ϕ[i,nxtj,k])/(2Δy)
        Ez[i,j,k] = (ϕ[i,j,prvk] - ϕ[i,j,nxtk])/(2Δz)
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

function periodic_boundary_for_E(nxti, nxtj, nxtk)
    if nxti == NX
        nxti = 1
    end
    if nxtj == NY
        nxtj = 1
    end
    if nxtk == NZ
        nxtk = 1
    end
    return (nxti, nxtj, nxtk)
end

function sum_electric_field(E, wx, wy, wz, i, j, k)
    nxti, nxtj, nxtk = periodic_boundary_for_E(i+1, j+1, k+1)
    e  = (1-wx) * (1-wy) * (1-wz) * E[i,j,k]
    e += wx     * (1-wy) * (1-wz) * E[nxti,j,k]
    e += wx     * wy     * (1-wz) * E[nxti,nxtj,k]
    e += wx     * wy     * wz     * E[nxti,nxtj,nxtk]
    e += (1-wx) * wy     * (1-wz) * E[i,nxtj,k]
    e += (1-wx) * wy     * wz     * E[i,nxtj,nxtk]
    e += (1-wx) * (1-wy) * wz     * E[i,j,nxtk]
    e += wx     * (1-wy) * wz     * E[nxti,j,nxtk]
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

function timestep!(sp::Species, factor)
    clear!()
    charge_deposition!(sp)
    compute_charge_density!()
    compute_potential_multigrid!()
    compute_electric_field!()
    boris_pusher!(sp, factor)
end

function timestep!(sp_e::Species, sp_i::Species, factor::Float64)
    clear!()
    charge_deposition!(sp_e)
    charge_deposition!(sp_i)
    compute_charge_density!()
    compute_potential_multigrid!()
    compute_electric_field!()
    boris_pusher!(sp_e, factor)
    boris_pusher!(sp_i, factor)
end
end