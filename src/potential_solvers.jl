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
    R2 = zeros(Float64, ceil((NX-1)/2), ceil((NY-1)/2), ceil((NZ-1)/2))
    EPS2 = zeros(Float64, ceil((NX-1)/2), ceil((NY-1)/2), ceil((NZ-1)/2))
    R4 = zeros(Float64, ceil((NX-1)/4), ceil((NY-1)/4), ceil((NZ-1)/4))
    EPS4 = zeros(Float64, ceil((NX-1)/4), ceil((NY-1)/4), ceil((NZ-1)/4))

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

function pbc_for_residuum(prv, nxt, N)
    if prv == 0
        prv = N
    end
    if nxt == N
        nxt = 1
    end
    return (prv, nxt)
end

function compute_potential_NRPCG!()
    L2 = 0.0
    conv = false

    @inbounds for m = 1:max_it
        ϕ_old .= ϕ
        @inbounds for k = 1:(NZ-1), j = 1:(NY-1), i = 1:(NX-1)
            prvi, nxti = periodic_boundary_conditions(i-1, i+1, NX)
            prvj, nxtj = periodic_boundary_conditions(j-1, j+1, NY)
            prvk, nxtk = periodic_boundary_conditions(k-1, k+1, NZ) 
            ϕ[i,j,k] = ((ϕ[prvi,j,k] + ϕ[nxti,j,k])/(Δx^2) + (ϕ[i,prvj,k] + ϕ[i,nxtj,k])/(Δy^2) + (ϕ[i,j,prvk] + ϕ[i,j,nxtk])/(Δz^2) + ρ[i,j,k]/ε_0) / (2/(Δx^2) + 2/(Δy^2) + 2/(Δz^2))

            ϕ[i,j,k] = ϕ_old[i,j,k] + ω*(ϕ[i,j,k] - ϕ_old[i,j,k])
        end

        if m % 25 == 0
            sum = 0
            @inbounds for k = 1:(NZ-1), j = 1:(NY-1), i = 1:(NX-1)
                prvi, nxti = periodic_boundary_conditions(i-1, i+1, NX)
                prvj, nxtj = periodic_boundary_conditions(j-1, j+1, NY)
                prvk, nxtk = periodic_boundary_conditions(k-1, k+1, NZ)
                r = -ϕ[i,j,k]*(2/(Δx^2) + 2/(Δy^2) + 2/(Δz^2)) + ρ[i,j,k]/ε_0 + (ϕ[prvi,j,k] + ϕ[nxti,j,k])/(Δy^2) + (ϕ[i,prvj,k] + ϕ[i,nxtj,k])/(Δx^2) + (ϕ[i,j,prvk] + ϕ[i,j,nxtk])/(Δz^2)
                sum += r^2
            end
            L2 = sqrt(sum / ((NX-1)*(NY-1)*(NZ-1)))
            if L2 < tolerance
                conv = true
                break
            end
        end
    end

    if conv == false
        println("GS failed to converge, L2 = ", L2)
    end
end