using LinearAlgebra
using SparseArrays
using IncompleteLU
using FFTW

const max_it    = 10000                         # Gauss-Seidel for ϕ
const ω         = 1.4                           # SOR
const tolerance = 1e-8                          # L2 tolerance


function pbc(A, i, j, k)
    if i == 0
        i = size(A)[1]
    elseif i > size(A)[1]
        i = 1
    end
    if j == 0
        j = size(A)[2]
    elseif j > size(A)[2]
        j = 1
    end
    if k == 0
        k = size(A)[3]
    elseif k > size(A)[3]
        k = 1
    end
    return A[i,j,k]
end


# GAUSS-SEIDEL SOR
function gauss_seidel_sor!(A, B, dx, dy, dz)
    A_new = 0.0
    @inbounds for k = 1:size(A)[3], j = 1:size(A)[2], i = 1:size(A)[1]
        A_new = ((pbc(A,i-1,j,k) + pbc(A,i+1,j,k))/(dx^2) +
                (pbc(A,i,j-1,k) + pbc(A,i,j+1,k))/(dy^2) +
                (pbc(A,i,j,k-1) + pbc(A,i,j,k+1))/(dz^2) + 
                B[i,j,k]) / (2/(dx^2) + 2/(dy^2) + 2/(dz^2))

        A[i,j,k] = A[i,j,k] + ω*(A_new - A[i,j,k])
    end
end

function compute_potential!()
    L2 = 0.0
    conv = false

    @inbounds for m = 1:max_it
        gauss_seidel_sor!(ϕ, ρ, Δx, Δy, Δz)

        if m % 25 == 0
            sum_L2 = 0
            @inbounds for k = 1:size(ϕ)[3], j = 1:size(ϕ)[2], i = 1:size(ϕ)[1]
                r = -ϕ[i,j,k] + (ρ[i,j,k] +
                (pbc(ϕ,i-1,j,k) + pbc(ϕ,i+1,j,k))/(Δx^2) +
                (pbc(ϕ,i,j-1,k) + pbc(ϕ,i,j+1,k))/(Δy^2) +
                (pbc(ϕ,i,j,k-1) + pbc(ϕ,i,j,k+1))/(Δz^2)) / (2/(Δx^2) + 2/(Δy^2) + 2/(Δz^2))
                sum_L2 += r^2
            end
            L2 = sqrt(sum_L2 / (size(ϕ)[1]*size(ϕ)[2]*size(ϕ)[3]))
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



# GAUSS-SEIDEL SOR WITH MULTIGRID
function grid_restriction!(F, C)
    @inbounds for k = 1:2:size(F)[3], j = 1:2:size(F)[2], i = 1:2:size(F)[1]
        C[ceil(Int,i/2),ceil(Int,j/2),ceil(Int,k/2)] = (6*pbc(F,i,j,k) +
                                                        pbc(F,i-1,j,k) + pbc(F,i+1,j,k) +
                                                        pbc(F,i,j-1,k) + pbc(F,i,j+1,k) +
                                                        pbc(F,i,j,k-1) + pbc(F,i,j,k+1)) / 12.0
    end
end

function grid_interpolation!(C, F)
    @inbounds for k = 1:size(F)[3], j = 1:size(F)[2], i = 1:size(F)[1]
        ic = ceil(Int,i/2)
        jc = ceil(Int,j/2)
        kc = ceil(Int,k/2)
        
        if i % 2 == 1 && j % 2 == 1 && k % 2 == 1
            F[i,j,k] = C[ic,jc,kc]
        elseif i % 2 == 0 && j % 2 == 1 && k % 2 == 1
            F[i,j,k] = 0.5*(C[ic,jc,kc] + pbc(C,ic+1,jc,kc))
        elseif i % 2 == 1 && j % 2 == 0 && k % 2 == 1
            F[i,j,k] = 0.5*(C[ic,jc,kc] + pbc(C,ic,jc+1,kc))
        elseif i % 2 == 1 && j % 2 == 1 && k % 2 == 0
            F[i,j,k] = 0.5*(C[ic,jc,kc] + pbc(C,ic,jc,kc+1))
        elseif i % 2 == 1 && j % 2 == 0 && k % 2 == 0
            F[i,j,k] = 0.25*(C[ic,jc,kc] + pbc(C,ic,jc+1,kc) + pbc(C,ic,jc,kc+1) + pbc(C,ic,jc+1,kc+1))
        elseif i % 2 == 0 && j % 2 == 1 && k % 2 == 0
            F[i,j,k] = 0.25*(C[ic,jc,kc] + pbc(C,ic+1,jc,kc) + pbc(C,ic,jc,kc+1) + pbc(C,ic+1,jc,kc+1))
        elseif i % 2 == 0 && j % 2 == 0 && k % 2 == 1
            F[i,j,k] = 0.25*(C[ic,jc,kc] + pbc(C,ic+1,jc,kc) + pbc(C,ic,jc+1,kc) + pbc(C,ic+1,jc+1,kc))
        else
            F[i,j,k] = 0.125*(C[ic,jc,kc] + pbc(C,ic+1,jc,kc) + pbc(C,ic,jc+1,kc) + pbc(C,ic,jc,kc+1) +
                            pbc(C,ic+1,jc+1,kc) + pbc(C,ic+1,jc,kc+1) + pbc(C,ic,jc+1,kc+1) + pbc(C,ic+1,jc+1,kc+1))
        end
    end
end


function compute_potential_multigrid!()
    L2 = 0.0
    conv = false

    fine_its = 3
    h2_its = 6
    h4_its = 10
    h8_its = 20

    for it = 1:max_it
        #1 fine mesh iterations
        @inbounds for m = 1:fine_its
            gauss_seidel_sor!(ϕ, ρ, Δx, Δy, Δz)
        end

        #1 fine mesh residuum and conv check
        sum_L2 = 0
        @inbounds for k = 1:size(ϕ)[3], j = 1:size(ϕ)[2], i = 1:size(ϕ)[1]
            R1[i,j,k] = -ϕ[i,j,k] + (ρ[i,j,k] +
            (pbc(ϕ,i-1,j,k) + pbc(ϕ,i+1,j,k))/(Δx^2) +
            (pbc(ϕ,i,j-1,k) + pbc(ϕ,i,j+1,k))/(Δy^2) +
            (pbc(ϕ,i,j,k-1) + pbc(ϕ,i,j,k+1))/(Δz^2)) / (2/(Δx^2) + 2/(Δy^2) + 2/(Δz^2))
            sum_L2 += R1[i,j,k]^2
        end
        L2 = sqrt(sum_L2 / (size(ϕ)[1]*size(ϕ)[2]*size(ϕ)[3]))
        if L2 < tolerance
            conv = true
            println("Converged after $(it) cycles, L2 = ", L2)
            break
        end

        # 2h mesh restriction
        grid_restriction!(R1, R2)

        # 4h mesh restriction
        grid_restriction!(R2, R4)

        # 8h mesh restriction
        grid_restriction!(R4, R8)

        # 8h mesh iterations
        @inbounds for m = 1:h8_its
            gauss_seidel_sor!(EPS8, R8, 8Δx, 8Δy, 8Δz)
        end

        # interpolation from 8h to 4h mesh
        grid_interpolation!(EPS8, EPS4)

        # 4h mesh iterations
        @inbounds for m = 1:h4_its
            gauss_seidel_sor!(EPS4, R4, 4Δx, 4Δy, 4Δz)
        end

        # interpolation from 4h to 2h mesh
        grid_interpolation!(EPS4, EPS2)
        
        # 2h mesh iterations
        @inbounds for m = 1:h2_its
            gauss_seidel_sor!(EPS2, R2, 2Δx, 2Δy, 2Δz)
        end

        # interpolation from 2h to fine mesh
        grid_interpolation!(EPS2, EPS1)


        # update fine mesh potential
        @inbounds for k = 1:size(ϕ)[3], j = 1:size(ϕ)[2], i = 1:size(ϕ)[1]
            ϕ[i,j,k] -= EPS1[i,j,k]
        end
    end

    if conv == false
        println("GS failed to converge, L2 = ", L2)
    end
end


# PRECONDITIONED CONJUGATE GRADIENT (PCG)

function build_coefficient_matrix_pbc!(A, nx, ny, nz)
    idx2 = 1/Δx^2
    idy2 = 1/Δy^2
    idz2 = 1/Δz^2

    for k = 1:nz, j = 1:ny, i = 1:nx
        u = i + (j-1)*nx + (k-1)*nx*ny
        #println(i, " ", j, " ", k)
        #println(u)
        
        A[u,u] = -2.0*(idx2 + idy2 + idz2)
        if i == 1                  # i-1
            A[u,u+nx-1] = idx2
        else
            A[u,u-1] = idx2
        end
        if i == nx                 # i+1
            A[u,u-nx+1] = idx2      
        else
            A[u,u+1] = idx2
        end
        if j == 1                  # j-1
            A[u,u+(ny-1)*nx] = idy2   
        else
            A[u,u-nx] = idy2
        end
        if j == ny                  # j+1
            A[u,u-(ny-1)*nx] = idy2 
        else
            A[u,u+nx] = idy2
        end
        if k == 1                  # k-1
            A[u,u+(nz-1)*nx*ny] = idz2   
        else
            A[u,u-nx*ny] = idz2
        end
        if k == nz                  # k+1
            A[u,u-(nz-1)*nx*ny] = idz2
        else
            A[u,u+nx*ny] = idz2
        end
    end
end

function build_coefficient_matrix_4th_order_pbc!(A, nx, ny, nz)
    idx2 = 1/(12*Δx^2)
    idy2 = 1/(12*Δy^2)
    idz2 = 1/(12*Δz^2)

    for k = 1:nz, j = 1:ny, i = 1:nx
        #println(i, " ", j, " ", k)
        u = i + (j-1)*nx + (k-1)*nx*ny
        #println(i, " ", j, " ", k)
        #println(u)
        
        A[u,u] = -30.0*(idx2 + idy2 + idz2)

        if i == 1                  # i-1
            A[u,u+nx-2] = -idx2
            A[u,u+nx-1] = 16*idx2
        elseif i == 2
            A[u,u+nx-2] = -idx2
            A[u,u-1] = 16*idx2
        else
            A[u,u-2] = -idx2
            A[u,u-1] = 16*idx2
        end

        if i == nx                  # i-1
            A[u,u-nx+2] = -idx2
            A[u,u-nx+1] = 16*idx2
        elseif i == (nx-1)
            A[u,u-nx+2] = -idx2
            A[u,u+1] = 16*idx2
        else
            A[u,u+2] = -idx2
            A[u,u+1] = 16*idx2
        end

        if j == 1                  # i-1
            A[u,u+(ny-2)*nx] = -idy2 
            A[u,u+(ny-1)*nx] = 16*idy2
        elseif j == 2
            A[u,u+(ny-2)*nx] = -idy2 
            A[u,u-nx] = 16*idy2
        else
            A[u,u-2*nx] = -idy2
            A[u,u-nx] = 16*idy2
        end

        if j == ny                  # i-1
            A[u,u-(ny-2)*nx] = -idy2 
            A[u,u-(ny-1)*nx] = 16*idy2
        elseif j == (ny-1)
            A[u,u-(ny-2)*nx] = -idy2 
            A[u,u+nx] = 16*idy2
        else
            A[u,u+2*nx] = -idy2
            A[u,u+nx] = 16*idy2
        end

        if k == 1                  # i-1
            A[u,u+(nz-2)*nx*ny] = -idz2 
            A[u,u+(nz-1)*nx*ny] = 16*idz2
        elseif k == 2
            A[u,u+(nz-2)*nx*ny] = -idz2 
            A[u,u-nx*ny] = 16*idz2
        else
            A[u,u-2*nx*ny] = -idz2
            A[u,u-nx*ny] = 16*idz2
        end

        if k == ny                  # i-1
            A[u,u-(nz-2)*nx*ny] = -idz2 
            A[u,u-(nz-1)*nx*ny] = 16*idz2
        elseif k == (ny-1)
            A[u,u-(nz-2)*nx*ny] = -idz2 
            A[u,u+nx*ny] = 16*idz2
        else
            A[u,u+2*nx*ny] = -idz2
            A[u,u+nx*ny] = 16*idz2
        end
        
    end
end


function build_coefficient_matrix!(A, nx, ny, nz)
    idx2 = 1/Δx^2
    idy2 = 1/Δy^2
    idz2 = 1/Δz^2

    for k in 1:nz, j in 1:ny, i in 1:nx
        u = i + (j-1)*nx + (k-1)*nx*ny

        A[u, u] = -2.0 * (idx2 + idy2 + idz2)
        if i > 1
            A[u, u-1] = idx2
        end
        if i < nx
            A[u, u+1] = idx2
        end
        if j > 1
            A[u, u-nx] = idy2
        end
        if j < ny
            A[u, u+nx] = idy2
        end
        if k > 1
            A[u, u-nx*ny] = idz2
        end
        if k < nz
            A[u, u+nx*ny] = idz2
        end
    end
end


function build_b_vector!(b, A)
    nx = (size(A)[1]-1)
    ny = (size(A)[2]-1)
    nz = (size(A)[3]-1)

    for k = 1:nz, j = 1:ny, i = 1:nx
        u = i + (j-1)*nx + (k-1)*nx*ny
        b[u] = -A[i,j,k]
    end
end

function jacobi_preconditioner(A)
    return Diagonal([1/A[i, i] for i in 1:size(A)[1]])
end

function incomplete_LU_preconditioner(A)
    return ilu(A, τ = 0.001).L
end

function apply_solution_to_ϕ!(x)
    for i in eachindex(x)
        ϕ[i] = x[i]
    end
end

normL2(x) = sqrt(sum(x.^2))

function compute_potential_PCG!()
    conv = false
    build_coefficient_matrix_pbc!(APCG, size(ϕ)[1], size(ϕ)[2], size(ϕ)[3])
    #println("COND A: ", cond(Array(APCG)))
    # println("COND A: ", cond(Array(APCG), 1))
    build_b_vector!(bPCG, ρ)
    M = incomplete_LU_preconditioner(APCG)
    x = zeros(Float64, size(bPCG))

    g = APCG*x - bPCG
    s = M*g
    d = -s

    L2 = normL2(g)
    #println(L2)
    it = 1
    while L2 > tolerance && it < max_it
        z = APCG*d
        α = dot(g,s)
        β = dot(d,z)
        x = x + (α/β)*d
        g = g + (α/β)*z
        s = M*g
        β = α
        α = dot(g,s)
        d = (α/β)*d - s
        
        L2 = normL2(g)
        if L2 < tolerance
            conv = true
            break
        end
        it += 1
    end

    if conv == false
        println("PCG failed to converge, L2 = $(L2)")
    end

    apply_solution_to_ϕ!(x)
end