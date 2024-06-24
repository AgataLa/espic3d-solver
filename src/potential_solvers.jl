using LinearAlgebra
using SparseArrays
using IncompleteLU
using FFTW

const max_it    = 10000                         # Gauss-Seidel for ϕ
const max_it_mul    = 100                         # Gauss-Seidel for ϕ
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
    last_L2 = 0.0
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
            if L2 < tolerance || abs(last_L2 - L2) < tolerance
                conv = true
                #println("Converged after $(m) iterations, L2 = ", L2)
                break
            end
            last_L2 = L2
        end
    end

    # if conv == false
    #     println("GS failed to converge, L2 = ", L2)
    # end
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
    last_L2 = 0.0
    conv = false

    fine_its = 3
    h2_its = 5
    h4_its = 10
    h8_its = 5

    for it = 1:max_it_mul
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
        if L2 < tolerance || abs(last_L2 - L2) < tolerance
            conv = true
            #println("Converged after $(it) cycles, L2 = ", L2)
            break
        end
        last_L2 = L2

        # 2h mesh restriction
        grid_restriction!(R1, R2)

        # 4h mesh restriction
        grid_restriction!(R2, R4)

        if @isdefined R8
            # 8h mesh restriction
            grid_restriction!(R4, R8)

            # 8h mesh iterations
            @inbounds for m = 1:h8_its
                gauss_seidel_sor!(EPS8, R8, 8Δx, 8Δy, 8Δz)
            end

            # interpolation from 8h to 4h mesh
            grid_interpolation!(EPS8, EPS4)
        end

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

        # fine mesh iterations
        @inbounds for m = 1:fine_its
            gauss_seidel_sor!(EPS1, R1, Δx, Δy, Δz)
        end

        # update fine mesh potential
        @inbounds for k = 1:size(ϕ)[3], j = 1:size(ϕ)[2], i = 1:size(ϕ)[1]
            ϕ[i,j,k] -= EPS1[i,j,k]
        end
    end

    # if conv == false
    #     println("GS multigrid failed to converge, L2 = ", L2)
    # end
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

    @inbounds for k in 1:(NZ-1), j in 1:(NY-1), i in 1:(NX-1)
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


function swap(x, i1, i2)
    temp = x[i1]
    x[i1] = x[i2]
    x[i2] = temp
end


function bit_reversal_permutation!(x, ip1, ip2, ip3)

    i2rev = 1
    @inbounds for i2 in 1:ip1:ip2
        if i2 < i2rev
            for i1 = i2:2:(i2+ip1-2)
                for i3 = i1:ip2:ip3
                    i3rev = i2rev + i3 - i2
                    swap(x, i3, i3rev)
                    swap(x, i3+1, i3rev+1)
                end
            end
        end
        ibit = ip2 >> 1
        while ibit >= ip1 && i2rev > ibit
            i2rev -= ibit
            ibit >>= 1
        end
        i2rev += ibit
    end
end

function fft1d_dim!(x, ip1, ip2, ip3, isign, N)
    ifp1 = ip1
    while ifp1 < ip2
        ifp2 = ifp1 << 1
        θ = -2π * isign / (ifp2 / ip1)
        wpr = -2.0 * sin(0.5θ)^2
        wpi = sin(θ)
        wr = 1.0
        wi = 0.0

        @inbounds for i3 = 1:ip1:ifp1
            for i1 = i3:2:(i3+ip1-2)
                for i2 = i1:ifp2:ip3
                    k1 = i2
                    k2 = k1 + ifp1
                    tempr = wr * x[k2]   - wi * x[k2+1]
                    tempi = wr * x[k2+1] + wi * x[k2]
                    x[k2] = x[k1] - tempr
                    x[k2+1] = x[k1+1] - tempi
                    x[k1] += tempr
                    x[k1+1] += tempi
                end
            end
            wrtemp = wr
            wr = wrtemp * wpr - wi    * wpi + wr
            wi = wi    * wpr + wrtemp * wpi + wi
        end
        ifp1 = ifp2
    end

    if isign == -1
        x ./= N
    end
end

function fft3d!(x, isign)

    nprev = 1
    for dim in 1:1:3
        N = dim == 1 ? size(x, dim) ÷ 2 : size(x, dim)
        nrem = (length(x) ÷ 2 ) ÷ (N*nprev)
        ip1 = nprev << 1
        ip2 = ip1 * N
        ip3 = ip2 * nrem

        bit_reversal_permutation!(x, ip1, ip2, ip3)
        fft1d_dim!(x, ip1, ip2, ip3, isign, N)

        nprev *= N
    end
end



function compute_potential_FFT!()
    ρ_cut = @view ρ[1:(NX-1),1:(NY-1),1:(NZ-1)]
    @inbounds for i in 1:length(ρ_cut)
        ρ̂[i*2-1] = ρ_cut[i]
        ρ̂[i*2] = 0.0
    end

    fft3d!(ρ̂, 1.)

    @inbounds for k in 1:size(ϕ̂, 3), j in 1:size(ϕ̂, 2), i in 1:size(ϕ̂, 1) ÷ 2
        if K2[i,j,k] != 0
            ϕ̂[i*2-1, j, k] = ρ̂[i*2-1, j, k] / K2[i,j,k]
            ϕ̂[i*2,   j, k] = ρ̂[i*2,   j, k] / K2[i,j,k]
        else
            ϕ̂[i*2-1, j, k] = 0.0
            ϕ̂[i*2,   j, k] = 0.0
        end
    end

    fft3d!(ϕ̂, -1.)

    @inbounds for i in 1:length(ϕ)
        ϕ[i] = ϕ̂[i*2-1]
    end
end