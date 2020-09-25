macro swap(a,b)
    esc(Expr(:(=), Expr(:tuple, a, b), Expr(:tuple, b, a)))
end

"""
MRI coil compression via PCA

Given multiple MRI surface coil images (`idata_orig`), use SVD/PCA to find a smaller number (`ncoil`) of virtual coil images.
"""
#=function ir_mri_coil_compress(idata_orig; ncoil=1)
    x, y, z = size(idata_orig)
    idata = reshape(idata_orig, x*y, z)
    ~, S, V = svd(idata, full=false);

    Vr = V[:, 1:ncoil]     # [z ncoil] compression matrix with rank = ncoil
    odata = idata * Vr     # [*N ncoil] compressed data
    odata = reshape(odata, x, y, ncoil) # [(N) ncoil]

    return odata, Vr, Diagonal(S)
end=#

"""
Generate radial sampling pattern

Arguments:
 - `n1, n2, n3`: image dimensions
 - `lines`: number of radial lines in sampling pattern
"""
function strucrand(n1, n2, n3, lines);
    samp = zeros(Int, n1, n2, n3)
    # Create an array of N points between -n1/2 and n1/2
    x = range(-n1/2, n1/2-1, length=n1)
    y = range(-n2/2, n2/2-1, length=length(x))
    for frame in 1:n3
        # Loop to traverse the kspace ; 0 to pi in steps of π/lines -- 
        # Succesive frames rotated by a small random angle (π/line)*rand()
        # Also, shift the cartesian locations such that the center
        # is now at (n1/2,n1/2) {Otherwise the center would be (0,0)}
        coords = [(round(Int,x[i]*sin(α)+n1/2+0.5),round(Int,y[i]*cos(α)+n2/2+0.5),frame)
            for i in 1:length(x), α in range(π/line*rand(), step=π/line, length=lines)]
        # Create the sampling pattern
        samp[CartesianIndex.(coords)] .= 1
    end
    return samp  
end

## Nonuniform FFT
# Julia implementation of the following matlab function: https://github.com/JeffFessler/reproduce-l-s-dynamic-mri/blob/master/operators/getEnufft.m

# Basis transform generator for getEnufft
function E_basis(basis::String;
        M::Int64, nt::Int64,
        fov::Union{Tuple{Int64,Int64},Nothing} = nothing,
        N::Union{Tuple{Int64,Int64},Nothing} = nothing,
        ksp::Union{Array{Complex{Float64}},Nothing} = nothing)
    if basis == "dirac"
        Bi = ones(M, nt)
    elseif basis == "sinc" || basis == "dirac*dx"
        Bi = ones(M, nt) * prod(fov ./ N, dim=1)
    elseif basis == "rect" # rect(x/dx) ⟺ |dx| * sinc(dx * u)
        dx = abs.(fov) ./ N # usual default
        Bi = ones(M, nt)
        for id = 1:size(ksp,2)
            if dx[id] ≠ 0
                Bi .*= sinc(dx[id] * ksp[:,id,:])
            end
        end
    else
        error("unknown basis_type \"$basis\"")
    end
    Bi
end

# Note that this function in not fully tested, it is just the raw transcription of the getEnufft Matlab function to Julia. I left it here for sake of completeness, but in the notebooks, I used a simplified version specialized for the given task.
function getEnufft_original(sense_maps::Array{Complex{Float64},3};
    nx::Int64 = size(sense_maps,1), ny::Int64 = size(sense_maps,2), nt::Int64,
    nc::Int64 = size(sense_maps,3), mask::Union{BitArray{3},Nothing} = nothing,
    samp::Array = [], N::Tuple{Int64,Int64} = (nx,ny), fov::Tuple{Int64,Int64} = (22,22),
    basis::String = "dirac", ksp::Array,
    M::Int64 = size(ksp,1),
    # nufft params
    om::Array = [], wi::Array = [], donufft::Bool = false, Jd::Array = [6,6],
    Kd::Tuple{Int64,Int64} = floor.(Int64, N.*1.5), n_shift::Tuple{Int64,Int64} = N .÷ 2)
    
    out_basistransform = E_basis(basis, M=M, nt=nt, fov=fov, N=N, ksp=ksp)
    basistransform = donufft ? out_basistransform :
        reshape(out_basistransform, N..., nt)
    
    if donufft
        # input ksp, om w/ size [M,d,nt], and wi w/ size[M,nt]
        isempty(om) && error("cl: need ksp,om,wi for nufft")
        # construct st for nufft
        size(ksp,3) ≠ nt && error("cl: double check ksp dimension")
        st_forw = Array{Any}(undef, size(ksp,3))
        st_backw = Array{Any}(undef, size(ksp,3))
        for tt = 1:size(ksp,3)
            st_forw[tt], st_backw[tt],_ =
                nufft_init(om[:,:,tt], N, n_shift=collect(n_shift))
        end
        E = LinearMap{Complex{Float64}}(
            x -> begin
                S = zeros(M,nt,nc)
                x = reshape(x, nx, ny, nt, 1)
                for tt=1:nt
                    tmp = x[:,:,tt,:] .* sense_maps
                    S[:,tt,:] = reshape(st_forw[tt](tmp)./sqrt(prod(N)),M,1,nc)
                end,
                S .* basistransform
            end,
            S -> begin
                x = zeros(Complex{Float64}, nx,ny,nt)
                S = reshape(S, M,nt,nc)
                S = S .* conj.(basistransform)
                wi ≠ [] && (S = S .* wi) # cl: from otazo
                for tt = 1:nt # cl: '/sqrt(prod(a.imSize))' from otazo
                    tmp = reshape(st_backw[tt](S[:,tt,:])/sqrt(prod(N)),nx,ny,nc)
                    x[:,:,tt] = sum(tmp .* conj.(sense_maps),dims=3)
                end
                x
            end,
            M*nt*nc, nx*ny*nt
        )
    else
        E = LinearMap{Complex{Float64}}(
            x -> begin
                x = reshape(x, nx, ny, nt, 1)
                S = x .* reshape(sense_maps,nx,ny,1,nc)
                S = fftshift(ifft(ifftshift(S,1),1),1)*√(nx)
                S = fftshift(ifft(ifftshift(S,2),2),2)*√(ny)
                samp ≠ nothing && (S = S .* samp) # cl: samp mask only when cartesian samp
                S
            end,
            S -> begin
                S = S .* conj.(basistransform)
                wi ≠ [] && (S = S .* wi) # cl: from otazo
                s = fftshift(fft(ifftshift(S,1),1),1)/√(nx)
                s = fftshift(fft(ifftshift(s,2),2),2)/√(ny)
                dropdims(sum(s .* reshape(conj.(sense_maps),nx,ny,1,nc),dims=4),dims=4)
            end,
            M*nt*nc, nx*ny*nt
        )
    end
    
    E, out_basistransform
end


function fftshift!(
        output::AbstractArray,
        input::AbstractArray,
        dims::NTuple{N,Int}) where {N}
    
    @assert input !== output "input and output must be two distinct arrays"
    @assert any(dims .> 0) "dims can contain only positive values!"
    @assert any(dims .<= ndims(input)) "dims cannot contain larger value than ndims(input) (=$(ndims(input)))"
    @assert size(output) == size(input) "input and output must have the same size"
    @assert eltype(output) == eltype(input) "input and output must have the same eltype"
    
    shifts = [dim in dims ? size(input, dim) ÷ 2 : 0 for dim in 1:ndims(input)]
    circshift!(output, input, shifts)
    
end

function ifftshift!(
        output::AbstractArray,
        input::AbstractArray,
        dims::NTuple{N,Int}) where {N}
    
    @assert input !== output "input and output must be two distinct arrays"
    @assert any(dims .> 0) "dims can contain only positive values!"
    @assert any(dims .<= ndims(input)) "dims cannot contain larger value than ndims(input) (=$(ndims(input)))"
    @assert size(output) == size(input) "input and output must have the same size"
    @assert eltype(output) == eltype(input) "input and output must have the same eltype"
    
    shifts = [dim in dims ? size(input, dim) ÷ 2 + size(input, dim) % 2 : 0 for dim in 1:ndims(input)]
    circshift!(output, input, shifts)
    
end

fftshift!(output::AbstractArray, input::AbstractArray, dims::Int) =
    fftshift!(output, input, (dims,))

ifftshift!(output::AbstractArray, input::AbstractArray, dims::Int) =
    ifftshift!(output, input, (dims,))

using NFFT, MIRT, EllipsisNotation

function nufft_plan(w::AbstractMatrix{<:Real}, N::Dims;
        n_shift::AbstractVector{<:Real} = zeros(Int, length(N)),
        nfft_m::Int = 4,
        nfft_sigma::Real = 1.25,
        pi_error::Bool = true,
        do_many::Bool = true)

    any(N .< 6) && throw("NFFT may be erroneous for small N")
    any(isodd.(N)) && throw("NFFT erroneous for odd N")
    pi_error && any(abs.(w) .> π) && throw(ArgumentError("|w| > π is likely an error"))

    M,D = size(w)
    length(N) != D && throw(DimensionMismatch("length(N) vs D=$D"))
    length(n_shift) != D && throw(DimensionMismatch("length(n_shift) vs D=$D"))

    T = MIRT.nufft_eltype(w)
    CT = Complex{T}
    f = convert.(T, w/(2π)) # note: NFFTPlan must have correct type
    p = NFFTPlan(f', N, nfft_m, nfft_sigma) # create plan

    # extra phase here because NFFT always starts from -N/2
    phasor = CT.(cis.(-w * (collect(N)/2. - n_shift)))
    phasor_conj = conj.(phasor)
    phasor_buffer = similar(phasor_conj)
    forw1! = (b, x) -> b .= NFFT.nfft!(p, MIRT.nufft_typer(CT, x), b) .* phasor
    backw1! = (b, y) -> NFFT.nfft_adjoint!(p, MIRT.nufft_typer(CT, phasor_buffer .= y .* phasor_conj), b)
    
    if do_many
        forw! = (b, x) -> begin
                for i in axes(x, ndims(x))
                    @views forw1!(b[.., i], x[.., i])
                end
            end
        backw! = (b, y) -> begin
                for i in axes(y, ndims(y))
                    @views backw1!(b[.., i], y[.., i])
                end
            end
    else
        forw! = forw1!
        backw! = backw1!
    end

    return (nufft=forw!, nufft_adjoint=backw!)
end

import LinearAlgebra.LAPACK

import LinearAlgebra.BLAS.@blasfunc

import LinearAlgebra: BlasFloat, BlasInt, LAPACKException,
    DimensionMismatch, SingularException, PosDefException, chkstride1, checksquare

using LinearAlgebra: triu, tril, dot

using Base: iszero, require_one_based_indexing

for (geev, gesvd, gesdd, ggsvd, elty, relty) in
    ((:dgeev_,:dgesvd_,:dgesdd_,:dggsvd_,:Float64,:Float64),
     (:sgeev_,:sgesvd_,:sgesdd_,:sggsvd_,:Float32,:Float32),
     (:zgeev_,:zgesvd_,:zgesdd_,:zggsvd_,:ComplexF64,:Float64),
     (:cgeev_,:cgesvd_,:cgesdd_,:cggsvd_,:ComplexF32,:Float32))
    @eval begin
        #    SUBROUTINE DGESDD( JOBZ, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK,
        #                   LWORK, IWORK, INFO )
        #*     .. Scalar Arguments ..
        #      CHARACTER          JOBZ
        #      INTEGER            INFO, LDA, LDU, LDVT, LWORK, M, N
        #*     ..
        #*     .. Array Arguments ..
        #      INTEGER            IWORK( * )
        #      DOUBLE PRECISION   A( LDA, * ), S( * ), U( LDU, * ),
        #                        VT( LDVT, * ), WORK( * )
        function gesdd!(job::AbstractChar, A::AbstractMatrix{$elty},
                U::AbstractMatrix{$elty}, S::AbstractVector{$relty}, VT::AbstractMatrix{$elty})
            require_one_based_indexing(A)
            chkstride1(A)
            m, n   = size(A)
            minmn  = min(m, n)
            
            #=if job == 'A'
                U  = similar(A, $elty, (m, m))
                VT = similar(A, $elty, (n, n))
            elseif job == 'S'
                U  = similar(A, $elty, (m, minmn))
                VT = similar(A, $elty, (minmn, n))
            elseif job == 'O'
                U  = similar(A, $elty, (m, m >= n ? 0 : m))
                VT = similar(A, $elty, (n, m >= n ? n : 0))
            else
                U  = similar(A, $elty, (m, 0))
                VT = similar(A, $elty, (n, 0))
            end=#
            
            if job == 'A'
                @assert size(U) == (m, m)
                @assert size(VT) == (n, n)
            elseif job == 'S'
                @assert size(U) == (m, minmn)
                @assert size(VT) == (minmn, n)
            elseif job == 'O'
                @assert size(U) == (m,  m >= n ? 0 : m)
                @assert size(VT) == (n, m >= n ? n : 0)
            else
                @assert size(U) == (m, 0)
                @assert size(VT) == (n, 0)
            end
            @assert eltype(U) <: $elty
            @assert eltype(VT) <: $elty
            work   = Vector{$elty}(undef, 1)
            lwork  = BlasInt(-1)
            @assert size(S, 1) == minmn
            @assert eltype(S) <: $relty
            #S      = similar(A, $relty, minmn)
            cmplx  = eltype(A)<:Complex
            if cmplx
                rwork = Vector{$relty}(undef, job == 'N' ? 7*minmn : minmn*max(5*minmn+7, 2*max(m,n)+2*minmn+1))
            end
            iwork  = Vector{BlasInt}(undef, 8*minmn)
            info   = Ref{BlasInt}()
            for i = 1:2  # first call returns lwork as work[1]
                if cmplx
                    ccall((@blasfunc($gesdd), LinearAlgebra.LAPACK.liblapack), Cvoid,
                          (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                           Ref{BlasInt}, Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$relty}, Ptr{BlasInt}, Ptr{BlasInt}),
                          job, m, n, A, max(1,stride(A,2)), S, U, max(1,stride(U,2)), VT, max(1,stride(VT,2)),
                          work, lwork, rwork, iwork, info)
                else
                    ccall((@blasfunc($gesdd), LinearAlgebra.LAPACK.liblapack), Cvoid,
                          (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                           Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Ptr{BlasInt}),
                          job, m, n, A, max(1,stride(A,2)), S, U, max(1,stride(U,2)), VT, max(1,stride(VT,2)),
                          work, lwork, iwork, info)
                end
                LinearAlgebra.LAPACK.chklapackerror(info[])
                if i == 1
                    # Work around issue with truncated Float32 representation of lwork in
                    # sgesdd by using nextfloat. See
                    # http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=13&t=4587&p=11036&hilit=sgesdd#p11036
                    # and
                    # https://github.com/scipy/scipy/issues/5401
                    lwork = round(BlasInt, nextfloat(real(work[1])))
                    resize!(work, lwork)
                end
            end
            if job == 'O'
                if m >= n
                    return (A, S, VT)
                else
                    # ()__
                    # ||::Z__
                    # ||::|:::Z____
                    # ||::|:::|====|
                    # ||==|===|====|
                    # ||""|===|====|
                    # ||  `"""|====|
                    # ||      `""""`
                    return (U, S, A)
                end
            end
            return (U, S, VT)
        end
    end
end

function LinearAlgebra.LinearAlgebra.svd!(A::StridedMatrix{T}, buffer::SVD;
        full::Bool = false, alg::LinearAlgebra.Algorithm = LinearAlgebra.default_svd_alg(A)) where {T<:BlasFloat, Tr<:Real}
    m,n = size(A)
    if m == 0 || n == 0
        u,s,vt = (Matrix{T}(I, m, full ? m : n), real(zeros(T,0)), Matrix{T}(I, n, n))
    else
        u,s,vt = _svd!(A,buffer,full,alg)
    end
    SVD(u,s,vt)
end

_svd!(A::StridedMatrix{T}, buffer::SVD, full::Bool, alg::LinearAlgebra.DivideAndConquer) where T<:BlasFloat = gesdd!(full ? 'A' : 'S', A, buffer.U, buffer.S, buffer.Vt)
