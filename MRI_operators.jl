using LinearAlgebra, FFTW, FunctionOperators
include("helper_functions.jl")

function getE(nx::Int, ny::Int, nt::Int, nc::Int, samp::AbstractArray{T,3} where T,
        smap::AbstractArray{Complex{T},3} where T, dType::Type)
    scaling = convert(dType, √(nx*ny))
    mask_with_scaling = repeat(samp, 1, 1, 1, nc) .* scaling
    reshaped_smap = reshape(convert.(dType, smap), nx, ny, 1, nc)
    reshaped_smap_conj_with_scaling = conj.(reshaped_smap) ./ scaling
    xcoils₁ = Array{dType}(undef, nx, ny, nt, nc)
    xcoils₂ = Array{dType}(undef, nx, ny, nt, nc)
    FFT_plan = plan_fft(xcoils₁, (1,2))
    iFFT_plan = inv(FFT_plan)
    E = FunctionOperator{dType}(name = "E", 
        forw = (b, x) -> begin # Don't ask me, why did Mr Otazo use ifft instead of fft...
                xcoils₁ .= reshape(x, (nx, ny, nt, 1)) .* reshaped_smap
                ifftshift!(xcoils₂, xcoils₁, (1, 2))
                mul!(xcoils₁, iFFT_plan, xcoils₂)
                fftshift!(b, xcoils₁, (1, 2))
                b .*= mask_with_scaling
            end,
        backw = (b, y) -> begin # But he used it consistently, so it doesn't make a big difference
                ifftshift!(xcoils₂, y, (1, 2))
                mul!(xcoils₁, FFT_plan, xcoils₂)
                fftshift!(xcoils₂, xcoils₁, (1, 2))
                xcoils₂ .*= reshaped_smap_conj_with_scaling
                sum!(reshape(b, (nx, ny, nt, 1)), xcoils₂)
                b
            end,
        inDims = (nx, ny, nt), outDims = (nx, ny, nt, nc))
end

function getΩQC(nx::Int, ny::Int, nt::Int, nc::Int, samp::AbstractArray{T,3} where T,
        smap::AbstractArray{Complex{T},3} where T, dType::Type)
    mask = repeat(convert.(dType, samp), 1, 1, 1, nc)
    reshaped_smap = reshape(convert.(dType, smap), nx, ny, 1, nc)
    reshaped_smap_conj = conj.(reshaped_smap)
    xcoils = Array{dType}(undef, nx, ny, nt, nc)
    FFT_plan = plan_fft!(xcoils, (1,2))
    iFFT_plan = inv(FFT_plan)
    scaling = convert(dType, √(nx*ny))
    C = FunctionOperator{dType}(name = "C",
        forw = (b, x) -> begin
                    b .= x .* reshaped_smap
            end,
        backw = (b, y) -> begin
                    xcoils .= y .* reshaped_smap_conj
                    sum!(b, xcoils)
            end,
        inDims = (nx, ny, nt, 1), outDims = (nx, ny, nt, nc))
    
    Q = FunctionOperator{dType}(name = "Q",
        forw = (b, x) -> begin
                ifftshift!(xcoils, x, (1, 2))
                iFFT_plan * xcoils
                fftshift!(b, xcoils, (1, 2))
                b .*= scaling
            end,
        backw = (b, y) -> begin
                ifftshift!(xcoils, reshape(y, nx, ny, nt, nc), (1, 2))
                FFT_plan * xcoils
                fftshift!(b, xcoils, (1, 2))
                b ./= scaling
            end,
        inDims = (nx, ny, nt, nc), outDims = (nx, ny, nt, nc))
    
    Ω = FunctionOperator{dType}(name = "Ω",
        forw = (b, x) -> b .= x .* mask,
        backw = (b, y) -> b .= y,
        inDims = (nx, ny, nt, nc), outDims = (nx, ny, nt, nc))
    
    return Ω, Q, C
end

function getEnufft(sense_maps::AbstractArray{Complex{T},3} where T; ksp::AbstractArray, om::AbstractArray,
        wi::Union{AbstractArray, Nothing} = nothing, dType::Type)
    
    nx,ny,nc = size(sense_maps)
    M,_,nt = size(ksp)
    n_shift = (nx, ny) .÷ 2
    
    sense_maps = convert.(dType, reshape(sense_maps, nx, ny, 1, nc))
    sense_maps_conj = conj.(sense_maps)
    !(wi isa Nothing) && (wi = convert.(dType, wi))
    
    #basistransform = E_basis("dirac", M=M, nt=nt, fov=(22,22), N=(nx,ny), ksp=ksp)
    st = [nufft_plan(@view(om[:, :, tt]), (nx, ny), nfft_m=2, nfft_sigma=1.25, n_shift=collect(n_shift))
        for tt = 1:nt]
    
    xcoils = Array{dType}(undef, nx, ny, nt, nc)
    xcoilsₜ = similar(xcoils, nx, ny, nc)
    kspₜ = similar(xcoils, M, nc)
    ksp_buffer = Array{dType}(undef, M, nt, nc)
    scaling = convert(dType, √(nx*ny))
    
    E = FunctionOperator{dType}(name = "E",
        forw = (b, x) -> begin
            xcoils .= reshape(x, nx, ny, nt, 1) .* sense_maps
            for tt=1:nt
                xcoilsₜ .= @view xcoils[:, :, tt, :]
                st[tt].nufft(kspₜ, xcoilsₜ) # nfft is calculated into kspₜ
                b[:, tt, :] .= kspₜ ./ scaling
            end
            #b .* basistransform
            b
        end,
        backw = (b, y) -> begin
            ksp = reshape(y, M, nt, nc)
            ksp = wi isa Nothing ? ksp : ksp_buffer .= ksp .* wi
            for tt=1:nt
                kspₜ .= @view ksp[:, tt, :]
                st[tt].nufft_adjoint(xcoilsₜ, kspₜ) # adjoint nfft is calculated into xcoilsₜ
                xcoils[:, :, tt, :] .= xcoilsₜ ./ scaling
            end
            #b .* basistransform
            xcoils .*= sense_maps_conj
            sum!(reshape(b, nx, ny, nt, 1), xcoils)
        end,
        inDims = (nx, ny, nt), outDims = (M, nt, nc))
end

function getT(nx::Int, ny::Int, nt::Int, dType::Type)
    buffer = Array{dType}(undef, nx*ny, nt)
    FFT_plan = plan_fft(buffer, 2)
    iFFT_plan = inv(FFT_plan)
    scaling = convert(dType, √(nt))
    T = FunctionOperator{dType}(name = "T",
        forw = (b, x) -> begin
            mul!(buffer, FFT_plan, x)
            fftshift!(b, buffer, 2)
            b ./= scaling
            end,
        backw = (b, y) -> begin
            ifftshift!(buffer, y, 2)
            mul!(b, iFFT_plan, buffer)
            b .*= scaling
            end,
        inDims = (nx*ny, nt), outDims = (nx*ny, nt))
end


