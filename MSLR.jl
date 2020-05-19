

using FFTW, LinearAlgebra, EllipsisNotation, FunctionOperators, Base.Cartesian, ProgressMeter, Random,
    Distributed

@generated function blocks_to_array!(
        output::AbstractArray{T, D},
        input::AbstractArray{T, D2},
        block_shape::NTuple{D, Int},
        shifts::NTuple{D, Int}) where {T, D, D2}
    @assert 2*D == D2 "Dimension mismatch! Proper dimensionality: 2*ndims(output) = ndims(input)"
    quote
        @nloops($D, block_idx, input, begin
                @nloops($D, point_idx, d -> axes(input, $D + d),
                    d -> begin
                        pos_in_output_d = (block_idx_d - 1) * shifts[d] + point_idx_d
                        pos_in_output_d > size(output, d) && break
                    end,
                    @inbounds @nref($D, output, pos_in_output) +=
                        input[@ntuple($D, block_idx)..., @ntuple($D, point_idx)...])
            end)
        return output
    end
end

@generated function array_to_blocks!(
        output::AbstractArray{T, D2},
        input::AbstractArray{T, D},
        block_shape::NTuple{D, Int},
        shifts::NTuple{D, Int}) where {T, D, D2}
    @assert 2*D == D2 "Dimension mismatch! Proper dimensionality: ndims(output) = 2*ndims(input)"
    quote
        @nloops($D, block_idx, output, begin
                @nloops($D, point_idx, d -> axes(output, $D + d),
                    d -> pos_in_output_d = (block_idx_d - 1) * shifts[d] + point_idx_d,
                    @inbounds output[@ntuple($D, block_idx)..., @ntuple($D, point_idx)...] = 
                        @nany($D, d -> pos_in_output_d > size(input, d)) ?
                            zero($T) :
                            @nref($D, input, pos_in_output))
            end)
        return output
    end
end

function hanning(shape)
    shape = shape isa Number ? (shape,) : shape
    hanningFunction(x, width) = 0.5 - 0.5 * cos(2π * x / max(1, (width - (width % 2))))
    multiDimHanning(coord) = prod([hanningFunction(coord[i]-1, shape[i]) for i in 1:length(shape)])
    [multiDimHanning(coord) for coord in CartesianIndices(tuple(UnitRange.(1, shape)...))]
end

# Some fancy stuff for naming
#   e.g. the array-to-block operator for the scale with block-size 8 as "𝓑₈"
digitToSubscript(d) = string(Char(Int('₀')+d))
numberToSubscript(n) = join([digitToSubscript(Int(d)-48) for d in string(n)])

function _get_ℳ(block_width, img_shape, dtype)
    
    block_shape = min.(img_shape, block_width)
    shifts = (block_shape .+ 1) .÷ 2
    num_blocks = ceil.(Int, img_shape ./ shifts .- 1)
    ndim = length(num_blocks)
    
    # Blocking operator:
    #  - forward embeds blocks in the output image in a sliding window manner
    #  - backward rearranges the image to blocks
    blockedShape = (num_blocks..., block_shape...)
    𝓑 = FunctionOperator{dtype}(name = "𝓑"*numberToSubscript(block_width),
        forw = (b, x) -> begin
            b .= zero(dtype)
            blocks_to_array!(b, reshape(x, blockedShape), block_shape, shifts)
        end,
        backw = (b, x) -> array_to_blocks!(b, x, block_shape, shifts),
        inDims = blockedShape,
        outDims = img_shape)
    
    # Multiplication with Hanning window to cancel the effect of sliding window.
    # Blocks are multiplied with the squre root of the window before and after the blocking operator.
    HanningWindow = reshape(sqrt.(hanning(block_shape)), fill(1, ndim)..., block_shape...)
    𝓦 = FunctionOperator{dtype}(name = "𝓦"*numberToSubscript(block_width),
        forw = (b, x) -> reshape(b, size(x)) .= x .* HanningWindow,
        backw = (b, x) -> reshape(b, size(x)) .= x .* HanningWindow,
        inDims = (num_blocks..., block_shape...),
        outDims = (num_blocks..., block_shape...))
    
    return 𝓑 * 𝓦
end
    
function _get_𝒩(t, samp, smap, frame_coils, img_shape, nc, dType)
    nx, ny = img_shape
    scaling = convert(dType, √(nx*ny))
    mask_with_scaling = reshape(samp[t], nx, ny, 1) .* scaling
    reshaped_smap = convert.(dType, smap)
    smap_conj_with_scaling = conj.(reshaped_smap) ./ scaling
    FFT_plan = plan_fft(frame_coils[Threads.threadid()], (1,2))
    iFFT_plan = inv(FFT_plan)
    E = FunctionOperator{dType}(name = "E", 
        forw = (b, x) -> begin # Don't ask me, why did Mr Otazo use ifft instead of fft...
                xcoils₁ = frame_coils[Threads.threadid()]
                xcoils₂ = b
                xcoils₁ .= reshape(x, (nx, ny, 1)) .* reshaped_smap
                ifftshift!(xcoils₂, xcoils₁, (1, 2))
                mul!(xcoils₁, iFFT_plan, xcoils₂)
                fftshift!(b, xcoils₁, (1, 2))
                b .*= mask_with_scaling
            end,
        backw = (b, y) -> begin # But he used it consistently, so it doesn't make a big difference
                xcoils₁ = frame_coils[Threads.threadid()]
                xcoils₂ = similar(xcoils₁)
                ifftshift!(xcoils₂, y, (1, 2))
                mul!(xcoils₁, FFT_plan, xcoils₂)
                fftshift!(xcoils₂, xcoils₁, (1, 2))
                xcoils₂ .*= smap_conj_with_scaling
                sum!(reshape(b, (nx, ny, 1)), xcoils₂)
                b
            end,
        inDims = (nx, ny), outDims = (nx, ny, nc))
end

function _get_𝒟²(t, dcfₜ, C, num_ro, tr_per_frame, dtype)
    FunctionOperator{dtype}(name = "𝒟²"*numberToSubscript(t),
        forw = (outputₜ, inputₜ) -> error("Not implemented"),
        backw = (outputₜ, inputₜ) -> begin
                outputₜ .= inputₜ .* dcfₜ[t]
            end,
        inDims = (C, num_ro, tr_per_frame),
        outDims = (C, num_ro, tr_per_frame))
end

function _get_𝒟(t, dcfₜ, C, num_ro, tr_per_frame, dtype)
    FunctionOperator{dtype}(name = "𝒟"*numberToSubscript(t),
        forw = (outputₜ, inputₜ) -> begin
                outputₜ .= inputₜ .* sqrt.(dcfₜ[t])
            end,
        backw = (outputₜ, inputₜ) -> begin
                outputₜ .= inputₜ .* sqrt.(dcfₜ[t])
            end,
        inDims = (C, num_ro, tr_per_frame),
        outDims = (C, num_ro, tr_per_frame))
end

function _get_λ(block_width, λ, img_shape, T, dtype)
    
    block_shape = min.(img_shape, block_width)
    shifts = (block_shape .+ 1) .÷ 2
    num_blocks = ceil.(Int, img_shape ./ shifts .- 1)
    
    N = prod(block_shape)
    B = prod(num_blocks)
    
    return λ * (√N + √T + √(2log(B)))
end

function MaxEig!(op, x, iter_num)
    λₘₐₓ = Inf
    #@showprogress 0.2 "Normalization of dcf... "
    for i in 1:iter_num
        mul!(x, op, x)
        λₘₐₓ = norm(x, 2)
        x ./= λₘₐₓ
    end
    λₘₐₓ
end

function _normalize(coord, dcf, ksp, mps, img_shape, C, T, tr_per_frame, max_power_iter, dtype)
    
    # Estimate maximum eigenvalue.
    print("Normalization of dcf:\t\t\t")
    @time begin
        coordₜ₁ = @view coord[:, :, 1:tr_per_frame]
        dcfₜ₁ = reshape(@view(dcf[:, 1:tr_per_frame]), 1, :, tr_per_frame)
        plan = SigJl.nufft_plan(coordₜ₁, img_shape, threaded = true)
        adj_plan = plan'
        𝒩ₜ₁ = FunctionOperator{dtype}(name = "𝒩ₜ₁",
            forw = (b, x) -> mul!(b, plan, x),
            backw = (b, x) -> mul!(b, adj_plan, x),
            inDims = img_shape,
            outDims = size(coordₜ₁))
        𝒟²ₜ₁ = FunctionOperator{dtype}(name = "𝒟ₜ₁",
            forw = (b, x) -> b .= x .* dcfₜ₁,
            inDims = size(coordₜ₁),
            outDims = size(coordₜ₁))
        x₀ = convert.(eltype(ksp), rand(img_shape...))
        λₘₐₓ = MaxEig!(𝒩ₜ₁' * 𝒟²ₜ₁ * 𝒩ₜ₁, x₀, max_power_iter)
        dcf ./= abs(λₘₐₓ)
    end
    
    # Estimate scaling
    print("Normalization of ksp:\t\t\t")
    @time begin
        dcf = reshape(dcf, 1, size(dcf)...)
        img = SigJl.nufft_adjoint(coord, ksp .* dcf, oshape=(C, img_shape...), threaded=true)
                #progressTitle="Normalization of ksp... ", progress_dt=0.2)
        img .*= conj.(mps)
        ksp ./= norm(sum(img, dims=1), 2)
    end
    
    dcf, ksp
end

function normalize_L!(L)
    D = ndims(L) ÷ 2
    L_norm = mapslices(norm, L, dims=D+1:2D)
    L ./= L_norm
    L_norm
end

function normalize_R!(Rₜ)
    R = parent(Rₜ[1])
    R ./= mapslices(norm, R, dims=ndims(R))
end

function _init_L(blocked_shape, dtype)
    L = randn(dtype, blocked_shape)
    normalize_L!(L)
    L
end

function _init_R(blocked_shape, D, T, dtype)
    Array{dtype}(undef, (blocked_shape[1:D]..., fill(1, D)..., T))
end

function powerIter_R!(Rⱼₜ, Aᴴy, AᴴyLᴴⱼ, t, 𝒜, ℳⱼ, kspₜ, Lⱼ)
    mul!(Aᴴy, 𝒜', kspₜ[t])
    for j in 1:length(Rⱼₜ)
        mul!(AᴴyLᴴⱼ[j], ℳⱼ[j]', Aᴴy)
        AᴴyLᴴⱼ[j] .*= conj.(Lⱼ[j])
        sum!(Rⱼₜ[j][t], AᴴyLᴴⱼ[j])
    end
end

function powerIter_L!(Lⱼ, Aᴴy, t, 𝒜, ℳⱼ, kspₜ, Rⱼₜ)
    mul!(Aᴴy, 𝒜', kspₜ[t])
    for j in 1:length(Lⱼ)
        Lⱼ[j] .+= (ℳⱼ[j]' * Aᴴy) .* conj.(Rⱼₜ[j][t])
    end
end

function reconstructImage(Lⱼ, Rⱼₜ, ℳⱼ, nx, ny, T, J, dtype)
    img = zeros(dtype, nx, ny, T)
    temp = Array{dtype}(undef, nx, ny)
    for t in 1:T
        for j in 1:J
            img[.., t] .+= mul!(temp, ℳⱼ[j], Lⱼ[j] .* Rⱼₜ[j][t])
        end
    end
    img
end

function _power_method!(Lⱼ, Rⱼₜ, 𝒩ₜ, ℳⱼ, kspₜ, img_shape, D, max_power_iter, dtype)
    
    σⱼ = nothing
    J = length(Lⱼ)
    T = length(Rⱼₜ[1])
    
    # We need these only to avoid re-allocation of same buffers over and over again
    Lⱼ_buffers = [deepcopy(Lⱼ) for _ in 1:Threads.nthreads()]
    Aᴴy_buffers = [similar(kspₜ[1], img_shape) for _ in 1:Threads.nthreads()]
    
    for it in 1:max_power_iter
        # Rⱼₜ = calc_norm_of_blocks(ℳⱼᴴ(Aₜᴴ⋅yₜᴴ)⋅Lⱼᴴ)
        #p = Progress(T+1, 0.2, "Power iteration (R, $it/$max_power_iter)... ")
        #@time begin
            Threads.@threads for t in 1:T
                Aᴴy_buffer = Aᴴy_buffers[Threads.threadid()]
                AᴴyLⱼᴴ_buffer = Lⱼ_buffers[Threads.threadid()]
                powerIter_R!(Rⱼₜ, Aᴴy_buffer, AᴴyLⱼᴴ_buffer, t, 𝒩ₜ(t), ℳⱼ, kspₜ, Lⱼ)
                #next!(p)
            end
            pmap(normalize_R!, Rⱼₜ)
            #finish!(p)
        #end
        
        # Lⱼ = Σₜ(ℳⱼᴴ(Aₜᴴ⋅yₜᴴ)⋅Rⱼₜᴴ)
        #p = Progress(T+3, 0.2, "Power iteration (L, $it/$max_power_iter)... ")
        #@time begin
            Threads.@threads for i in 1:Threads.nthreads()
                fill!.(Lⱼ_buffers[i], zero(dtype))
            end
            #next!(p)
            Threads.@threads for t in 1:T
                Aᴴy_buffer = Aᴴy_buffers[Threads.threadid()]
                Lⱼ_buffer = Lⱼ_buffers[Threads.threadid()]
                powerIter_L!(Lⱼ_buffer, Aᴴy_buffer, t, 𝒩ₜ(t), ℳⱼ, kspₜ, Rⱼₜ)
                #next!(p)
            end
            for i in 1:Threads.nthreads(), j in 1:J
                i == 1 ? Lⱼ[j] .= Lⱼ_buffers[i][j] : Lⱼ[j] .+= Lⱼ_buffers[i][j]
            end
            #next!(p)
            
            σⱼ = pmap(normalize_L!, Lⱼ) # save the norm of L at each level
            #finish!(p)
        #end
        
    end

    for j in 1:J
        Lⱼ[j] .*= sqrt.(σⱼ[j])
        parent(Rⱼₜ[j][1]) .*= sqrt.(σⱼ[j])
    end
    
    σⱼ
end

function _sgd(Xᴳᵀ, 𝒩ₜ, ℳⱼ, kspₜ, Lⱼ, Rⱼₜ, T, α, β, σⱼ, λⱼ, D, max_epoch, decay_epoch, img_shape, dtype)
    J = length(Rⱼₜ)
    
    cost_vec = OffsetVector{real(dtype)}(undef, 0:max_epoch)
    time_vec = OffsetVector{Float64}(undef, 0:max_epoch)
    time_vec[0] = 0
    img = reconstructImage(Lⱼ, Rⱼₜ, ℳⱼ, nx, ny, T, J, dtype)
    cost_vec[0] = norm(Xᴳᵀ - img)
    
    N = Threads.nthreads()
    imgₙ = [similar(Rⱼₜ[1], dtype, img_shape) for n in 1:N]
    ∇Lₙⱼ = [[similar(Lⱼ[j]) for j in 1:J] for n in 1:N]
    ∇Rₙⱼ = [[similar(Rⱼₜ[j][1]) for j in 1:J] for n in 1:N]
    lossₙ = fill(zero(eltype(λⱼ[1])), N)
    
    interval = 1
    @showprogress interval "MSLR " for epoch in 1:max_epoch
        time_vec[epoch] = time_vec[epoch-1] + @elapsed begin
            loss = 0
            #p = Progress(T, 0.2, "Epoch {epoch}/{max_epoch}... ")
            for tₙ in Iterators.partition(randperm(T), N)
                Threads.@threads for n in 1:length(tₙ)
                    ∇Lⱼ, ∇Rⱼ, t, img = ∇Lₙⱼ[n], ∇Rₙⱼ[n], tₙ[n], imgₙ[n]
                    lossₙ[n] = _step!(∇Lⱼ, ∇Rⱼ, t, 𝒩ₜ, ℳⱼ, img, kspₜ, img_shape,
                                        Lⱼ, Rⱼₜ, σⱼ, λⱼ, D, dtype)
                    #next!(p)
                end
                loss += sum(lossₙ[1:length(tₙ)])
                ∇Lⱼ = ∇Lₙⱼ[1]; [∇Lⱼ .+= ∇Lₙⱼ[n] for n in 2:length(tₙ)]
                for j in 1:J
                    Lⱼ[j] .-= α .* β^(epoch ÷ decay_epoch) .* ∇Lⱼ[j]
                    [Rⱼₜ[j][tₙ[n]] .-= α .* ∇Rₙⱼ[n][j] for n in 1:length(tₙ)]
                end
            end
        end
        
        img = reconstructImage(Lⱼ, Rⱼₜ, ℳⱼ, nx, ny, T, J, dtype)
        cost_vec[epoch] = norm(Xᴳᵀ - img)
        #iterationPrint("k" => epoch, "‖Xᴳᵀ - Xᵏ‖₂" => cost_vec[epoch])
        
    end
    
    cost_vec, time_vec
end

function _step!(∇Lⱼ, ∇Rⱼ, t, 𝒩ₜ, ℳⱼ, img, kspₜ, img_shape, Lⱼ, Rⱼₜ, σⱼ, λⱼ, D, dtype)
    J = length(Rⱼₜ)
    T = length(Rⱼₜ[1])
    # Form image.
    fill!(img, zero(dtype))
    temp = similar(img)
    [img .+= mul!(temp, ℳⱼ[j], Lⱼ[j] .* Rⱼₜ[j][t]) for j in 1:J]

    # Data consistency.
    𝒩 = 𝒩ₜ(t)
    ksp_hat = 𝒩 * img
    diff_hat = ksp_hat .-= kspₜ[t]
    loss = 0.5 * norm(diff_hat)^2
    diff = mul!(img, 𝒩', diff_hat)

    # Compute gradient.
    for j in 1:J

        # Loss.
        loss += λⱼ[j] / 2 * (norm(Lⱼ[j])^2 / T + norm(Rⱼₜ[j][t])^2)
        isnan(loss) && error("loss is NaN")
        (isinf(loss)) && throw(OverflowError("loss overflow"))

        # L gradient.
        diff_blocks = ℳⱼ[j]' * diff
        @. ∇Lⱼ[j] = T * (diff_blocks * conj(Rⱼₜ[j][t]) + λⱼ[j] / T * Lⱼ[j])

        # R gradient.
        sum!(∇Rⱼ[j], diff_blocks .*= conj.(Lⱼ[j]))
        ∇Rⱼ[j] .+= λⱼ[j] .* Rⱼₜ[j][t]

        # Precondition.
        ∇Lⱼ[j] ./= J .* σⱼ[j] .+ λⱼ[j]
        ∇Rⱼ[j] ./= J .* σⱼ[j] .+ λⱼ[j]

    end

    return loss
end

function MultiScaleLowRank(
        Xᴳᵀ::AbstractArray,                     # ground truth for MSE evaluation
        ksp::AbstractArray,                     # under-sampled data
        coord::AbstractArray,                   # sampling mask
        mps::AbstractArray;                     # sensitivity maps
        img_size::NTuple = size(Xᴳᵀ),           # size of output matrix
        block_widths::NTuple = (32, 64, 128),# block sizes
        α::Real = 1,                            # step size
        β::Real = 0.5,                          # decay for step size
        λ::Real = 1e-8,                         # scaling factor for Lagrangian multiplier
        max_epoch::Int = 10,                    # number of iterations over all frames
        decay_epoch::Int = max_epoch ÷ 3,       # number of iterations over all frames
        max_power_iter::Int = 3,                # maximal number of power iterations at initialization
        verbose::Bool = false)                  # print rank and loss value in each iteration
    
    # Initialize variables
    @assert 3 ≤ length(img_size) ≤ 4
    if length(img_size) == 3
        nx,ny,nt = img_size
        nc = size(ksp)[end]
    else
        nx,ny,nt,nc = img_size
    end
    
    ## --- Initialization --- ##
    println(" ---- Initialization ----")
    dtype = eltype(ksp)
    img_shape = nx, ny
    D = length(img_shape)
    J = length(block_widths)
    T = nt
    
    Random.seed!(0)
    
    print("initialization:   ")
    @time begin
        # create views to access time frames more easily
        coordₜ = [@view coord[:,:,t] for t in 1:T]
        kspₜ = [@view ksp[:,:,t,:] for t in 1:T]
        # Regularization parameters for each levels
        λⱼ = [_get_λ(bw, λ, img_shape, T, dtype) for bw in block_widths]
        # buffers for each thread to store intermediate results of shape (C, img_shape...)
        frame_coils = [similar(mps) for _ in 1:Threads.nthreads()]
        # Blocking operators (image array ⟷ blocks)
        ℳⱼ = [_get_ℳ(bw, img_shape, dtype) for bw in block_widths]
        # NUFFT operators (lazily created to save memory)
        𝒩ₜ = t -> _get_𝒩(t, coordₜ, mps, frame_coils, img_shape, nc, dtype)
        # Initialize L and R vectors
        Lⱼ_init = [_init_L(ℳⱼ[j].inDims, dtype) for j in 1:J]
        Rⱼ_init = [_init_R(ℳⱼ[j].inDims, D, T, dtype) for j in 1:J]
        Rⱼₜ_init = [[@view(Rⱼ_init[j][.., t]) for t in 1:T] for j in 1:J]
    end
    
    print("power iterations: ")
    @time σⱼ = _power_method!(Lⱼ_init, Rⱼₜ_init, 𝒩ₜ, ℳⱼ, kspₜ, img_shape, D, max_power_iter, dtype)
    
    println(" ---- Reconstruction ----")
    done = false
    cost_vec, time_vec = nothing, nothing
    while !done
        Lⱼ, Rⱼₜ = deepcopy(Lⱼ_init), deepcopy(Rⱼₜ_init)
        try
            cost_vec, time_vec = _sgd(Xᴳᵀ, 𝒩ₜ, ℳⱼ, kspₜ, Lⱼ, Rⱼₜ, T, α, β, σⱼ, λⱼ, D,
                max_epoch, decay_epoch, img_shape, dtype)
            done = true
        catch err
            if err isa OverflowError && err.msg == "loss overflow"
                α *= β
                println("Reconstruction diverged. New step size: $α")
                if α < 1e-4
                    error("too small α value, reconstruction seems not to converge...")
                end
            else
                rethrow(err)
            end
        end
    end
    
    img = zeros(dtype, nx, ny, nt)
    temp = Array{dtype}(undef, nx, ny)
    for t in 1:T
        for j in 1:J
            img[.., t] .+= mul!(temp, ℳⱼ[j], Lⱼ_init[j] .* Rⱼₜ_init[j][t])
        end
    end
    
    return img, cost_vec, time_vec
end
