using LinearAlgebra, FFTW, FunctionOperators, OffsetArrays, IterativeSolvers, ToeplitzMatrices, Printf, ProgressMeter
include("helper_functions.jl")
rank_rtol = 1e-4

function update_H!(H, σ, ϵᵏ)
    for ind in CartesianIndices(H)
        i, j = ind[1], ind[2]
        H[ind] = 1 / (max(σ[i], ϵᵏ) * max(σ[j], ϵᵏ))
    end
end

function update_dH!(dH, σ, ϵᵏ, r̃)
    for j in eachindex(dH)
        dH[j] = 1 / (max(σ[r̃+1], ϵᵏ) * max(σ[j], ϵᵏ))
    end
end

    
split(γ, r̃, d₁, d₂) = @views begin
    γ₁ = reshape(γ[1:r̃^2], r̃, r̃)
    γ₂ = reshape(γ[r̃^2+1:r̃*(r̃+d₂)], r̃, d₂)
    γ₃ = reshape(γ[r̃*(r̃+d₂)+1:r̃*(r̃+d₁+d₂)], d₁, r̃)
    γ₁, γ₂, γ₃
end

function update_𝒟⁻¹!(𝒟⁻¹, H, dH, r̃, d₁, d₂)
    𝒟⁻¹₁, 𝒟⁻¹₂, 𝒟⁻¹₃ = split(𝒟⁻¹, r̃, d₁, d₂)
    𝒟⁻¹₁ .= H
    for i in 1:d₂
        𝒟⁻¹₂[:,i] .= dH
    end
    for i in 1:d₁
        𝒟⁻¹₃[i,:] .= dH
    end
    𝒟⁻¹ .= 1 ./ 𝒟⁻¹
end

function get_P_operator(Uᵏ, Vᵏ, Vtᵏ, tempᵈ¹ˣᵈ², r̃, d₁, d₂, dType)
    
    tempᵈ¹ˣʳ, tempʳˣᵈ² = Array{dType}(undef, d₁, r̃), Array{dType}(undef, r̃, d₂)
    
    I_VV, I_UU = Array{dType}(undef, d₂, d₂), Array{dType}(undef, d₁, d₁)
    Iᵈ¹ˣᵈ¹, Iᵈ²ˣᵈ² = Diagonal(ones(d₁)), Diagonal(ones(d₂))
    
    Pᵏ = FunctionOperator{dType}(name="Pᵏ", inDims = (r̃*(r̃+d₁+d₂),), outDims = (d₁, d₂),
        forw = (b,γ) -> begin
                γ₁, γ₂, γ₃ = split(γ, r̃, d₁, d₂)
                # According to (2.169), the equation would be:
                # Uᵏ * γ₁ * Vᵏ' + Uᵏ * γ₂' * (I - Vᵏ*Vᵏ') + (I - Uᵏ*Uᵏ') * γ₃' * Vᵏ'
                # But as the columns of γ₂ are orthogonal to the ones in Uᵏ,
                # the rows of γ₃ are orthogonal to the columns of Vᵏ,
                # the expression can be simplified:
                # (Uᵏ * γ₁ + γ₃) * Vᵏ' + Uᵏ * γ₂
                # And this is implemented avoiding array re-allocations:
                mul!(tempᵈ¹ˣʳ, Uᵏ, γ₁)
                tempᵈ¹ˣʳ .+= γ₃
                mul!(b, tempᵈ¹ˣʳ, Vtᵏ)
                mul!(tempᵈ¹ˣᵈ², Uᵏ, γ₂)
                b .+= tempᵈ¹ˣᵈ²
            end,
        backw = (γ,Φᵃy) -> begin
                γ₁, γ₂, γ₃ = split(γ, r̃, d₁, d₂)
                # Things to do:
                # γ₁ .= Uᵏ' * Φᵃy * Vᵏ
                # γ₂ .= Uᵏ' * Φᵃy * (I - Vᵏ*Vᵏ')
                # γ₃ .= (I - Uᵏ*Uᵏ') * Φᵃy * Vᵏ
                # Efficient implementation:
                I_VV .= Iᵈ²ˣᵈ² .- mul!(I_VV, Vᵏ, Vtᵏ) # same as I - Vᵏ*Vtᵏ
                I_UU .= Iᵈ¹ˣᵈ¹ .- mul!(I_UU, Uᵏ, Uᵏ') # same as I - Uᵏ*Uᵏ'
                mul!(tempᵈ¹ˣʳ, Φᵃy, Vᵏ)
                mul!(γ₁, Uᵏ', tempᵈ¹ˣʳ)
                mul!(γ₃, I_UU, tempᵈ¹ˣʳ)
                mul!(tempʳˣᵈ², Uᵏ', Φᵃy)
                mul!(γ₂, tempʳˣᵈ², I_VV)
                γ
                #vcat(vec(γ₁), vec(γ₂), vec(γ₃))
            end)
    
    Pᵏ
end

function get_CG_operator(PᵃΦᵃΦP, 𝒟_weighting, tempʳ⁽ʳ⁺ᵈ¹⁺ᵈ²⁾, r̃, d₁, d₂, dType)
    FunctionOperator{dType}(name = "CG_op", inDims = (r̃*(r̃+d₁+d₂),), outDims = (r̃*(r̃+d₁+d₂),),
        forw = (b,γ) ->  begin
            # An efficient implementation for:
            # b .= (ϵᵏ^2 * I / (𝒟⁻¹ - ϵᵏ^2 * I)) * γ + Pᵏ' * Φ' * Φ * Pᵏ * γ
            mul!(tempʳ⁽ʳ⁺ᵈ¹⁺ᵈ²⁾, PᵃΦᵃΦP, γ)
            mul!(b, Diagonal(𝒟_weighting), γ)
            b .+= tempʳ⁽ʳ⁺ᵈ¹⁺ᵈ²⁾
        end)
end

function HM_IRLS(
        Xᴳᵀ::AbstractArray,                     # ground truth for MSE evaluation
        y::AbstractArray,                       # under-sampled data
        Φ::FunctionOperator;                    # sampling operator
        img_size::NTuple = size(Xᴳᵀ),           # size of output matrix
        r̃::Int = 0,                             # rank estimate of solution
        maxIter::Union{Int, Nothing} = nothing, # number of CG iteration steps
        N::Int = 10,                            # number of iterations
        verbose::Bool = false)                  # print rank and loss value in each iteration

    # Initialize variables
    @assert 3 ≤ length(img_size) ≤ 4
    if length(img_size) == 3
        nx,ny,nt = img_size
        nc = size(y)[end]
    else
        nx,ny,nt,nc = img_size
    end
    dType = eltype(y)
    d₁, d₂ = nx*nx, nt
    y = reshape(y, :, nt, nc)
    Φ = reshape(Φ, inDims = (d₁, d₂), outDims = size(y))
    Xᴳᵀ = reshape(Xᴳᵀ, d₁, d₂)
    r̃ == 0 && (r̃ = rank(Xᴳᵀ))
    maxIter = maxIter isa Nothing ? r̃*(r̃+d₁+d₂) : maxIter
    ϵᵏ = Inf
    Xᵏ = Φ' * y
    
    # Preallocate arrays
    F = svd(Xᵏ)
    Uᵏ, σ, Vᵏ, Vtᵏ = F.U[:, 1:r̃], F.S, F.V[:, 1:r̃], F.Vt[1:r̃, :]
    Hᵏᵤᵥ = Array{dType}(undef, r̃, r̃)
    dHᵏ = Array{dType}(undef, r̃)
    𝒟⁻¹, 𝒟_weighting, b, γᵏ, tempʳ⁽ʳ⁺ᵈ¹⁺ᵈ²⁾ = [Vector{dType}(undef, r̃*(r̃+d₁+d₂)) for _ in 1:5]
    tempᵈ¹ˣᵈ² = Array{dType}(undef, d₁, d₂)
    rᵏ, γᵏ_tilde = similar(y), similar(γᵏ)
    statevars = IterativeSolvers.CGStateVariables(similar(γᵏ), similar(γᵏ), similar(γᵏ))
    
    # Create operators
    Pᵏ= get_P_operator(Uᵏ, Vᵏ, Vtᵏ, tempᵈ¹ˣᵈ², r̃, d₁, d₂, dType)
    PᵃΦᵃΦP = Pᵏ' * Φ' * Φ * Pᵏ
    ΦP, PᵃΦᵃ = Φ * Pᵏ, Pᵏ' * Φ'
    CG_op = get_CG_operator(PᵃΦᵃΦP, 𝒟_weighting, tempʳ⁽ʳ⁺ᵈ¹⁺ᵈ²⁾, r̃, d₁, d₂, dType)
    
    cost_vec = OffsetVector{real(dType)}(undef, 0:N)
    rank_vec = OffsetVector{Int}(undef, 0:N)
    time_vec = OffsetVector{Float64}(undef, 0:N)
    fill!(cost_vec, -1), fill!(rank_vec, -1), fill!(time_vec, -1)
    time_vec[0] = 0

    cost_vec[0] = norm(tempᵈ¹ˣᵈ² .= Xᴳᵀ .-  Xᵏ)
    rank_vec[0] = sum(Int, σ .> rank_rtol)
    verbose && iterationPrint("k" => 0, "rank(Xᵏ)" => rank_vec[0],
            "‖Xᴳᵀ - Xᵏ‖₂" => cost_vec[0], "σ₁" => σ[1], "ϵᵏ" => ϵᵏ)
    
    interval = verbose ? Inf : 1
    @showprogress interval "IRLS " for k in 1:N
        
        time_vec[k] = time_vec[k-1] + @elapsed begin
            
            svd!(tempᵈ¹ˣᵈ² .= Xᵏ, F)
            @views begin Uᵏ .= F.U[:, 1:r̃]; Vᵏ .=  F.V[:, 1:r̃]; Vtᵏ .= F.Vt[1:r̃, :]; end

            ϵᵏ = min(ϵᵏ, σ[r̃+1])

            update_H!(Hᵏᵤᵥ, σ, ϵᵏ)
            update_dH!(dHᵏ, σ, ϵᵏ, r̃)
            update_𝒟⁻¹!(𝒟⁻¹, Hᵏᵤᵥ, dHᵏ, r̃, d₁, d₂)

            # An efficient implementation of 𝒟_weighting = ϵᵏ^2 * I / (𝒟⁻¹ - ϵᵏ^2 * I):
            𝒟_weighting .= ϵᵏ^2 ./ (𝒟⁻¹ .- ϵᵏ^2)

            mul!(b, PᵃΦᵃ, y) # right hand side for CG
            mul!(γᵏ, Pᵏ', Xᵏ) # initial value for CG

            cg!(γᵏ, CG_op, b, maxiter = maxIter, statevars = statevars) # 2.167

            # An efficient implementation of rᵏ = y - Φ * Pᵏ * γᵏ:
            rᵏ .= y .- mul!(rᵏ, ΦP,  γᵏ)

            # An efficient implementation of γᵏ_tilde = Diagonal(𝒟⁻¹ ./ (𝒟⁻¹ .- ϵᵏ^2)) * γᵏ - Pᵏ' * Φ' * rᵏ
            𝒟_weighting .= 𝒟⁻¹ ./ (𝒟⁻¹ .- ϵᵏ^2) # same as Diagonal(𝒟⁻¹ ./ (𝒟⁻¹ .- ϵᵏ^2))
            mul!(tempʳ⁽ʳ⁺ᵈ¹⁺ᵈ²⁾, PᵃΦᵃ, rᵏ)
            mul!(γᵏ_tilde, Diagonal(𝒟_weighting), γᵏ)
            γᵏ_tilde .-= tempʳ⁽ʳ⁺ᵈ¹⁺ᵈ²⁾

            # An efficient implementation of Xᵏ = Φ' * rᵏ + Pᵏ * γᵏ_tilde
            mul!(Xᵏ, Pᵏ, γᵏ_tilde)
            Xᵏ .+= mul!(tempᵈ¹ˣᵈ², Φ', rᵏ)   # 2.168
            
        end

        cost_vec[k] = norm(tempᵈ¹ˣᵈ² .= Xᴳᵀ .-  Xᵏ)
        rank_vec[k] = sum(Int, σ .> rank_rtol)
        verbose && iterationPrint("k" => k, "rank(Xᵏ)" => rank_vec[k],
                "‖Xᴳᵀ - Xᵏ‖₂" => cost_vec[k], "σ₁" => σ[1], "ϵᵏ" => ϵᵏ)
        if cost_vec[k] > cost_vec[0]
            break
        end
        
    end
    
    reshape(Xᵏ, nx, ny ,nt), cost_vec, rank_vec, time_vec
end

function AL_CG(
        Xᴳᵀ::AbstractArray,                   # ground truth for MSE evaluation
        d::AbstractArray,                     # under-sampled k-t data
        E::FunctionOperator,                  # acquisition operator
        T::FunctionOperator;                  # temporal Fourier tranform
        img_shape::NTuple = size(d),          # size of output image
        scale_L::Real = 1,                    # scaling factor for L
        scale_S::Real = 1,                    # scaling factor for S
        λ_L::Real = 0.01,                     # singular value threshold
        λ_S::Real = 0.05,                     # sparsity threshold
        δ₁::Real = 1//10,                     # first AL penalty parameter
        δ₂::Real = 1//100,                    # second AL penalty parameter
        iterL::Int = 3,                       # number of CG iteration steps for S
        iterS::Int = iterL,                   # number of CG iteration steps for L
        N::Int = 10,                          # number of iterations
        verbose::Bool = false)                # print rank and loss value in each iteration
    
    complexType = eltype(d)
    floatType = real(complexType)
    
    scale_L, scale_S, λ_L, λ_S, δ₁, δ₂ = convert(floatType, scale_L), convert(floatType, scale_S),
        convert(floatType, λ_L), convert(floatType, λ_S), convert(floatType, δ₁), convert(floatType, δ₂)
    
    #Initialize
    @assert 3 ≤ length(img_shape) ≤ 4
    if length(img_shape) == 3
        nx,ny,nt = img_shape
        nc = size(d)[end]
    else
        nx,ny,nt,nc = img_shape
    end
    d = reshape(d, :, nt, nc)
    E = reshape(E, inDims = (nx*nx, nt), outDims = size(d))
    Xᴳᵀ = reshape(Xᴳᵀ, nx*nx, nt)
    x₀ = E' * d # initial guess
    
    L, S = copy(x₀), zeros(complexType, size(x₀))
    V₁, V₂ = zeros(complexType, size(L)), zeros(complexType, size(L))
    
    SVT! = getSVT()
    
    cg_tol = convert(floatType, 1e-5)
    P, Q = similar(L), similar(L)
    temp₁, temp₂ = similar(L), similar(L)
    
    cost_vec = OffsetVector{floatType}(undef, 0:N)
    rank_vec = OffsetVector{Int}(undef, 0:N)
    time_vec = OffsetVector{floatType}(undef, 0:N)
    time_vec[0] = 0
    cost_vec[0] = norm(temp₁ .= Xᴳᵀ .- L .- S)
    rank_vec[0] = sum(Int, svdvals!(temp₁ .= L) .> rank_rtol)
    verbose && iterationPrint("k" => 0, "rank(Xᵏ)" => rank_vec[0],
            "‖Xᴳᵀ - Xᵏ‖₂" => cost_vec[0])
    
    EᴴE_op, Tᴴ = E'*E, T'
    cg_op₁, cg_op₂ = EᴴE_op + δ₁*I, EᴴE_op + δ₂*I
    CGstate = CGStateVariables(temp₂, similar(L), similar(L))
    
    interval = verbose ? Inf : 1
    @showprogress interval "AL-CG " for k in 1:N
        
        time_vec[k] = time_vec[k-1] + @elapsed begin
        
            P = SVT!(P .= L .+ V₁, scale_L * λ_L / δ₁)
            Q = Λ!(mul!(Q, T, S) .+= V₂, scale_S * λ_S / δ₂)

            temp₁ .= x₀ .- mul!(temp₁, EᴴE_op, S) .+ δ₁.*(temp₂ .= P .- V₁)
            cg!(L, cg_op₁, temp₁, tol=cg_tol, maxiter=iterL, statevars=CGstate)

            mul!(temp₂, Tᴴ, temp₁ .= Q .- V₂)
            temp₁ .= x₀ .- mul!(temp₁, EᴴE_op, L) .+ δ₂.*temp₂
            cg!(S, cg_op₂, temp₁, tol=cg_tol, maxiter=iterS, statevars=CGstate)

            V₁ .+= L .- P
            V₂ .+= mul!(temp₁, T, S) .- Q

        end
        
        cost_vec[k] = norm(temp₁ .= Xᴳᵀ .- L .- S)
        rank_vec[k] = sum(Int, svdvals!(temp₁ .= L) .> rank_rtol)
        verbose && iterationPrint("k" => k, "rank(Xᵏ)" => rank_vec[k],
                "‖Xᴳᵀ - Xᵏ‖₂" => cost_vec[k])
        
    end
    
    reshape(L, nx, ny, nt) + reshape(S, nx, ny, nt), rank_vec, cost_vec, time_vec
end

function AL_2(
        Xᴳᵀ::AbstractArray,                   # ground truth for MSE evaluation
        d::AbstractArray,                     # under-sampled k-t data
        Ω::FunctionOperator,                  # under-sampling mask
        Q::FunctionOperator,                  # Fourier encoding operator
        C::FunctionOperator,                  # coil sensitivity maps
        T::FunctionOperator;                  # temporal Fourier tranform
        img_shape::NTuple = size(d),          # size of output image
        scale_L::Real = 1,                    # scaling factor for L
        scale_S::Real = 1,                    # scaling factor for S
        λ_L::Real = 0.01,                     # singular value threshold
        λ_S::Real = 0.05,                     # sparsity threshold
        δ₁::Real = 1//10,                     # first AL penalty parameter
        δ₂::Real = 1//100,                    # second AL penalty parameter
        N::Int = 10,                          # number of iterations
        verbose::Bool = false)                # print rank and loss value in each iteration
    
    complexType = eltype(d)
    floatType = real(complexType)
    
    scale_L, scale_S, λ_L, λ_S, δ₁, δ₂ = convert(floatType, scale_L), convert(floatType, scale_S),
        convert(floatType, λ_L), convert(floatType, λ_S), convert(floatType, δ₁), convert(floatType, δ₂)
    
    #Initialize
    @assert 3 ≤ length(img_shape) ≤ 4
    if length(img_shape) == 3
        nx,ny,nt = img_shape
        nc = size(d)[end]
    else
        nx,ny,nt,nc = img_shape
    end
    E = reshape(Ω * Q * C, inDims = (nx*nx, nt), outDims = size(d))
    QC = reshape(Q * C, inDims = (nx*nx, nt), outDims = size(d))
    Xᴳᵀ = reshape(Xᴳᵀ, nx*nx, nt)
    Tᴴ, QCᴴ = T', QC'
    
    L = reshape(E' * d, nx*ny, nt) # initial guess
    S = zeros(complexType, size(L))
    X = L + S
    V₁ = zeros(complexType, size(d))
    V₂ = zeros(complexType, size(L))
    Z_scaler = repeat(1 ./ (samp .+ δ₁), 1, 1, 1, nc) # equivalent to (Ω'*Ω + δ₁*I)'
    temp₁, temp₂, Z, temp₃, temp₄ = similar(L), similar(L), similar(d), similar(d), similar(d)
    
    SVT! = getSVT()
    
    cost_vec = OffsetVector{floatType}(undef, 0:N)
    rank_vec = OffsetVector{Int}(undef, 0:N)
    time_vec = OffsetVector{floatType}(undef, 0:N)
    time_vec[0] = 0
    cost_vec[0] = norm(temp₁ .= Xᴳᵀ .- L .- S)
    rank_vec[0] = sum(Int, svdvals!(temp₁ .= L) .> rank_rtol)
    verbose && iterationPrint("k" => 0, "rank(Xᵏ)" => rank_vec[0],
            "‖Xᴳᵀ - Xᵏ‖₂" => cost_vec[0])
    
    # Iteration
    interval = verbose ? Inf : 1
    @showprogress interval "AL-2 " for k in 1:N
        
        time_vec[k] = time_vec[k-1] + @elapsed begin
        
            temp₃ .= d #mul!(temp₃, Ω', d)
            mul!(temp₄, QC, X)
            Z .= Z_scaler .* (temp₃ .+ δ₁ .* (temp₄ .- V₁))

            mul!(temp₁, QCᴴ, temp₃ .= Z .+ V₁)
            temp₂ .= L .+ S .- V₂
            X .= δ₁./(δ₁ .+ δ₂).*temp₁ .+ δ₂./(δ₁ .+ δ₂).*temp₂

            L = SVT!(L .= X .- S .+ V₂, scale_L * λ_L / δ₂)

            mul!(temp₁, T, temp₁ .= X .- L .+ V₂)
            mul!(S, Tᴴ, Λ!(temp₁, scale_S * λ_S / δ₂))

            mul!(temp₃, QC, X)
            V₁ .+= Z .- temp₃

            V₂ .+= X .- L .- S
            
        end
        
        cost_vec[k] = norm(temp₁ .= Xᴳᵀ .- L .- S)
        rank_vec[k] = sum(Int, svdvals!(temp₁ .= L) .> rank_rtol)
        verbose && iterationPrint("k" => k, "rank(Xᵏ)" => rank_vec[k],
                "‖Xᴳᵀ - Xᵏ‖₂" => cost_vec[k])
    end
    
    reshape(L, nx, ny, nt) + reshape(S, nx, ny, nt), rank_vec, cost_vec, time_vec
end

function createXLS(dType, row_dim, nt)
    X = zeros(dType, row_dim, 2*nt)
    L = view(X, :, 1:nt)
    S = view(X, :, nt + 1:2*nt)
    return X,L,S
end
function allocate(dType, row_dim, nt)
    createXLS(dType, row_dim, nt)..., createXLS(dType, row_dim, nt)...
end

function PGM(
        Xᴳᵀ::AbstractArray,                 # ground truth for MSE evaluation
        d::AbstractArray,                   # under-sampled k-t data
        E::FunctionOperator,                # acquisition operator
        T::FunctionOperator;                # sparsifying operator
        img_shape::NTuple = size(d),        # size of output image
        scale_L::Real = 1,                  # scaling factor for L
        scale_S::Real = 1,                  # scaling factor for S
        tscale::Real = 1,                   # scaling factor for t
        λ_L::Real = 0.01,                   # singular value threshold
        λ_S::Real = 0.05,                   # sparsity threshold
        N::Int = 10,                        # number of iterations
        restart::Bool = true,               # reset θₖ if cost increased
        momentum::Symbol = :pogm,           # update rule (:pogm, :ista, :fista)
        verbose::Bool = false)              # print rank and loss value in each iteration
    
    complexType = eltype(d)
    floatType = real(complexType)
    
    scale_L, scale_S, tscale, λ_L, λ_S = convert(floatType, scale_L), convert(floatType, scale_S),
        convert(floatType, tscale), convert(floatType, λ_L), convert(floatType, λ_S)
    
    #Initialize
    @assert 3 ≤ length(img_shape) ≤ 4
    if length(img_shape) == 3
        nx,ny,nt = img_shape
        nc = size(d)[end]
    else
        nx,ny,nt,nc = img_shape
    end
    row_dim = nx*ny
    d = reshape(d, :, nt, nc)
    E = reshape(E, inDims = (row_dim, nt), outDims = size(d))
    Xᴳᵀ = reshape(Xᴳᵀ, nx*nx, nt)
    x₀ = E' * d # initial guess
    
    Xₖ₋₁,Lₖ₋₁,Sₖ₋₁,Xₖ,Lₖ,Sₖ = allocate(complexType, row_dim, nt)
    X̃ₖ₋₁,L̃ₖ₋₁,S̃ₖ₋₁,X̃ₖ,L̃ₖ,S̃ₖ = allocate(complexType, row_dim, nt)
    if momentum == :pogm
        X̄ₖ₋₁,L̄ₖ₋₁,S̄ₖ₋₁,X̄ₖ,L̄ₖ,S̄ₖ = allocate(complexType, row_dim, nt)
    else
        X̄ₖ₋₁,L̄ₖ₋₁,S̄ₖ₋₁,X̄ₖ,L̄ₖ,S̄ₖ = X̃ₖ₋₁,L̃ₖ₋₁,S̃ₖ₋₁,X̃ₖ,L̃ₖ,S̃ₖ
    end
    Lₖ₋₁ .= x₀
    X̄ₖ₋₁ .= X̃ₖ₋₁ .= Xₖ₋₁
    temp₁, temp₂ = similar(Lₖ₋₁), similar(d)
    Eᴴ, Tᴴ = E', T'
    
    t = (in(momentum, (:fista, :pogm)) ? 5//10 : 99//100) * tscale
    
    mul!(temp₂, E, x₀)
    mul!(temp₁, E', temp₂ .-= d)
    Mₖ = Mₖ₋₁ = x₀ .- t .* temp₁ # we don't need two arrays for M, but it looks better this way
    
    θₖ₋₁ = ζₖ₋₁ = 1.
    
    SVT! = getSVT()
    
    cost_vec = OffsetVector{floatType}(undef, 0:N)
    rank_vec = OffsetVector{Int}(undef, 0:N)
    time_vec = OffsetVector{floatType}(undef, 0:N)
    time_vec[0] = 0
    cost_vec[0] = norm(temp₁ .= Xᴳᵀ .- Lₖ₋₁ .- Sₖ₋₁)
    rank_vec[0] = sum(Int, svdvals!(temp₁ .= Lₖ₋₁) .> rank_rtol)
    verbose && iterationPrint("k" => 0, "rank(Xᵏ)" => rank_vec[0],
            "‖Xᴳᵀ - Xᵏ‖₂" => cost_vec[0])
    
    # Iteration
    title = momentum == :ista ? "ISTA " : (momentum == :fista ? "FISTA " : "POGM ")
    interval = verbose ? Inf : 1
    @showprogress interval title for k in 1:N
        
        time_vec[k] = time_vec[k-1] + @elapsed begin
            
            @. L̃ₖ = Mₖ₋₁ - Sₖ₋₁
            @. S̃ₖ = Mₖ₋₁ - Lₖ₋₁

            θₖ  = (1 + √(1 + (k < N ? 4 : 8)*θₖ₋₁^2))/2

            if momentum == :pogm
                @. X̄ₖ = X̃ₖ + (θₖ₋₁-1)/θₖ*(X̃ₖ - X̃ₖ₋₁) +
                    (θₖ₋₁)/θₖ*(X̃ₖ - Xₖ₋₁) + (θₖ₋₁-1)/(ζₖ₋₁*θₖ)*t*(X̄ₖ₋₁ - Xₖ₋₁)
            elseif momentum == :fista
                @. X̄ₖ = X̃ₖ + (θₖ₋₁-1)/θₖ*(X̃ₖ - X̃ₖ₋₁)
            else
                # nothing to do as X̄ₖ == X̃ₖ
            end

            ζₖ  = t*(1 + (θₖ₋₁-1)/θₖ + (θₖ₋₁)/θₖ)

            SVT!(Lₖ .= L̄ₖ, scale_L*λ_L)

            mul!(temp₁, T, temp₁ .= S̄ₖ) # T operator in doesn't like subarrays
            mul!(temp₁, Tᴴ, Λ!(temp₁, scale_S*λ_S))
            Sₖ .= temp₁

            mul!(temp₂, E, temp₁ .= Lₖ .+ Sₖ)
            mul!(temp₁, Eᴴ, temp₂ .-= d)
            @. Mₖ = Lₖ + Sₖ - t * temp₁
        
        end
        
        cost_vec[k] = norm(temp₁ .= Xᴳᵀ .- Lₖ .- Sₖ)
        rank_vec[k] = sum(Int, svdvals!(temp₁ .= Lₖ) .> rank_rtol)
        verbose && iterationPrint("k" => k, "rank(Xᵏ)" => rank_vec[k],
                "‖Xᴳᵀ - Xᵏ‖₂" => cost_vec[k])
        
        # Move (k) -> (k-1), and avoid allocation for new (k)
        #      => switch (k) and (k-1) matrices
        @swap(Xₖ₋₁, Xₖ); @swap(X̃ₖ₋₁, X̃ₖ); @swap(X̄ₖ₋₁, X̄ₖ)
        @swap(Lₖ₋₁, Lₖ); @swap(L̃ₖ₋₁, L̃ₖ); @swap(L̄ₖ₋₁, L̄ₖ)
        @swap(Sₖ₋₁, Sₖ); @swap(S̃ₖ₋₁, S̃ₖ); @swap(S̄ₖ₋₁, S̄ₖ)
        @swap(Mₖ₋₁, Mₖ)
        θₖ₋₁, ζₖ₋₁ = θₖ, ζₖ
    end
    
    reshape(Lₖ, nx, ny, nt) + reshape(Sₖ, nx, ny, nt), rank_vec, cost_vec, time_vec
end

pos(x) = x < 0 ? zero(x) : x
Λ! = (v,p) -> @. v = sign(v) * pos(abs(v) - p)

function getSVT()
    F = nothing
    (A,p) -> begin
        F isa Nothing ? (F = svd!(A)) : svd!(A, F)
        mul!(A, F.U, mul!(F.Vt, Diagonal(Λ!(F.S, p)), F.Vt))
    end
end

normₙ!(A) = sum(svdvals!(A))

import Base.size
function Base.size(FO::FunctionOperator, d::Int)
    @assert d in [1, 2]
    prod(d == 1 ? FO.outDims : FO.inDims)
end

format(value) = value isa Real && !(value isa Int) ? @sprintf("%.3f", value) : string(value)

iterationPrint(pairs...) =
    println(join([key*" = "*format(value) for (key,value) in pairs], ",\t"))


