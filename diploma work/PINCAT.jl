@assert length(ARGS) == 1

using ConfParser

println("Reading conf file: $(ARGS[1]).conf")
conf = ConfParse("$(ARGS[1]).conf")
parse_conf!(conf)

line = parse(Int, retrieve(conf, "line"))
target_rank = parse(Int, retrieve(conf, "target_rank"))
sparse_rel_thresh = parse(Float64, retrieve(conf, "sparse_rel_thresh"))
SNR_dB = parse(Float64, retrieve(conf, "SNR_dB"))
println("Parameters: line = $line, target rank = $target_rank, relative threshold for sparse part = $sparse_rel_thresh, SNR = $SNR_dB dB")

maxIter_IRLS = parse(Int, retrieve(conf, "maxIter_IRLS"))
N_IRLS = parse(Int, retrieve(conf, "N_IRLS"))
N_CG = parse(Int, retrieve(conf, "N_CG"))
N_AL2 = parse(Int, retrieve(conf, "N_AL2"))
N_ISTA = parse(Int, retrieve(conf, "N_ISTA"))
N_FISTA = parse(Int, retrieve(conf, "N_FISTA"))
N_POGM = parse(Int, retrieve(conf, "N_POGM"))
N_MSLR = parse(Int, retrieve(conf, "N_MSLR"))
println("Max iterations: IRLS = $N_IRLS, CG = $N_CG, AL-2 = $N_AL2, ISTA = $N_ISTA, FISTA = $N_FISTA, POGM = $N_POGM, MSLR = $N_MSLR")
println("Number of threads: ", Threads.nthreads())

println("Loading libraries...")
using FFTW, MAT, Plots, MIRT, LinearAlgebra, Random, JLD
include("algorithms.jl")
include("MRI_operators.jl")
include("MSLR.jl")
FFTW.set_num_threads(4)

println("Reading data...")
data = matread("../reproduce-l-s-dynamic-mri/data/aperiodic_pincat.mat")["new"]
X_orig = permutedims(data, [2, 1, 3]) ./ maximum(data)
nx,ny,nt = size(X_orig)

println("Simulate coils...")
nc = 8;
nring = 4;
b1,_ = ir_mri_sensemap_sim(dims=(nx,ny), dx=1.5, dz=1.5,
    ncoil=nc*nring, nring=nring, rcoil=120, coil_distance=1.2, chat=false)
b1c = b1 ./ repeat(sqrt.(sum(abs2.(b1), dims=3)),outer=[1,1,nc*nring])

smap0,_,_ = ir_mri_coil_compress(b1c, ncoil=nc);
smap = ComplexF64.(smap0 ./ sqrt.(sum(abs2.(smap0), dims=3)))

println("Simulated sampling mask")
#line = 24
Random.seed!(1)
samp = strucrand(nx, ny, nt, line)
pre_mask = fftshift(repeat(samp, 1, 1, 1, nc), 1:2)
mask = pre_mask .== 1

println("Create operators...")
E = getE(nx, ny, nt, nc, samp, smap, ComplexF64)

Ω, Q, C = getΩQC(nx, ny, nt, nc, samp, smap, ComplexF64)

T = getT(nx, ny, nt, ComplexF64)

println("Create low-rank matrix...")
F = svd(reshape(X_orig, nx*ny, nt))
orig_rank = sum(F.S .> 1e-8)
E_orig = sum(abs2, X_orig)
@show E_orig
F.S[target_rank+1:end] .= 0
X_low_rank = reshape(F.U * Diagonal(F.S) * F.Vt, nx, ny, nt)
E_low_rank = sum(abs2, X_low_rank)
@show E_low_rank

println("Calculate sparse part...")
X_sparse = X_orig - X_low_rank
E_sparse = sum(abs2, X_sparse)
@show E_sparse

threshold = maximum(abs2, X_sparse) * sparse_rel_thresh
X_sparse[abs2.(X_sparse) .≤ threshold] .= 0
s = sum(abs2.(X_sparse) .> 0)
@show s
E_sparse_trunc = sum(abs2, X_sparse)
@show E_sparse_trunc

println("Simulate acquisition...")
Xtrue = X_low_rank + X_sparse
Xtrue_rank = rank(reshape(Xtrue, nx*ny, nt))
@show Xtrue_rank
ytrue = E * ComplexF64.(Xtrue)

noise = randn(Float32, size(ytrue)) + randn(Float32, size(ytrue))im # complex noise!
scale_noise = norm(ytrue) / norm(noise) / 10^(SNR_dB / 20.f0)
y = ytrue + scale_noise * noise
rms(x) = norm(x-ytrue) / sqrt(length(x-ytrue))
snr(x) = 20 * log10(norm(ytrue) / norm(x-ytrue))
@show rms(y)
@show snr(y);

println("Run algorithms...")

X_irls, cost_irls, rankL_irls, time_irls = HM_IRLS(Xtrue, y, E, maxIter = maxIter_IRLS, N = N_IRLS, verbose = false)
@show cost_irls[end]
@show rankL_irls[end]

x₀ = E' * y
St = svdvals(reshape(x₀ - E' * reshape(E * x₀ - y, nx, ny, nt, nc), nx*ny, nt));

X_cg, rankL_cg, cost_cg, time_cg = AL_CG(Xtrue, y, E, T, scale_L = maximum(St), scale_S = 1/1.887,
    δ₁ = 1//2, δ₂ = 1//2, N = N_CG, verbose = false)
@show cost_cg[end]
@show rankL_cg[end]

X_al, rankL_al, cost_al, time_al = AL_2(Xtrue, y, Ω, Q, C, T, scale_L = maximum(St), scale_S = 1/1.887,
    δ₁ = 1//3, δ₂ = 1//10, N = N_AL2, verbose = false)
@show cost_al[end]
@show rankL_al[end]

X_ista, rankL_ista, cost_ista, time_ista = PGM(Xtrue, y, E, T, scale_L = maximum(St), scale_S = 1/1.887,
    momentum = :ista, N = N_ISTA, verbose = false)
@show cost_ista[end]
@show rankL_ista[end]

X_fista, rankL_fista, cost_fista, time_fista = PGM(Xtrue, y, E, T, scale_L = maximum(St), scale_S = 1/1.887,
    momentum = :fista, N = N_FISTA, verbose = false)
@show cost_fista[end]
@show rankL_fista[end]

X_pogm, rankL_pogm, cost_pogm, time_pogm = PGM(Xtrue, y, E, T, scale_L = maximum(St), scale_S = 1/1.887,
    momentum = :pogm, N = N_POGM, verbose = false)
@show cost_pogm[end]
@show rankL_pogm[end]

X_mslr, cost_mslr, time_mslr =
    MultiScaleLowRank(Xtrue, y, samp, λ = 1e-8, smap, max_epoch = N_MSLR, max_power_iter = 1);
@show cost_mslr[end]
@show time_mslr[end]

println("Save results...")
fname = "$(ARGS[1]).jld"
@show fname
@save(fname, line, target_rank, sparse_rel_thresh, SNR_dB,
    Xtrue, X_low_rank, X_sparse, E_orig, E_low_rank, E_sparse, s, E_sparse_trunc,
    X_irls, cost_irls, rankL_irls, time_irls,
    X_cg, rankL_cg, cost_cg, time_cg,
    X_al, rankL_al, cost_al, time_al,
    X_ista, rankL_ista, cost_ista, time_ista,
    X_fista, rankL_fista, cost_fista, time_fista,
    X_pogm, rankL_pogm, cost_pogm, time_pogm,
    X_mslr, cost_mslr, time_mslr)

println("Done!")
