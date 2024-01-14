using Test, PhageTree, LinearAlgebra, OneHot, Flux, Zygote
using PhageTree: cgf, sum_, hidden_inputs

A = 4; L = 7; S = 5; m = 2
seqs = OneHotArray(rand(1:A, L, S), A)

@testset "gaussian" begin
    @test eltype(Gaussian(2, Float32).θ) == eltype(Gaussian(2, Float32).γ) == Float32
    @test eltype(Gaussian(2).θ) == eltype(Gaussian(2).γ) == Float64

    hlayer = Gaussian(m)
    I = randn(m, S)
    @test size(cgf(I, hlayer)) == (m, S)
    ps = Flux.params(hlayer)
    gs = Zygote.gradient(ps) do
        sum(cgf(I, hlayer))
    end
    μ = (hlayer.θ .+ I) ./ abs.(hlayer.γ)
    ν = inv.(abs.(hlayer.γ))
    @test gs[hlayer.θ] ≈ sum_(μ; dims = 2)
    @test gs[hlayer.γ] ≈ sum_(-(ν .+ μ.^2) / 2; dims = 2)
end

@testset "bernoulli" begin
    @test eltype(Bernoulli(2, Float32).θ) == Float32
    @test eltype(Bernoulli(2).θ) == Float64

    hlayer = Bernoulli(2)
    I = randn(2, S)
    @test size(cgf(I, hlayer)) == (m, S)
    ps = Flux.params(hlayer)
    gs = gradient(ps) do
        sum(cgf(I, hlayer))
    end
    μ = sigmoid.(hlayer.θ .+ I)
    @test gs[hlayer.θ] ≈ sum_(μ; dims = 2)
end

@testset "rbm" begin
    m = 2
    for H in (Gaussian, Bernoulli)
        @test size(energies(seqs, RBM(A, L, H(m)))) == (S,)
        @test eltype(RBM(A, L, H(m, Float32)).weights) == Float32
        @test eltype(RBM(A, L, H(m, Float32)).fields) == Float32
        @test eltype(RBM(A, L, H(m)).weights) == Float64
        @test eltype(RBM(A, L, H(m)).fields) == Float64
        @test size(hidden_inputs(seqs, RBM(A, L, H(m)))) == (m, S)
    end
end
