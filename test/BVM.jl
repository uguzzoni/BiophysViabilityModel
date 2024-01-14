using Test, BiophysViabilityModel, Random, LinearAlgebra, OneHot, Flux, FiniteDifferences
import ChainRulesTestUtils
using BiophysViabilityModel: unsqueeze_left, unsqueeze_right, split_states

A = 3; L = 5; T = 5; W = 4; S = 128
ancestors = (0,1,1,2,2)
R = BiophysViabilityModel.number_of_edges(ancestors)
sequences = falses(A,L,S)
@assert all(sum(sequences; dims = 1) .== 0)
for s=1:S, i=1:L
    sequences[rand(1:A),i,s] = true
end
@assert all(sum(sequences; dims = 1) .== 1)
data = Data(sequences, rand(S,T), ancestors)

select, washed = falses(W,R), falses(W,R)
for r=1:R
    w = rand(1:W)
    select[w, r] = true
    w_ = rand(setdiff(1:W, w))::Int
    @assert 1 ≤ w_ ≤ W && w_ ≠ w
    washed[w_, r] = true
end
model = IndepModel(randn(A,L,2) ./ L, W - 2, randn(W,R) ./ L, randn(R) ./ R, select, washed)
@assert number_of_rounds(model) == number_of_rounds(data)

data = simulate(model, data)
@test all(sum(data.counts; dims = 1) .≈ 1)

Random.seed!(1)
data_ = BiophysViabilityModel.sample_reads(data, 10^6)
@testset "sample_reads" begin
    @test data_.ancestors == data.ancestors
    @test data_.sequences == data.sequences
    @test data.counts ./ sum(data.counts; dims = 1) ≈ data_.counts ./ sum(data_.counts; dims = 1) rtol=0.1
end

model = split_states(model)
# make sure states[1] is relevant, otherwise gradient below is 0 and ≈ comparison fails
model.select[1,:] .= model.washed[2,:] .= true
model.washed[1,:] .= model.select[2,:] .= false
function fun(x)
    model_ = Model(
        (IndepSite(x), model.states[2:end]...),
        model.μ, model.ζ, model.select, model.washed
    )
    log_likelihood(model_, data)
end
ps = Flux.params(model)
gs = gradient(ps) do
    log_likelihood(model, data)
end
@test gs[model.states[1].h] ≈ grad(central_fdm(5, 1), fun, model.states[1].h)[1]
