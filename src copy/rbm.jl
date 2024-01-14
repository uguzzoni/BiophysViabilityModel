abstract type AbstractRBMLayer{T<:Real} end

struct RBM{T<:Real, H<:AbstractRBMLayer{T}}
    fields::Matrix{T}
    weights::Array{T,3}
    hlayer::H
end
function RBM(A::Int, L::Int, hlayer::AbstractRBMLayer{T}) where {T}
    fields = randn(T, A, L)
    weights = randn(T, A, L, number_hidden(hlayer)) / convert(T, √length(fields))
    return RBM(fields, weights, hlayer)
end

number_hidden(rbm) = size(rbm.weights, 3)
@functor RBM

function energies(sequences::Sequences, rbm::RBM)
    A, L, _ = size(sequences)
    @assert size(rbm.fields) == (A, L)
    I = hidden_inputs(sequences, rbm)
    return reshape(sequences, A * L, :)' * vec(rbm.fields) + sum_(cgf(I, rbm.hlayer); dims = 1)
end

function hidden_inputs(sequences::Sequences, rbm::RBM)
    @assert number_hidden(rbm) == number_hidden(rbm.hlayer)
    A, L, _ = size(sequences)
    return reshape(rbm.weights, A * L, :)' * reshape(sequences, A * L, :)
end


struct Gaussian{T<:Real} <: AbstractRBMLayer{T}
    θ::Vector{T}
    γ::Vector{T}
end
Gaussian(m::Int, ::Type{T} = Float64) where {T} = Gaussian(zeros(T, m), ones(T, m))
function number_hidden(hlayer::Gaussian)
    @assert length(hlayer.θ) == length(hlayer.γ)
    return length(hlayer.θ)
end
function cgf(I::Matrix, hlayer::Gaussian)
    result = @. (hlayer.θ + I)^2 / abs(hlayer.γ) / 2
    π_ = convert(eltype(result), π)
    return @. result - log(abs(hlayer.γ) / π_ / 2) / 2
end
@functor Gaussian


struct Bernoulli{T<:Real} <: AbstractRBMLayer{T}
    θ::Vector{T}
end
Bernoulli(m::Int, ::Type{T} = Float64) where {T} = Bernoulli(zeros(T, m))
cgf(I::Matrix, hlayer::Bernoulli) = @. log1pexp(hlayer.θ + I)
number_hidden(hlayer::Bernoulli) = length(hlayer.θ)
@functor Bernoulli
