struct IndepSite{H<:AbstractMatrix}
    h::H
end
@functor IndepSite
IndepSite(A::Int, L::Int, ::Type{T} = Float64) where {T} = IndepSite(randn(T,A,L) ./ T(√L))
function energies(sequences::Sequences, state::IndepSite)
    A, L, S = size(sequences)
    @assert size(state.h) == (A, L)
    Eh = -reshape(sequences, A * L, S)' * vec(state.h)
    @assert size(Eh) == (S,)
    return Eh
end

#
# inespecific constant energy model
#
struct ConstEnergy{E<:AbstractArray{<:Real,0}}
    e::E
end
@functor ConstEnergy
ConstEnergy(x::Real) = ConstEnergy(fill(x))
ConstEnergy(::Type{T} = Float64) where {T} = ConstEnergy(fill(zero(T)))
energies(sequences::Sequences, state::ConstEnergy) = repeat(state.e, size(sequences, 3))


#
# inespecific zero energy model
#
struct ZeroEnergy{E<:AbstractArray{<:Real,0}} end
ZeroEnergy(T::Type{<:Real} = Bool) = ZeroEnergy{Array{T,0}}()
@functor ZeroEnergy
energies(sequences::Sequences, state::ZeroEnergy) = energies(sequences, ConstEnergy(state))
function ChainRulesCore.rrule(::typeof(energies), sequences::Sequences, state::ZeroEnergy)
    zero_energies_pullback(_) = (NoTangent(), NoTangent(), NoTangent())
    return energies(sequences, state), zero_energies_pullback
end
ConstEnergy(::ZeroEnergy{E}) where {E} = ConstEnergy{E}(fill(false))


#
# deep neural network energy model
#
struct DeepEnergy{T}
    m::T
end
energies(sequences::Sequences, state::DeepEnergy) = vec(state.m(sequences))
@functor DeepEnergy

#
# KELSIC Model
#

function create_model_energy()
    model = Chain(
        x-> reshape(x, size(x,1), size(x,2), 1, size(x,3)), #adds the channel dimension
        Conv((20,7), 1 => 12, relu, pad=SamePad()),
        BatchNorm(12),
        MaxPool((20,2), stride=(1,2)),
        Conv((1,7), 12 => 24, relu, pad=SamePad()),
        BatchNorm(24),
        MaxPool((1,2),stride=(1,2)),
        Flux.flatten,
        x -> reshape(x, size(x,1), 1, :),
        Dense(672 => 128, relu),
        BatchNorm(1),
        Dense(128 => 64, relu),
        BatchNorm(1),
        x -> reshape(x, size(x,1), :),
        Dense(64 => 2, identity),
        @views x -> x[2,:] .- x[1,:]
    )
    return model
end

Kelsic_states = (ZeroEnergy(),DeepEnergy(create_model_energy()))


#
#create_model_energy() #kelsic data
#
function create_model_kelsic()
    model = Chain(
        x-> reshape(x, size(x,1), size(x,2), 1, size(x,3)), #adds the channel dimension
        Conv((20,7), 1 => 12, relu, pad=SamePad()),
        BatchNorm(12),
        MaxPool((20,2), stride=(1,2)),
        Conv((1,7), 12 => 24, relu, pad=SamePad()),
        BatchNorm(24),
        MaxPool((1,2),stride=(1,2)),
        Flux.flatten,
        x -> reshape(x, size(x,1), 1, :),
        Dense(672 => 128, relu),
        BatchNorm(1),
        Dense(128 => 64, relu),
        BatchNorm(1),
        x -> reshape(x, size(x,1), :),
        Dense(64 => 2, identity),
        @views x -> x[2,:] .- x[1,:]
    )
    return model
end

Kelsic_model() = DeepEnergy(create_model_kelsic())


