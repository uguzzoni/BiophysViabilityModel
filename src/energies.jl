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
#create NN model for kelsic data
#

struct Add_Channel end
struct Inflate_Mat end
struct Squeeze_Mat end
struct Output_Diff end

function (a::Add_Channel)(x)
    return reshape(x, size(x,1), size(x,2), 1, size(x,3))
end

function (a::Inflate_Mat)(x)
    return reshape(x, size(x,1), 1, :)
end

function (a::Squeeze_Mat)(x)
    return reshape(x, size(x,1), :)
end

function (a::Output_Diff)(x)
    return @views x[2,:] .- x[1,:]
end

function create_model_kelsic()
    model = Chain(
        Add_Channel(), #adds the channel dimension
        Conv((20,7), 1 => 12, relu, pad=SamePad()),
        BatchNorm(12),
        MaxPool((20,2), stride=(1,2)),
        Conv((1,7), 12 => 24, relu, pad=SamePad()),
        BatchNorm(24),
        MaxPool((1,2),stride=(1,2)),
        Flux.flatten,
        Inflate_Mat(),
        Dense(672 => 128, relu),
        BatchNorm(1),
        Dense(128 => 64, relu),
        BatchNorm(1),
        Squeeze_Mat(),
        Dense(64 => 2, identity),
        Output_Diff()
    )
    return model
end

Kelsic_model() = DeepEnergy(create_model_kelsic())


