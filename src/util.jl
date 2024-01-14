# these functions reduce a dimension and drop it
logsumexp_(A::AbstractArray; dims = :) = dropdims(logsumexp(A; dims = dims); dims = dims)
sum_(A::AbstractArray; dims = :) = dropdims(sum(A; dims = dims); dims = dims)
mean_(A::AbstractArray; dims = :) = dropdims(mean(A; dims = dims); dims = dims)

"""
    collect_all_possible_sequences(A, L)

Generate all possible A^L sequences of length L from the alphabet 1:A.
Returns a L x (A^L) Potts matrix.
"""
function collect_all_possible_sequences(A::Integer, L::Integer)
    generator = Iterators.product(repeat([1:A], L)...)
    return hcat(collect.(vec(collect(generator)))...)
end

"""
    collect_all_possible_sequences_onehot(A, L)

Generate all possible A^L sequences of length L from the alphabet 1:A.
Returns a A x L x (A^L) one-hot tensor.
"""
function collect_all_possible_sequences_onehot(A::Integer, L::Integer)
    potts = collect_all_possible_sequences(A, L)
    return BitArray(OneHotArray(potts, A))
end

NumArray{T<:Number,N} = AbstractArray{<:T,N}

"""
	tensordot(X, W, Y)

`X*W*Y`, contracting all dimensions of `W` with the corresponding first
dimensions of `X` and `Y`, and matching the remaining last dimensions of
`X` to the remaining last dimensions of `Y`.

For example, `C[b] = sum(X[i,j,b] * W[i,j,μ,ν] * Y[μ,ν,b])`.
"""
function tensordot(X::NumArray, W::NumArray, Y::NumArray)
	xsize, ysize, bsize = tensorsizes(X, W, Y)
	Xmat = reshape(X, prod(xsize), prod(bsize))
	Ymat = reshape(Y, prod(ysize), prod(bsize))
	Wmat = reshape(W, prod(xsize), prod(ysize))
	if size(Wmat, 1) ≥ size(Wmat, 2)
		Cmat = sum(Ymat .* (Wmat' * Xmat); dims = 1)
	else
		Cmat = sum(Xmat .* (Wmat * Ymat); dims = 1)
	end
	return reshape(Cmat, bsize)
end

function tensorsizes(X::NumArray, W::NumArray, Y::NumArray)
	@assert iseven(ndims(X) + ndims(Y) - ndims(W))
	bdims = div(ndims(X) + ndims(Y) - ndims(W), 2)
	@assert ndims(X) ≥ bdims && ndims(Y) ≥ bdims
	xdims = ndims(X) - bdims
	ydims = ndims(Y) - bdims
	xsize = ntuple(d -> size(X, d), xdims)
	ysize = ntuple(d -> size(Y, d), ydims)
	bsize = ntuple(d -> size(X, d + xdims), bdims)
	@assert size(W) == (xsize..., ysize...)
	@assert size(X) == (xsize..., bsize...)
	@assert size(Y) == (ysize..., bsize...)
	return xsize, ysize, bsize
end

"""
    broadlike(A, B...)

Broadcasts `A` to the size of `A .+ B .+ ...`, without actually summing anything.
"""
broadlike(A, B...) = broadcast(first ∘ tuple, A, B...)

"""
	invsigmoid(y)

Inverse sigmoid function.
"""
invsigmoid(y::Real) = log(y / (1 - y))

"""
	invlogsigmoid(y)

Inverse logsigmoid function.
"""
invlogsigmoid(y::Real) = -logexpm1(-y)

"""
    select_mask(select)

Given an array `select`, returns another array with `Inf` where `select` is > 0,
and zeros elsewhere.
"""
select_mask(select::AbstractArray) = (select .> 0) .* Inf

unsqueeze_left(A::AbstractArray) = reshape(A, 1, size(A)...)
unsqueeze_right(A::AbstractArray) = reshape(A, size(A)..., 1)

"""
    log_multinomial(N; dims = :)

Log of multinomial coefficients, reduced across the given dimension of `N`.
"""
function log_multinomial(N::AbstractArray; dims = :)
    return loggamma.(sum(N; dims) .+ 1) .- sum(loggamma.(N .+ 1); dims)
end

#=
The functions xexpx(x) = x * exp(x), xexpy(x,y) = x * exp(y), treat 'x' as a strong zero,
so that xexpx(0) = 0 and xexpy(0, y) = 0. See:

    https://github.com/JuliaStats/LogExpFunctions.jl/issues/15
    https://github.com/JuliaStats/StatsFuns.jl/pull/58
=#
xexpx(x::Real) = isfinite(x) ? x * exp(x) : exp(x)
xexpy(x::T, y::T) where {T<:Real} = isnan(x) || isfinite(y) ? x * exp(y) : exp(y)
xexpy(x::Real, y::Real) = xexpy(promote(x, y)...)
