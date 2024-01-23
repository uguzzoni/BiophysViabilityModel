include("init.jl")

@testset "logsumexp_" begin
    A = randn(5,4,3)
    S = logsumexp(A; dims = 2)
    S_ = logsumexp_(A; dims = 2)
    @test vec(S_) ≈ vec(S)
    @test size(S) == (5,1,3)
    @test size(S_) == (5,3)
end

@testset "sum_" begin
    A = randn(5,4,6)
    X = sum_(A; dims = (1,2))
    @test vec(X) ≈ vec(sum(A; dims = (1,2)))
    @test size(X) == (6,)
end

@testset "mean_" begin
    A = randn(5,4,6)
    X = mean_(A; dims = (1,2))
    @test vec(X) ≈ vec(mean_(A; dims = (1,2)))
    @test size(X) == (6,)
end

@testset "generate all sequences" begin
    collect_all_possible_sequences(2, 3) == [1 2 1 2 1 2 1 2;
                                             1 1 2 2 1 1 2 2;
                                             1 1 1 1 2 2 2 2]
    collect_all_possible_sequences(3, 3) == [1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3;
                                             1 1 1 2 2 2 3 3 3 1 1 1 2 2 2 3 3 3 1 1 1 2 2 2 3 3 3;
                                             1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3]
    A, L = 4,5
    X = collect_all_possible_sequences(A, L)
    O = collect_all_possible_sequences_onehot(A, L)
    @test size(X) == (L, A^L)
    @test size(O) == (A, L, A^L)
    for s = 1:A^L, i = 1:L, a = 1:A
        @test O[a,i,s] == (X[i,s] == a)
    end
end

@testset "invsigmoid" begin
    for x in 1:5
        @test invsigmoid(sigmoid(+x)) ≈ +x
        @test invsigmoid(sigmoid(-x)) ≈ -x
        @test sigmoid(invsigmoid(1/x)) ≈ 1/x
    end
end

@testset "invlogsigmoid" begin
    for x in 1:5
        @test invlogsigmoid(logsigmoid(+x)) ≈ +x
        @test invlogsigmoid(logsigmoid(-x)) ≈ -x
        @test logsigmoid(invlogsigmoid(-x)) ≈ -x
    end
end

@testset "select_mask" begin
    select = rand(Bool, 10, 12)
    mask = select_mask(select)
    @test size(mask) == size(select)
    @test mask == replace(select, true => Inf, false => 0)

    select = randn(10, 12)
    @test select_mask(select) == select_mask(select .> 0)
    mask = select_mask(select)
    @test size(mask) == size(select)
    @test mask == replace(select .> 0, true => Inf, false => 0)
end

@testset "unsqueeze" begin
    A = randn(12, 11)

    A_ = unsqueeze_left(A)
    @test size(A_) == (1, size(A)...)
    @test vec(A_) == vec(A)
    @inferred unsqueeze_left(A)

    A_ = unsqueeze_right(A)
    @test size(A_) == (size(A)..., 1)
    @test vec(A_) == vec(A)
    @inferred unsqueeze_right(A)
end

@testset "log_multinomial" begin
    for _ = 1:10
        N = rand(1:5, 2, 3, 2)
        M = log.(factorial.(sum(N; dims = 1)) ./ prod(factorial, N; dims = 1))
        @test log_multinomial(N; dims = 1) ≈ M
    end
end

@testset "tensordot" begin
    X = randn(3,4,5,6)
    Y = randn(7,2,5,6)
    W = randn(3,4,7,2)
    C = zeros(5,6)
    for b2=1:6, b1=1:5, j2=1:2, j1=1:7, i2=1:4, i1=1:3
        C[b1,b2] += X[i1,i2,b1,b2]*W[i1,i2,j1,j2]*Y[j1,j2,b1,b2]
    end
    @test size(tensordot(X, W, Y)) == (5, 6)
    @test tensordot(X, W, Y) ≈ C
    @inferred tensordot(X, W, Y)

    X = randn(7,2,5,6)
    Y = randn(3,4,5,6)
    W = randn(7,2,3,4)
    C = zeros(5,6)
    for b2=1:6, b1=1:5, j2=1:4, j1=1:3, i2=1:2, i1=1:7
        C[b1,b2] += X[i1,i2,b1,b2]*W[i1,i2,j1,j2]*Y[j1,j2,b1,b2]
    end
    @test size(tensordot(X, W, Y)) == (5, 6)
    @test tensordot(X, W, Y) ≈ C
    @inferred tensordot(X, W, Y)

    A = randn(10, 10)
    v = randn(10, 100)
    @test tensordot(v, A, v) ≈ diag(v' * A * v)
end

@testset "xexpx" begin
    @test iszero(xexpx(-Inf))
    @test iszero(xexpx(0))
    @test isnan(xexpx(NaN))

    @test xexpx(1) ≈ exp(1)
    @test xexpx(2) ≈ 2exp(2)

    @inferred xexpx(0)
    @inferred xexpx(0.)
    @inferred xexpx(0.0f0)
end

@testset "xexpy" begin
    @test iszero(xexpy(Inf, -Inf))

    @test isnan(xexpy(NaN, -Inf))
    @test isnan(xexpy(NaN, 1))
    @test isnan(xexpy(0, NaN))

    @inferred xexpy(0,-Inf)
    @inferred xexpy(1,1)
    @inferred xexpy(1.,-Inf)

    @test xexpy(2,3) ≈ 2exp(3)

    for x = -10:10, y = -10:10
        @test xexpy(x,y) ≈ x * exp(y)
    end
end
