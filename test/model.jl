include("init.jl")

@testset "select_sequences" begin
    data = random_data()
    selseqflag = rand(Bool, number_of_sequences(data))
    selseqdata = select_sequences(data, selseqflag)
    @test selseqdata.sequences == data.sequences[:,:,selseqflag]
    @test selseqdata.counts == data.counts[selseqflag,:]
    @test selseqdata.ancestors == data.ancestors
end

@testset "simulate" begin
    data = random_data()
    A, L, S = size(data.sequences)
    R = number_of_rounds(data)
    states = (IndepSite(A,L), Epistasis(A,L), IndepSite(A,L), ZeroEnergy())
    W = length(states)
    select, washed = random_select(W, R)
    model = Model(states, randn(W,R), randn(R), select, washed)
    lp = log_selectivities(model, data)
    data_ = simulate(model, data)
    r = 0
    for (t, a) in enumerate(data.ancestors)
        if a > 0
            r += 1
            N0 = data_.counts[:,t]
            N1 = data_.counts[:,a] .* exp.(lp[:,r])
            @test N0 ./ sum(N0; dims = 1) ≈ N1 ./ sum(N1; dims = 1)
        end
    end
end

@testset "sample_reads" begin
    Random.seed!(1)
    data = random_data()
    data_ = BiophysViabilityModel.sample_reads(data, 10^6)
    @test data_.ancestors == data.ancestors
    @test data_.sequences == data.sequences
    @test normalize_counts(data.counts) ≈ normalize_counts(data_.counts) rtol=1e-1
end

@testset "log_likelihood" begin
    data = random_data()
    A, L, S = size(data.sequences)
    R = number_of_rounds(data)
    W = 4
    select, washed = random_select(W, R)
    states = (IndepSite(A,L), Epistasis(A,L), ConstEnergy(fill(randn())), ZeroEnergy())
    model = Model(states, randn(W, R), randn(R), select, washed)
    randn!(model.μ)
    randn!(model.ζ)

    lp = log_selectivities(model, data)
    lM = dropdims(log_multinomial(data.counts; dims=1); dims=1)
    Ls = lM .+ sum_(data.counts .* log_abundances(model, data); dims=1)

    @test log_likelihood(lp, model.ζ, data) ≈ log_likelihood(model, data)
    @test energies(data, model) == energies(data.sequences, model)
    @test log_selectivities(model, data; rare_binding=false) == log_selectivities(model, data)
    @test all(log_selectivities(model, data; rare_binding=false) .≤ log_selectivities(model, data; rare_binding=true))
    @test log_likelihood_samples(model, data) ≈ Ls / number_of_sequences(data)
    @test log_likelihood(model, data) ≈ sum(Ls) / number_of_sequences(data)
    @test log_likelihood(model, data) ≈ sum(log_likelihood_samples(model, data))

    function fun(x)
        model_ = Model(
            (IndepSite(x), states[2:end]...),
            model.μ, model.ζ, model.select, model.washed
        )
        return log_likelihood(model_, data)
    end

    ps = Flux.params(model)
    gs = gradient(ps) do
        log_likelihood(model, data)
    end
    @test gs[states[1].h] ≈ grad(central_fdm(5, 1), fun, model.states[1].h)[1]

    function gun(J)
        model_ = Model(
            (states[1], Epistasis(states[2].h, J), states[3:end]...),
            model.μ, model.ζ, model.select, model.washed
        )
        return log_likelihood(model_, data)
    end

    ps = Flux.params(model)
    gs = gradient(ps) do
        log_likelihood(model, data)
    end
    @test gs[states[2].J] ≈ grad(central_fdm(5, 1), gun, states[2].J)[1]
end

@testset "depletion_gradient" begin
    data = random_data()
    lp = randn(number_of_sequences(data), number_of_rounds(data))
    ζ = randn(number_of_rounds(data))
    lN = log_abundances(lp, ζ, data)
    gs = gradient(Flux.params(ζ)) do
        log_likelihood(lp, ζ, data)
    end
    G = depletion_gradient!(zero(ζ), lN, data)
    @test G ≈ -gs[ζ]
end

@testset "rare binding gauge" begin
    data = random_data()
    A, L, S = size(data.sequences)
    R = number_of_rounds(data)
    states = (IndepSite(A,L), IndepSite(A,L), IndepSite(A,L), IndepSite(A,L))
    W = length(states)
    select, washed = random_select(W, R)
    model = Model(states, randn(W, R), randn(R), select, washed)
    randn!(model.μ)
    randn!(model.ζ)

    model_ = rare_binding_gauge(model)
    @test norm(sum(model_.μ .* model_.select; dims=1)) ≤ 1e-10
    @test norm(sum(model_.μ .* model_.washed; dims=1)) ≤ 1e-10
    @test model_.states == model.states
    lpz  = log_selectivities(model,  data; rare_binding=true) .- reshape(model.ζ,  1, :)
    lpz_ = log_selectivities(model_, data; rare_binding=true) .- reshape(model_.ζ, 1, :)
    @test lpz_ ≈ lpz

    model__ = deepcopy(model)
    rare_binding_gauge!(model__)
    @test model_.μ == model__.μ
    @test model_.ζ == model__.ζ
    for w = 1:W
        @test model_.states[w].h == model__.states[w].h
    end
end

@testset "rare_binding_gauge_zeta" begin
    data = random_data()
    A, L, S = size(data.sequences)
    R = number_of_rounds(data)
    states = (IndepSite(A,L), IndepSite(A,L), IndepSite(A,L), IndepSite(A,L))
    W = length(states)
    select, washed = random_select(W, R)
    model = Model(states, randn(W, R), randn(R), select, washed)
    randn!(model.μ)
    randn!(model.ζ)

    model_ = rare_binding_gauge_zeta(model)
    @test iszero(model_.ζ)
    @test model_.states == model.states
    lpz  = log_selectivities(model,  data; rare_binding=true) .- reshape(model.ζ,  1, :)
    lpz_ = log_selectivities(model_, data; rare_binding=true) .- reshape(model_.ζ, 1, :)
    @test lpz_ ≈ lpz

    model__ = deepcopy(model)
    rare_binding_gauge_zeta!(model__)
    @test model_.μ == model__.μ
    @test model_.ζ == model__.ζ
    for w = 1:W
        @test model_.states[w].h == model__.states[w].h
    end
end

@testset "zerosum" begin
    data = random_data()
    A, L, S = size(data.sequences)
    R = number_of_rounds(data)
    states = (IndepSite(A,L), IndepSite(A,L), IndepSite(A,L), IndepSite(A,L))
    W = length(states)
    select, washed = random_select(W, R)
    model = Model(states, randn(W, R), randn(R), select, washed)
    randn!(model.μ)
    randn!(model.ζ)

    indep_model = IndepModel(model)
    indep_model_ = zerosum(indep_model)
    @test norm(mean(indep_model_.states.h; dims=1)) ≤ 1e-10
    @test indep_model.ζ == indep_model_.ζ
    @test indep_model.select == indep_model_.select
    @test indep_model.washed == indep_model_.washed
    e0 = energies(data, indep_model)  .- unsqueeze_left(indep_model.μ)
    e1 = energies(data, indep_model_) .- unsqueeze_left(indep_model_.μ)
    @test e0 ≈ e1
end
