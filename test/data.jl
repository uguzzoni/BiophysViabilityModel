include("init.jl")

import BiophysViabilityModel:minibatches,MiniBatch
 
@testset "minibatches" begin
    @test minibatches(10, 3; fullonly=true)  == [1:3, 4:6, 7:9]
    @test minibatches(10, 3; fullonly=false) == [1:3, 4:6, 7:9, 10:10]
    @test minibatches(10, 3) == minibatches(10, 3; fullonly=false)
    @test minibatch_count(10, 3; fullonly=true) == 3
    @test minibatch_count(10, 3; fullonly=false) == 4

    data = random_data()
    @test MiniBatch(data).sequences == data.sequences
    @test MiniBatch(data).counts == data.counts
    @test MiniBatch(data).lRs == data.lRs
end

@testset "empirical selectivities" begin
    data = random_data()
    θ = selectivities(data)
    r = 0
    for (t, a) in enumerate(data.ancestors)
        if a > 0
            r += 1
            θ[:,r] ≈ data.counts[:,t] ./ data.counts[:,a]
        end
    end
    @test θ ≈ exp.(log_selectivities(data))
    @test all(sum(selectivities(data; normalize=true); dims = 1) .≈ 1)
end

@testset "data (repetition $rep)" for rep in 1:10
    data = random_data()
    @test number_of_libraries(data) + number_of_rounds(data) == number_of_samples(data)
    @test number_of_samples(data) == length(data.ancestors) == size(data.counts, 2)
    @test alphabet_size(data) == size(data.sequences, 1)
    @test sequence_length(data) == size(data.sequences, 2)
    @test number_of_sequences(data) == size(data.sequences, 3) == size(data.counts, 1)
end
