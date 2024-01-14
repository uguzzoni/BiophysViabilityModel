"""
    simulate(model, data; rare_binding = false)

Simulates `model`, using initial abundances from `data`.
"""
function simulate(model::Model, data::Data; rare_binding::Bool = false)
    lp = log_selectivities(model, data; rare_binding = rare_binding)
    lN = log.(data.counts)
    r = 0
    for (t, a) in enumerate(data.ancestors)
        if a > 0
            lN[:,t] .= lp[:, r += 1] .+ lN[:,a]
        end
    end
    lN .-= logsumexp(lN; dims = 1) # normalize
    return Data(data.sequences, exp.(lN), data.ancestors)
end

"""
    sample_reads(data, R)

Generates a new data with sampled reads.
"""
function sample_reads(data::Data, R::Vector{Int})
    @assert length(R) == number_of_samples(data)
    p = data.counts ./ sum(data.counts; dims = 1)
    reads = zeros(Int, size(data.counts))
    for t in eachindex(data.ancestors)
        reads[:,t] = rand(Multinomial(R[t], p[:,t]))
    end
    return Data(data.sequences, reads, data.ancestors)
end

function sample_reads(data::Data, R::Int)
    R_ = fill(R, number_of_samples(data))
    return sample_reads(data, R_)
end

function sample_reads(data::Data)
    R = sum_(data.counts; dims = 1)
    return sample_reads(data, R)
end
