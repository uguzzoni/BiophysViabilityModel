module BiophysViabilityModel
    using Random, LinearAlgebra, Statistics
    using Base: front, tail
    using Flux: gradient, params, @functor, update!, Adam, logsoftmax, flatten
    using ChainRulesCore: rrule, NoTangent, unthunk
    using ChainRulesCore: ignore_derivatives, @ignore_derivatives, @non_differentiable
    using SpecialFunctions: loggamma
    using AbstractTrees: PreOrderDFS
    using ValueHistories: MVHistory
    using OneHot: OneHotArray
    using Optim: optimize, only_fg!, LBFGS, NelderMead, ZerothOrderOptimizer, Options
    using Distributions: Multinomial
    using LogExpFunctions: logexpm1, logsumexp, log1pexp, log1mexp
    import Flux, AbstractTrees, ChainRulesCore
    import Flux: Chain, Conv, SamePad, BatchNorm, MaxPool, Dense, relu, identity

    export Data, Experiment, Model
    export energies, log_selectivities, selectivities
    export DeepEnergy, ZeroEnergy, learn!
    export number_of_states, number_of_rounds, number_of_libraries, number_of_samples, number_of_sequences, sequence_length, alphabet_size
    export log_likelihood, log_likelihood_samples, log_abundances

    include("data.jl")
    include("energies.jl")
    include("model.jl")
    include("util.jl")
    include("ancestors.jl")
    include("node.jl")
    include("simulate.jl")
    include("indep_model.jl")
    include("analysis.jl")

end # module
