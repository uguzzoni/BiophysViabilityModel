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

    export Data, Model, IndepModel, MiniBatch
    export energies, log_selectivities, selectivities
    export log_likelihood, log_likelihood_samples
    export ConstEnergy, ZeroEnergy, AndEnergy
    export IndepSite, Epistasis
    export DeepEnergy, SimpleAR
    export RBM, Gaussian, Bernoulli
    export learn!
    export simulate
    export number_of_rounds, number_of_libraries, number_of_samples, number_of_sequences
    export alphabet_size, sequence_length, select_sequences
    export log_abundances
    export number_of_states
    export optimize_depletion!, optimize_depletion
    export tensordot
    export zerosum, rare_binding_gauge, rare_binding_gauge!
    export Δenergies, flip, flip!, MCParetoTrial!, MCPareto!, ToParetoFront!


    include("data.jl")
    include("energies.jl")
    include("model.jl")
    include("util.jl")
    include("rbm.jl")
    include("ancestors.jl")
    include("node.jl")
    include("simulate.jl")
    include("indep_model.jl")
    include("optimize_depletion.jl")
    include("analysis.jl")
    include("learn_mu.jl")

end # module
