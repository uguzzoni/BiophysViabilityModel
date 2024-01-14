using Test, Random, LinearAlgebra, Statistics
using Flux
using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences
using AbstractTrees

using BiophysViabilityModel
using BiophysViabilityModel: sum_, mean_, logsumexp_, select_mask
using BiophysViabilityModel: number_of_rounds, number_of_libraries, normalize_counts
using BiophysViabilityModel: depletion_gradient!
using BiophysViabilityModel: log_multinomial, invsigmoid, invlogsigmoid
using BiophysViabilityModel: collect_all_possible_sequences, collect_all_possible_sequences_onehot
using BiophysViabilityModel: unsqueeze_left, unsqueeze_right
using BiophysViabilityModel: minibatches, minibatch_count
using BiophysViabilityModel: rare_binding_gauge, rare_binding_gauge!,
    rare_binding_gauge_zeta, rare_binding_gauge_zeta!
using BiophysViabilityModel: tensordot, xexpx, xexpy

# ancestors
using BiophysViabilityModel: valid_ancestors, number_of_nodes, number_of_edges, number_of_roots
using BiophysViabilityModel: isroot, isleaf, find_root
using BiophysViabilityModel: node_costs, edge_costs
using BiophysViabilityModel: subtree_sum, tree_sum, subtree_maximum, tree_maximum, tree_logsumexp

"""
    random_ancestors(n)

Creates a random ancestors list of length `n`.
"""
function random_ancestors(n::Int)
    ancestors = ntuple(i -> rand(0:(i - 1)), n)
    @assert valid_ancestors(ancestors)
    return ancestors
end

function random_data(; A::Int = 3, L::Int = 4, T::Int = 7, W::Int = 4, S::Int = 128)
    sequences = falses(A, L, S)
    for s=1:S, i=1:L
        sequences[rand(1:A), i, s] = true
    end
    ancestors = random_ancestors(T)
    counts = rand(S, T)
    return Data(sequences, counts, ancestors)
end

function random_select(W::Int, R::Int)
    select = falses(W, R)
    for t in 1:R
        select[rand(1:W), t] = true
    end
    washed = (!).(select)
    return select, washed
end
