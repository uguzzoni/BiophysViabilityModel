"""
    selection_kl(model, data; rare_binding = false)

Computes the selection KL divergence of each round. That is,

    KL(N[t] || N[parent of t])

where N[t] and N[parent of t] are the phage abundances at sample 't' and its parent.
"""
function selection_kl(model::Model, data::Data; rare_binding::Bool = false)
    lN = log_abundances(model, data; rare_binding = rare_binding)
    H = entropies(lN)
    C = [selection_cross_entropy(lN[:,t], lN[:,a]) for (t, a) in enumerate(data.ancestors) if a > 0]
    return C - H
end

function entropies(model::Model, data::Data; rare_binding::Bool = false)
    lN = log_abundances(model, data; rare_binding = rare_binding)
    return entropies(lN)
end

entropies(lN::AbstractMatrix) = -sum(xexpx.(lN); dims = 1)
selection_cross_entropy(lNt::AbstractMatrix, lNa::AbstractMatrix) = -dot(exp.(lNt), lNa)
