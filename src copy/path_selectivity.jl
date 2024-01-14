"""
    log selectivities(data, round_pairs; pseudocount = 0, normalize = false)
    of a list of node pairs
"""
function log_selectivity(data::Data, round_pairs;  pseudocount::Real=0, normalize::Bool=false)
    c = data.counts .+ pseudocount;
    N = c ./ sum(c; dims = 1);
    θs = [N[:,t] ./ N[:,a] for (t,a) in round_pairs]
    θ = hcat(θs...)
    if normalize
        return log.(θ ./ sum(θ; dims = 1))
    else
        return log.(θ)
    end
end


"""
selectivity along a path connecting two nodes (0 if are not on the same tree branch)
"""
function path_log_selectivity(
    node1::Int,node2::Int,
    model::Model,
    sequences::Sequences;
    rare_binding::Bool=false
)
    pth = collect_path(node1, node2, data.ancestors)
    if pth==0 return 0 end
    pth.-=1
    path_model = PhageTree.Model(model.states, model.μ[:,pth], model.ζ[pth], model.select[:,pth], model.washed[:,pth]);
    return sum(log_selectivities(path_model,sequences;rare_binding=rare_binding),dims=2);
end

function path_log_selectivity(
    node1::Int,node2::Int,
    model::Model,
    data::Data;
    rare_binding::Bool=false
)
    pth = collect_path(node1, node2, data.ancestors)
    if pth==0 return 0 end
    pth.-=1
    path_model = PhageTree.Model(model.states, model.μ[:,pth], model.ζ[pth], model.select[:,pth], model.washed[:,pth]);
    return sum(log_selectivities(path_model,data;rare_binding=rare_binding),dims=2);
end

"""
selectivities of list of paths connecting two nodes (0 if are not on the same tree branch)
"""
function path_log_selectivities(
    round_pairs,
    model::Model,
    data::Data;
    rare_binding::Bool=false
)
    sel=[]
    for (n1,n2) in round_pairs
        push!(sel,path_log_selectivity(n1,n2,model,data;rare_binding=rare_binding))
    end
    return hcat(sel...)
end

function path_log_selectivities(
    round_pairs,
    model::Model,
    sequences::Sequences;
    rare_binding::Bool=false
)
    sel=[]
    for (n1,n2) in round_pairs
        push!(sel,path_log_selectivity(n1,n2,model,sequences;rare_binding=rare_binding))
    end
    return hcat(sel...)
end


"""
    collect_path(node1, node2, experiments...)

Traverses nodes from node1 backward to node2 (if parent) and collects them.
otherwise return 0
"""
function collect_path(node1::Int, node2::Int, ancestors)
    @assert valid_ancestors(ancestors)
    @assert 1 ≤ node1 ≤ length(ancestors)
    @assert 1 ≤ node2 ≤ length(ancestors)

    if node1<node2
        t=node1
        node1=node2
        node2=t
    end

    edges=[]
    a = node1
    while a > 0 && a!=node2
        push!(edges,a)
        a = ancestors[a]
    end
    if a==0 return 0 end
    return edges[end:-1:1]
end
