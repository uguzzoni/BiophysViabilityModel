"""
    learn_mu!(model, data; kwargs...)

Train a tree model on data, learing only μ (all other model parameters are frozen
to their initial values).
"""
function learn_mu!(
    model::Model,
    data::Data;
    opt = Adam(), # optimizer
    epochs = 1:100, # epoch indices
    reg = () -> false, # regularization
    batchsize = number_of_sequences(data),
    callback = nothing,
    history = MVHistory(),
    rare_binding::Bool = false
)
    @assert number_of_rounds(model) == number_of_rounds(data)
    _epochs = epochs2range(epochs)
    for epoch = _epochs
        for batch in minibatches(data, batchsize)
            gs = gradient(model.μ) do μ
                _model = Model(model.states, μ, model.ζ, model.select, model.washed)
                ll = log_likelihood(_model, data; rare_binding, batch)
                @ignore_derivatives push!(history, :loglikelihood_batch, ll)
                @ignore_derivatives push!(history, :epoch, epoch)
                return reg() - ll
            end
            update!(opt, model.μ, only(gs))
            callback === nothing || callback()
        end
        push!(
            history,
            :loglikelihood,
            log_likelihood(model, data; rare_binding = rare_binding)
        )
    end
    return history
end
