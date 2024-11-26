export loglc
"""
     w = loglc(a, λ=0.001,η=0.01,max_iter=1e4)

Fit a logistic classifier on dataset `a`, regularized by regularization parameter `λ`, trained by optimizing the loglikelihood using gradient descent. The learning rate is `η`, and the maximum number of updates is given by `max_iter`.
"""
function loglc(lambda=0.001,eta=0.01,max_iter=1e4)
    # This is the untrained mapping
    params = Dict{String,Any}("lambda"=>lambda, "eta"=>eta,"max_iter"=>max_iter)
    return Prmapping("Logistic Classifier","untrained",fitLoglc!,predictLoglc,params,nothing)
end
# The call directly with some dataset:
function loglc(a::Prdataset,lambda=0.001, eta=0.01,max_iter=1e4)
    return a*loglc(lambda,eta,max_iter)
end

function fitLoglc!(w,a)
    lambda = w.data["lambda"]   # regularizer
    eta = w.data["eta"]         # learning rate
    max_iter = w.data["max_iter"] # max. number of iterations
    min_rel_change = 1e-6       # minimum relative change in likelihood
    verysmall = 1e-12           # avoid log of 0

    # get the data
    N,dim = size(a)
    X = [+a ones(N)]
    if (nrclasses(a)!=2)
        error("The logistic classifier works for two classes only.")
    end
    y = (a.nlab.==1) 
    noty = (a.nlab.!=1)
    weights = zeros(dim+1)
    # perform gradient descent on loglikelihood:
    LL = 4*N*log(2) # something larger than newLL:
    newLL = 2*N*log(2) # something larger than N log(2)
    t = 0
    while ((abs(LL)-abs(newLL))>min_rel_change*abs(LL)) & (t<max_iter)
        t += 1
        fx = 1.0 ./(1.0 .+ exp.(-X*weights))
        # gradient:
        dLLdw = sum(repeat(fx .- y,1,dim+1).*X, dims=1)
        # keep track of the loglikelihood:
        LL = newLL
        newLL = y'*log.(fx.+verysmall) + noty'*log.(1.0.-fx.+verysmall)
        #println("Iteration ",t,": newLL = ",newLL)
        # weight update:
        weights .-= eta*(dLLdw' .+ 2.0*lambda*weights)
    end
    # store results
    w.data["weights"] = weights
    w.labels = a.lablist
    w.nrin = dim
    w.nrout = 2  # aiai, hardcoded!!

    return w
end
function predictLoglc(w,a)
    weights = w.data["weights"]
    N = size(a,1)
    X = [+a ones(N)]
    pred = 1.0 ./(1.0 .+ exp.(-X*weights))

    b = deepcopy(a)
    b.data = [pred 1.0.-pred]
    b.featlab = w.labels
    return b
end
