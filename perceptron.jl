export perlc

"""
     w = perlc(a, η=0.01,max_iter=1e4)

Fit a perceptron linear classifier on dataset `a`, trained by optimizing the perceptron loss using gradient descent. The learning rate is `η`, and the maximum number of updates is given by `max_iter`.
"""
function perlc(eta=0.01,max_iter=1e4)
    # This is the untrained mapping
    params = Dict{String,Any}("eta"=>eta,"max_iter"=>max_iter)
    return Prmapping("Perceptron Linear Classifier","untrained",fitPerlc!,predictPerlc,params,nothing)
end
# The call directly with some dataset:
function perlc(a::Prdataset, eta=0.01,max_iter=1e4)
    return a*perlc(eta,max_iter)
end

function fitPerlc!(w,a)
    eta = w.data["eta"]         # learning rate
    max_iter = w.data["max_iter"] # max. number of iterations

    # get the data
    N,dim = size(a)
    X = [+a ones(N)]
    if (nrclasses(a)!=2)
        error("The logistic classifier works for two classes only.")
    end
    y = 2 .*a.nlab .- 3
    signedX = X.*y
    weights = zeros(dim+1)
    # perform gradient descent on perceptron loss:
    nonzeroerror = true
    t = 0
    while ((t<max_iter) & nonzeroerror)
        t += 1
        # classify all training data, and find the errors
        pred = signedX*weights
        wrong = (pred .<= 0)
        println("Iteration ",t,": err = ",mean(wrong))
        nonzeroerror = (mean(wrong)>0)
        if nonzeroerror
            # weight update:
            weights .+= eta*vec(sum(signedX[findall(wrong),:], dims=1))
        end
    end
    # store results
    w.data["weights"] = weights
    w.labels = a.lablist
    w.nrin = dim
    w.nrout = 2  # aiai, hardcoded!!

    return w
end
function predictPerlc(w,a)
    weights = w.data["weights"]
    N = size(a,1)
    X = [+a ones(N)]
    pred = X*weights

    b = deepcopy(a)
    b.data = [-pred pred]
    b.featlab = w.labels
    return b
end

