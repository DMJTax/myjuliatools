export prcrossval,crossval, crossval!

"""
    prcrossval(a::Prdataset,u,k=10,testfun=testc)

Apply k-fold crossvalidation on dataset `a` by training untrained Prmapping `u` on to the training folds, and evaluate the trained mapping on the test fold. For evaluation the `testfun` function is used.
"""
function prcrossval(a::Prdataset,u,nrfolds=10,testfun=testc)
    if isa(u,Vector)
        nrcl = length(u)
    elseif isa(u,Prmapping)
        u = [u]
        nrcl = 1
    else
        error("Second input should be a (vector of) Prmapping(s).")
    end
    err = zeros(nrcl,nrfolds)
    I = crossval(nrfolds,getlabels(a))
    for j=1:nrfolds
        Itr,Itst = crossval!(I)
        x = a[Itr,:]
        z = a[Itst,:]
        for i = 1:nrcl
            w = x*u[i]
            err[i,j] = testfun(z*w)
        end
    end
    return err
end

"""
    I = crossval(K,N)
    I = crossval(K,y)
Perform `K`-fold crossvalidation (possibly stratified if a vector of
labels `y` is supplied, otherwise it is assumed we have `N` objects.)

Standard application would be:
```julia-repl
  nrfolds = 5
  I = crossval(nrfolds,lab)
  for i=1:nrfolds
     Itr,Itst = crossval!(I)
     x = data[Itr,:]
     z = data[Itst,:]
     err[i] = ...
  end
```
"""
function crossval(nrfolds,y)
    if length(nrfolds)>1
        error("crossval expects the number of folds as input.")
    end
    if length(y)>1   # we have a vector of labels for stratified xval
        n = length(y)
        J = zeros(Int,n)
        # go through each of the classes and split them in nrfolds folds:
        lablist = unique(y)
        for i=1:length(lablist)
            Ii = findall(y.==lablist[i])
            I = collect(0:length(Ii)-1)
            J[Ii] = rem.(I,nrfolds) .+ 1
        end
        return [nrfolds; 0; 0; 0; 170673; J]

    else  # standard crossval on n objects
        I = collect(0:y-1)
        J = rem.(I,nrfolds) .+ 1
        return [nrfolds; 0; 0; 0; 170673; J]
    end

end


"""
    I = crossval!(K,y)
Perform `K`-fold crossvalidation (possibly stratified if a vector of
labels `y` is supplied)

Standard application would be:
```julia-repl
  nrfolds = 5
  I = crossval(nrfolds,lab)
  for i=1:nrfolds
     Itr,Itst = crossval!(I)
     x = data[Itr,:]
     z = data[Itst,:]
     err[i] = ...
  end
```
"""
function crossval!(I)
    if I[5]!=170673
        error("Valid index vector I should be supplied.")
    end
    I[2] += 1
    testfold = I[2]
    J = I[6:end]
    Itest = (J.==testfold)
    return findall(.!Itest), findall(Itest)
end
