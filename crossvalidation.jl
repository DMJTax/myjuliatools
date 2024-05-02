export crossval, crossval!
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
