# Try to define prmapping: what do we need to make it work?
#
# A prmapping can be in different 'states':
# - untrained:  parameters of the model are not defined yet
#               (data==nothing)
# - trained:    parameters of the model are defined
# - fixed:      parameters may be given, but the fit-function is not
#               defined (fit==nothing)
# simple least squares fitting
#
# To define a mapping, the user has to define 3 things:
# 1. an untrained mapping where hyperparameters can be defined/set:
#     u = linearr(degree=2)
# 2. a function that accepts a prdataset such that it can be trained:
#     w = fitf!(param, dataset)
#    It is expected that this fitting function is updating the
#    parameters param 
# 3. a function that processes a dataset to get a prediction out:
#     pred = predictf(param, dataset)
#
# An example is given by the linear regression `linearr`
using Statistics
using Plots

export Prmapping,prmapping,plotr!,linearr

# Define a basic prmapping
# # DXD: should I make explicit somewhere if we're dealing with
# regression or classification mappings?
mutable struct Prmapping
   type          # type of mapping (untrained/trained)
   fit           # fit function
   predict       # predict function
   data          # parameters of the model
   labels        # feature labels of the output 
end
# Next we define an untrained mapping:
# (How should we store the parameters of the function: leave it to the
# user?)
function prmapping(fit,predict)
   return Prmapping("untrained",fit,predict,nothing,nothing)
end
# The training step
function Base.:*(a::Prdataset,w::Prmapping)
   if w.type=="untrained"
      out = deepcopy(w)
      w.fit(out.data, a) # DXD better?
      out.type = "trained"
      return out
   elseif w.type=="trained"
      pred = w.predict(w.data,a)
      out = deepcopy(a)
      setfield!(out,:data,pred)
      return out
   end
end
# Plot a regression function
function plotr!(w::Prmapping)
   xl = xlims()
   z = collect(xl[1]:0.1:xl[2])
   out = Prdataset(z[:,:])*w
   plot!(z,+out)
end

#  ------- Example of a linear regression   -----------
#
# For each method we require a 'fit', a 'predict' function, and one call
# for an untrained mapping
function fitLS!(params, a)
   degree = params["degree"] # ungainly storage
   X = a.data.^(collect(0:degree)')
   params["weights"] = inv(X'*X)*X'*a.targets
   return params
end
function predictLS(params, a)
   degree = params["degree"] # ungainly storage
   w = params["weights"] # ungainly storage
   X = a.data.^(collect(0:degree)')
   return X*w
end
function linearr(degree=1)
   params = Dict{String,Any}("degree"=>degree)
   return Prmapping("untrained",fitLS!,predictLS,params,nothing)
end
"""
    w = linearr(a, degree=1)

Fit a linear least square regression on dataset `a`. If requested, with `degree` higher order terms of the data features can also be used.
"""
function linearr(a::Prdataset,degree=1)
   return a*linearr(degree)
end

