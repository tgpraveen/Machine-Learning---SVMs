--[[
Cross-validation implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 10/08/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

In this file you should implement a cross-validation mechanism that will be
used to perform multiple training on a model. You can implement it in anyway
you want, but the following is how I implemented it:

crossvalid(mfunc, k, dataset)
in which mfunc is a callable that creates a model with train and test methods,
k is the number of folds, dataset is the dataset to be used. dataset must
be randomized (our spambase and mnist datasets automatically do so when calling
getDatasets()).

The returned parameters are models, errors_train and erros_test.

models is atable in which models[i] should store the ith model returned by
calling mfunc.

errors_train is a torch tensor of size k indicating training errors returned by
model[i]:train(dataset).

errors_test is a torch tensor of size k indicating testing errors returned
by model[i]:test(dataset) after training it.

--]]

-- How I implemented cross validation:
-- k: number of folds;
-- mfunc: a callable that creates a model with train() and test() methods.
-- model.train(dataset) should train a model and return the training error
-- model.test(dataset) should return the testing error
-- The return list is: models, errors_train, errors_test where
-- models is a table in which models[k] indicates the kth one returned by mfunc
-- errors_train is a vector of size k indicating the training errors
-- errors_test is a vector of size k indicating the cross-validation errors

function mfunc(degree,C_var_formal_arg)
   return xsvm.vectorized{kernel = kernPoly(1,degree), C = C_var_formal_arg}
   end

function crossvalid(mfunc, k, dataset)
   -- Remove the following line and add your stuff
   -- print("You have to define this function by yourself!");
   
   local model = {}
   errors_train = torch.ones(k)
   errors_test = torch.ones(k)
   
   local z = 4
   local deg = 4
   models = torch.ones(z*deg)
   
   local p = 0

   --  Just creating our models[i] with different values of C and polynomial kernel degree.
   for i = 0, deg-1 do
     p = 0
     for j = 2^-z, 2^z, 2 do
     p=p+1
     models[i*(2*z+1) + p]=mfunc((i+1),j)
     end
   end

   for i = 1, k do
   	
   	errors_train[i] = models[i]:train(dataset)
   	errors_test[i] = models[i]:test(dataset)
   end

end
