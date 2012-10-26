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

--[[
function mfunc(degree,C_var_formal_arg)
   return xsvm.vectorized{kernel = kernPoly(1,degree), C = C_var_formal_arg}
end
]]--

function crossvalid(mfunc, k, dataset)
   -- Remove the following line and add your stuff
   -- print("You have to define this function by yourself!");
   
   models = {}
   errors_train = torch.zeros(k)
   errors_test = torch.zeros(k)
   

--[[   --  Just creating our models[i] with different values of C and polynomial kernel degree.
   local p = 0
   local z = 4
   local deg = 4
   for i = 0, deg-1 do
     p = 0
     for j = 2^-z, 2^z, 2 do
     p=p+1
     models[i*(2*z+1) + p]=mfunc((i+1),j)
     end 
]]--
  
 -- end
       
  -- for a = 1,models:size()  
   for i = 1, k do
     print("Hey: "..i)
    local cv_train_dataset = {}
	local cv_test_dataset = {}

    function cv_train_dataset:size() return ((k-1)*dataset:size())/k end
    function cv_test_dataset:size() return dataset:size()/k end

    -- function cv_train_dataset:size() return ((k)*dataset:size())/k+1 end
    -- function cv_test_dataset:size() return dataset:size()/k+1 end    

    models[i] = mfunc()

    --print("Crossvalid.lua debug pt1: ")
    --print(dataset[1])
    local cv_train_dataset_counter = 1
    local cv_test_dataset_counter = 1

    -- local size_of_cv_test_dataset = 
    if i~=1 then
    for q = 1,((i-1)/k)*dataset:size() do  
      -- Cloning data instead of referencing, so that the datset can be split multiple times
     --[[ print("11 i: "..i)
      print("11 (i-1)/k: "..i-1/k)
	  print("11 dataset_size: "..dataset:size())
      print("11: "..cv_train_dataset_counter) ]] --
      cv_train_dataset[cv_train_dataset_counter] = {dataset[q][1]:clone(), dataset[q][2]:clone()}
      cv_train_dataset_counter=cv_train_dataset_counter+1
      end
    end
    for r = (i/k)*dataset:size()+1,(k/k)*dataset:size() do
      -- Cloning data instead of referencing, so that the datset can be split multiple times
     --print("22: "..cv_train_dataset_counter)
     cv_train_dataset[cv_train_dataset_counter] = {dataset[r][1]:clone(), dataset[r][2]:clone()}
     cv_train_dataset_counter=cv_train_dataset_counter+1
    end
   -- iterate over rows
   for w = ((i-1)/k)*dataset:size()+1,(i/k)*dataset:size() do
      -- Cloning data instead of referencing
      --print("33: "..cv_test_dataset_counter)
      cv_test_dataset[cv_test_dataset_counter] = {dataset[w][1]:clone(), dataset[w][2]:clone()}
      cv_test_dataset_counter=cv_test_dataset_counter+1
   end
   -- print("------------------------------------------")
   -- print(cv_train_dataset)
   -- print("------------------------------------------")
   	errors_train[i] = models[i]:train(cv_train_dataset)
   	errors_test[i] = models[i]:test(cv_test_dataset)
   end
   
return models,errors_train,errors_test
  -- end
end
