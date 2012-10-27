--[[
Multi-class classification using binary classifiers implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 09/24/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

In this file you have to implement multOneVsAll and multOneVsOne. As an example
, part of multOneVsAll is given. These functions accept a parameter mfunc,
which is a function. Upon calling mfunc, a trainable model is returned with
whom you can run model:train(dataset) to train and return training error.
model:g(x) should give a classification, and model:l(x,y) should give the loss
on sample x y.

Of course, you can implement everything in your own way and disregard the code
here. 
--]]

-- mfunc is a callable that will return a trainable object
-- The trainable object must have to protocols:
-- model:train(dataset) will train the object. Return values are discarded
-- model:l(x,y) will return the loss on sample x with label y (-1 or 1).
function multOneVsAll(mfunc)
   -- Create an one-vs-all trainer
   local mult = {}
   -- Transform the dataset for one versus all
   local function procOneVsAll(dataset)
      -- The data table consists of dataset:classes() datasets
      local data = {}
      -- Iterate through each dataset
      for i = 1, dataset:classes() do
	 -- Create this dataset, with size() method returning the same thing as dataset
	 data[i] = {size = dataset.size}
	 -- Modify the labels
	 for j = 1, dataset:size() do
	    -- Create entry
	    data[i][j] = {}
	    -- Copy the input
	    data[i][j][1] = dataset[j][1]
	    if dataset[j][2][1] == i then
	       -- The label same to this class i is set to 1
	       data[i][j][2] = torch.ones(1)
	    else
	       -- The label different from this class i is set to -1
	       data[i][j][2] = -torch.ones(1)
	    end
	 end
      end
      -- Return this set of datsets
      return data
   end
   -- Train models
   function mult:train(dataset)
      -- Define mult:classes
      mult.classes = dataset.classes
      -- Preprocess the data
      local data = procOneVsAll(dataset)
      -- Iterate through the number of classes
      for i = 1, dataset:classes() do
	 -- Create a model
	 mult[i] = mfunc()
	 -- Train the model
	 mult[i]:train(data[i])
      end
      -- Return the training error
      return mult:test(dataset)
   end
   -- Test on dataset
   function mult:test(dataset)
      -- Set returning testing error
      local error = 0
      -- Iterate through the number of classes
      for i = 1, dataset:size() do
	 -- Iterative error rate computation
	 if torch.sum(torch.ne(mult:g(dataset[i][1]), dataset[i][2])) == 0 then
	    error = error/i*(i-1)
	 else
	    error = error/i*(i-1) + 1/i
	 end
      end
      -- Return the testing error
      return error
   end
   -- The decision function
   function mult:g(x)
      -- Remove the following line and add your stuff
      -- print("You have to define this function by yourself!");
      -- Define mult:classes
      mult.classes = dataset.classes
      -- Preprocess the data
      local data = procOneVsAll(dataset)
      -- Iterate through the number of classes
      for i = 1, dataset:classes() do
	 -- Create a model
	 mult[i] = mfunc()
	 -- Train the model
	 mult[i]:train(data[i])
      end
    -- local largest_W_dot_X = -99999999
    -- local largest_w_dot_X_corresponding_i = -1
       local largest_f_X = -99999999
       local largest_f_X_corresponding_i = -1
    for i = 1, dataset:classes() do
      --if (torch.dot(mult[i].W,x)>largest_W_dot_X) then
        --     largest_W_dot_X = torch.dot(mult[i].W,x)
          --   largest_w_dot_X_corresponding_i = i
        if (mult[i]:f(x)>largest_f_X) then
               largest_f_X = mult[i]:f(x) 
               largest_f_X_corresponding_i = i              
    end
   end
   return i
   end
   -- Return this one-vs-all trainer
   return mult
end


-- mfunc is a callable that will return a trainable object
-- The trainable object must have to protocols:
-- model:train(dataset) will train the object. Return values are discarded
-- model:l(x,y) will return the loss on sample x with label y (-1 or 1).
-- model:g(x) will determine the label of a given x.
function multOneVsOne(mfunc)
   -- Remove the following line and add your stuff
   -- print("You have to define this function by yourself!");
-- Create an one-vs-one trainer
   local mult = {}
   -- Transform the dataset for one versus all
   local function procOneVsOne(dataset,class1,class2)
      -- The data table consists of data samples from dataset which have class1 or class2 as their classification.
      -- class1 is a particular class.
      -- class2 is a particular class.
      local data = {}
      local data_size = 0
      -- Iterate through each dataset
      --for i = 1, dataset:classes() do
	 -- Create this dataset, with size() method returning the same thing as dataset
     --	data[i] = {size = dataset.size}
     
	 for j = 1, dataset:size() do
	    -- Create entry
	    data[j] = {}
	    -- Copy the input
	    
	    if dataset[j][2][1] == class1 then
	       -- The label same to this class 1 is set to 1
           data_size = data_size + 1
           data[j][1] = dataset[j][1]
	       data[j][2] = torch.ones(1)
	    else
           if dataset[j][2][1] == class2 then
	       -- The label from this class 2 is set to -1
           data_size = data_size + 1
           data[j][1] = dataset[j][1]
	       data[j][2] = -torch.ones(1)
        end
	    end
	 end
      function data:size() return data_size end
      --end
      -- Return this set of datsets
      return data
   end
   -- Train models
   function mult:train(dataset)
      -- Define mult:classes
      mult.classes = dataset.classes
      local i_cntr = -1
      local j_cntr = -1
      local no_of_model = 0
      local data = {}

      for i_cntr = 1, mult.classes-1 do
      for j_cntr = i+1, mult.classes do
      -- Preprocess the data
      -- Make selection from data for mult.classes Combine 2 ie select each pair of 2 classes out of our total no. of classes.
       data[no_of_model] = procOneVsOne(dataset,i_cntr,j_cntr)
     -- for i = 1, dataset:classes() do
	 -- Create a model
     no_of_model = no_of_model + 1
	 mult[no_of_model] = mfunc()
	 -- Train the model
	 mult[no_of_model]:train(data[i])
      end
     end
      -- Return the training error
      return mult:test(dataset)
   end
   -- Test on dataset
   function mult:test(dataset)


      -- Set returning testing error
      local error = 0
      -- Iterate through the number of classes
      for i = 1, dataset:size() do
	 -- Iterative error rate computation
	 if torch.sum(torch.ne(mult:g(dataset[i][1]), dataset[i][2])) == 0 then
	    error = error/i*(i-1)
	 else
	    error = error/i*(i-1) + 1/i
	 end
      end
      -- Return the testing error
      return error
   end
   -- The decision function
   function mult:g(x)
      -- Remove the following line and add your stuff
      -- print("You have to define this function by yourself!");
      -- Define mult:classes
      mult.classes = dataset.classes
      predicted_class = torch.zeroes(mult.classes)
      local no_of_model = 0
      for i_cntr = 1, mult.classes-1 do
      for j_cntr = i+1, mult.classes do
      no_of_model = no_of_model + 1

      -- Preprocess the data
      local data = procOneVsOne(dataset)
      -- Iterate through the number of classes
      --for i = 1, dataset:classes() do
	 -- Create a model
	 mult[no_ofmodel] = mfunc()
	 -- Train the model
	 mult[i]:train(data,i_cntr,j_cntr)
     -- end
     if (multi[i]:g(x)[1]==1) then
     predicted_class[i_cntr] = predicted_class[i_cntr] + 1
     else
     predicted_class[j_cntr] = predicted_class[j_cntr] + 1
	 end

    local largest_class_value = -99999999
    local largest_corresponding_class_num = -1
    for i = 1, mult.classes do
      if (predicted[i] > largest_class_value) then
             largest_class_value = predicted[i]
             largest_corresponding_class_num = i              
      end
    end
   end
   end
return i   
end
-- Return this one-vs-one trainer
   return mult
end
