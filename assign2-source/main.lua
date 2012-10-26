--[[
Main file
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com>) @ New York University
Version 0.1, 10/10/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

This file contains sample of experiments.
--]]

-- Load required libraries and files
dofile("spambase.lua")
dofile("model.lua")
dofile("regularizer.lua")
dofile("trainer.lua")
dofile("kernel.lua")
dofile("crossvalid.lua")
dofile("xsvm.lua")
dofile("mult.lua")
dofile("mnist.lua")

-- An example of using xsvm
function main()
   -- 1. Load spambase dataset
   print("Initializing datasets...")
   -- local data_train, data_test = spambase:getDatasets(3000,1000)
   -- local data_train_kern_poly, data_test_kern_poly = spambase:getDatasets(3000,1000)
   local data_train_crossvalid, data_test_crossvalid = spambase:getDatasets(10,1000)

   -- 2. Initialize a dual SVM with linear kernel, and C = 0.05.
   print("Initializing a Polynomial kernel SVM with C = 0.05...")
   -- local model = xsvm.vectorized{kernel = kernLin(), C = 0.05}
   -- local model_kern_poly = xsvm.vectorized{kernel = kernPoly(1,3), C = 0.05}
   --print("Breakpoint1")

   -- 3. Train the kernel SVM
   print("Training the kernel SVM...")
   -- local error_train = model:train(data_train)
   -- local error_train_kern_poly = model_kern_poly:train(data_train_kern_poly)
   
   -- 4. Testing using the kernel SVM
   print("Testing the kernel SVM...")
   -- local error_test = model:test(data_test)
   -- local error_test_kern_poly = model_kern_poly:test(data_test_kern_poly)

   --Using Cross validation
   -- Q.2. a. 
   function do_cross_valid_q2_a()
       -- degree,C_var_formal_arg
     local best_cross_validation_error=10
     local best_C = 0
     local p = 0
     local z = 4
     local deg = 4
     for i = 0, deg-1 do
     p = 0
     for j = 2^-z, 2^z, 2 do
     p=p+1
     -- models[i*(2*z+1) + p]=mfunc((i+1),j)
     local degree = 0
     local C_var_formal_arg = 0

     degree = i+1
     C_var_formal_arg = j

     local function mfunc()
	   		return xsvm.vectorized{kernel = kernPoly(1,degree), C = C_var_formal_arg}
	 end

     -- current_models, current_errors_train, current_errors_test = {}
     local current_models, current_errors_train, current_errors_test = crossvalid(mfunc,10,data_train_crossvalid)


     -- Now our job is to find the avg cross validation error for our current degree and C based model.
     local avg_cross_validation_error = 0.0
     -- print("current_errors_test:size(1) is: "..current_errors_test:size(1))
     -- print("current_errors_test is: ")
     -- print(current_errors_test)
     for f = 1,current_errors_test:size(1) do
	     avg_cross_validation_error = (avg_cross_validation_error*(current_errors_test:size(1)-1)/current_errors_test:size(1)) + (current_errors_test[f]/current_errors_test:size(1))
--(avg_cross_validation_error*(current_errors_test:size()-1)/current_errors_test:size())
--+(current_errors_test[f]/current_errors_test:size())
     end

    if avg_cross_validation_error<best_cross_validation_error then
        best_cross_validation_error = avg_cross_validation_error
        best_C = C_var_formal_arg
    end
     end
    end
    print("Best Average Cross Validation error is: "..best_cross_validation_error)
    print("Best correseponding C is: "..best_C)
   end
   do_cross_valid_q2_a()
   

   -- 6. Print the result
   -- print("Train error = "..error_train.."; Testing error = "..error_test)
   -- print("Train error for Polynomial Kernel= "..error_train_kern_poly.."; Testing error for Polynomial Kernel = "..error_test_kern_poly)

end

main()
