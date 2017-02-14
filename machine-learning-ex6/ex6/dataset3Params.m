function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

variables = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
%variables = [0.1, 0.3, 1]
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
end_index = length(variables) ^ 2;
error_val = zeros(end_index,3);
index = 1;
for C=variables,
  for sigma=variables,
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    %predict using Xval
    index
    predictions = svmPredict(model,Xval);
    err = mean(double(predictions ~= yval));
    error_val(index,:) = [err C sigma]; 
    index +=1
  end;
end;

best_fit = min(error_val(:,1));

indexes = find(error_val(:,1) == best_fit);
error_val
error_val(indexes,:)
C = error_val(indexes,2:end)(1);
sigma = error_val(3,2:end)(2);


% =========================================================================

end
