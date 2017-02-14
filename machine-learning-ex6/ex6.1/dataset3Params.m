function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
% parameters = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% for i=parameters,
%   for j=parameters,
%     C = i;
%     sigma = j;
% % ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%               will return the predictions on the cross validation set.
%   
%   
% f = [];
% for i=1:size(X);
%   x1 = X(i,:);
%   feature = size(x1)(end);
%   index = 1;
%   for l = x1;
%     x2 = zeros(1,feature);
%     x2(1,index) = l;
%     f(i,index) = gaussianKernel(x1,x2,sigma);
%     index += 1;
%   end;
% end; 
%     % model = svmTrain()  
%     predictions = svmPredict(model, Xval);
%     % mean(double(predictions ~= yval))
%   end;
% end;



%  Note: You can compute the prediction error using 





% =========================================================================

end
