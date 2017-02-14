function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

hx =((X * theta) - y).^ 2;
A = sum(hx)/(2*m);
ntheta = theta;
ntheta(1) = 0;
B = (lambda * sum(ntheta .^ 2))/(2*m);
J = A+B;

% row wise multiplication
A1 =  sum(((X*theta) - y) .* X)';
B1 = lambda * ntheta;
grad = (A1 + B1)/m;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
