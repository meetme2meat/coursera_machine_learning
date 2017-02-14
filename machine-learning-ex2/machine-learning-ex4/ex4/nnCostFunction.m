function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
% a1 is also called X 
X = [ones(m, 1) X];

z2 = X * Theta1';
a2 = sigmoid(z2);

m2 = size(a2,1);
a2 = [ones(m2, 1) a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

yval=([1:num_labels] == y);

% Need to do a element wise muliptlication
J = -yval .* log(a3) - (1-yval) .* log(1-a3);
J = sum(sum(J,2));
J = J/m;

% ignore Theta0
T1 = Theta1;
T1(:,1) = zeros(size(Theta1)(1),1);

T2 = Theta2;
T2(:,1) = zeros(size(Theta2)(1),1);

T1 = T1 .^ 2;
T2 = T2 .^ 2;

regularization = sum(sum(T1,2)) + sum(sum(T2,2));

J = J + ((lambda * regularization) / (2 *m));



% regularization = (lambda/(2*m)) * sum(sum(Theta1 .^ 2) + sum(Theta2 .^ 2))
% J = J + regularization;
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% a2 = sigmoid(X * Theta1');

% m2 = size(a2,1);

% a2 = [ones(m2, 1) a2];

% prediction = sigmoid(a2 * Theta2');

% J = y .* log(prediction) + (1-y) .* log(1-prediction);

% J = -J / m;

% J = sum(J);
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

delta_3 = a3 - yval;

theta_2 = Theta2(:,2:end);

indelta_2 = (Theta2(:,2:end)' * delta_3');
delta_2 = indelta_2' .* sigmoidGradient(z2); 

% delta_2 = delta_2(2:end);
a1 = X;
DELTA_1 =  delta_2' * a1;
DELTA_2 =  delta_3' * a2;

regularized_grad1 = Theta1;
regularized_grad2 = Theta2;

n1 = size(regularized_grad1)(1);
regularized_grad1(:,1) = zeros(n1,1);

n2 = size(regularized_grad2)(1);
regularized_grad2(:,1) = zeros(n2,1);

Theta1_grad = (DELTA_1/m) + (lambda * regularized_grad1)/m;
Theta2_grad = (DELTA_2/m) + (lambda * regularized_grad2)/m;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

