function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m, 1) X];

% the layer 2 (also the first hidden layer because layer 1 is input layer)
% layer 3 is the output unit
%z2 = X * Theta1'
% sigmoid(z2)
a2 = sigmoid(X * Theta1');

m2 = size(a2,1);
% add bias unit

a2 = [ones(m2, 1) a2];
%z3 = a2 * theta2'
% a3 = sigmoid(z3) # a3 is output unit


a3 = sigmoid(a2 * Theta2');

% one vs all hence we have to find the max value of Job to predict the number
maxsigX = max(a3,[],2);
% find the index 
for i=1:m,
  p(i) = find(a3(i,:) == maxsigX(i));
end
% =========================================================================


end
