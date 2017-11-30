function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

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

% add ones to X
X = [ones(m,1) X];

% for a single example: a2 has to be a column vector with outputs from each node: a2 = sigmoid(Theta1*x)
% for m examples: a2 has to be a matrix with a row for each example
A2 = sigmoid(X * Theta1');

% add the column of ones to a2
A2 = [ones(m,1) A2];
             
% a3 has to be an m * K matrix:
% each row contains the probabilities for every category for the corresponding example

A3 = sigmoid(A2 * Theta2');

% now grab predictions: for each row take the index of the maximum
[m p] = max(A3, [], 2);


% =========================================================================


end
