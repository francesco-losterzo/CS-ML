function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

values = [0.01 0.03 0.1 0.3 1 3 10 30];

% matrix to store errors for each pair of (C,sigma)
errors = zeros(size(values,2), size(values,2));

for cidx = 1:size(values,2)
  this_C = values(cidx);
  for sidx = 1:size(values,2)
    this_sigma = values(sidx);

    % train the model
    test_model= svmTrain(X, y, this_C, @(x1, x2) gaussianKernel(x1, x2, this_sigma));

    % take model predictions
    test_pred = svmPredict(test_model, Xval);

    % compute the error
    errors(cidx, sidx) = mean( double(test_pred ~= yval) );
  end
end

% retrieve C and sigma as the values giving the minimum error
[minval, row] = min(min(errors,[],2));
[minval, col] = min(min(errors,[],1));

C = values(row);
sigma = values(col);

% =========================================================================

end
