function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% X.shape = [m, n], y.shape = [m, 1], theta.shape = [n, 1]
% y_pred = sigmoid(X*theta), y_pred.shape = [m, 1]
y_pred = sigmoid(X*theta);
J = -1.0 / m * sum( y.*log(y_pred) + (1-y).*log(1-y_pred) );
grad = 1.0 / m .* (X' * (y_pred - y));





% =============================================================

end
