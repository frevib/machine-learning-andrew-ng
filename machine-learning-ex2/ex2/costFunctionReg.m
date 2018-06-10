function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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



%%%%% ANSWER

hx = sigmoid(X * theta);
summed_error = (-y' * log(hx) - (1 - y)' * log(1 - hx)) ./ m;

regularized_theta = lambda * sum(theta(2:end) .^ 2) ./ m ./ 2;
regularized_summed_error = summed_error + regularized_theta;

regularization_term = (theta(2:end) * lambda ./ m);
regularization_term = [0; regularization_term];


grad = ((X' * (hx - y )) ./ m) + regularization_term;
J = sum(regularized_summed_error);

%%%%%%



% =============================================================

end
