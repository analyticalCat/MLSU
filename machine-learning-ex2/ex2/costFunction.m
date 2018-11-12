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

%calculating unregularized J
Xtheta = sigmoid(X * theta);
lXt = log(Xtheta);
lXt2 = log(1-Xtheta);
ylXt = y .* lXt;
ylTx2 = (1-y) .* lXt2;
Jm = -ylXt - ylTx2;
J=sum(Jm)/m;
                 
%calculating unregularized gradient
hx_y = Xtheta - y;
grad = X' * (hx_y) / m;

%Calculating regularized J and gradient
%theta1 = theta;
%theta1(1)=0;
%regfactorJ=lambda/(2*m)*(sum(theta1.^2));
%J=J+regfactorJ
%regfactorgrad= lambda/m*theta1;
%grad=grad + regfactorgrad






% =============================================================

end
