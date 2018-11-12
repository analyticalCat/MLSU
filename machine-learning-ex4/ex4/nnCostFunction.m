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
Theta1_grad = zeros(size(Theta1)); %size 25 x 401
Theta2_grad = zeros(size(Theta2)); %size 10 x 26


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% convert y from digit labels 1,2,...0 

%Y=zeros(m,num_labels);
%I=eye(num_labels);

%for i=1:num_labels
%    Y(i,:) = I(y(i),:);
%endfor

%or 
Y=eye(num_labels)(y,:);

 % Add ones to the X data matrix
A1 = [ones(m,1), X];
Z2 = A1 * Theta1';
A2 = sigmoid(Z2);
A2 = [ones(m,1), A2];
Z3 = A2 * Theta2';
A3 = sigmoid(Z3);  %A3 = htheta(x) size m,num_labels

costs = -Y .* log(A3) - (1-Y) .* log(1-A3);

J_unreg=sum(sum(costs))/m;

Theta1b = Theta1(:,2:end);
Theta2b = Theta2(:,2:end);
reg = (sum(sum(Theta1b .^ 2, 2)) + sum(sum(Theta2b .^ 2)))/(2 * m) * lambda;

J=J_unreg + reg;



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


%for t=1:m
%    a1 = A1(t,:);
%    a2 = A2(t,:);
%    a3 = A3(t,:);
%    z2 = Z2(t,:);
%    z3 = Z3(t,:);
%    y3 = Y(t,:);

%    d3 = a3 - y3;
%    d2 = d3 * Theta2;
%    d2 = d2(2:end);
%    d2 = d2 .* sigmoidGradient(z2);
    
%    Theta2_grad = Theta2_grad + d3' * a2;
%    Theta1_grad = Theta1_grad + d2' * a1;
%endfor

%Theta1_grad = Theta1_grad/m;
%Theta2_grad = Theta2_grad/m;

D3 =A3 - Y;

D2=(D3 * Theta2)(:, 2:end);
D2 = D2 .* sigmoidGradient(Z2);


%delta1 should be 2 x 3 and delta2 should be 4 x 3
delta_1 = D2' * A1;
delta_2 = D3' * A2;
Theta2_grad = delta_2/m;
Theta1_grad = delta_1/m;





% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1c = Theta1(:,2:end);
Theta1c = [zeros(rows(Theta1c),1),Theta1c];
Theta2c = Theta2(:,2:end);
Theta2c = [zeros(rows(Theta2c),1),Theta2c];

Theta1_grad = Theta1_grad + lambda / m * Theta1c
Theta2_grad = Theta2_grad + lambda / m * Theta2c
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
