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
                 hidden_layer_size, (input_layer_size + 1)); % Theta1 is a 25 x 401 matrix

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % Theta2 is a 10 x 26 matrix

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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

% convert y from a 5000 x 1 vector to a 5000 x 10 matrix Y
Y = zeros(m,num_labels); 
for index = 1:m
    Y(index, y(index)) = 1; % Y is a 5000 x 10 matrix
end

% X is 5000 x 400
% Theta1 is 25 x 401 matrix
% Theta2 is 10 x 26 matrix

X_b = [ones(m,1) X]; % X incl bias unit. 5000 x 401 matrix

% hidden layer output activations
z_2 = Theta1 * X_b'; % 25 x 5000 matrix
a_2 = sigmoid(z_2); % 25 x 5000 matrix 

% ouput layer output activations
z_3 = Theta2 * [ones(m,1) (a_2)']';
a_3 = sigmoid(Theta2 * [ones(m,1) (a_2)']'); % 10 x 5000  matrix
h = a_3; % h is 10 x 5000  matrix

J = (1/m) * sum( sum( ( -Y'.*log(h) - (1-Y)'.*log(1 - h) ), 1 ), 2);

% backpropagation algorithm

% calculate error at each output unit
d3 = (h - Y'); % 10 x 5000  matrix

% calculate error at each hidden layer unit. The first term is the gradient of the Cost with a_2. The second term is the gradient of a_2 with z_2.
d2 = (Theta2(:,2:end)' * d3).*sigmoidGradient(z_2); % 25 x 5000 matrix

% knowing the errors at each unit and level, calculate the gradient of the cost function with Theta2. 
Delta_2 = (d3*[ones(m,1) (a_2)']); % 10 x 26 matrix
% knowing the errors at each unit and level, calculate the gradient of the cost function with Theta1.
Delta_1 = d2*(X_b); % 25 x 401

Theta2_grad =  Delta_2/m;

Theta1_grad =  Delta_1/m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
