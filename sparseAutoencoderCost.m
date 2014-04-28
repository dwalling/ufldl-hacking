function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

assert(visibleSize == size(data, 1))
assert(2*hiddenSize*visibleSize + hiddenSize + visibleSize == size(theta, 1)); % check size

numSamples = size(data, 2);

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

z2 = W1*data + repmat(b1, 1, numSamples);
a2 = sigmoid(z2);

z3 = W2*a2 + repmat(b2, 1, numSamples);
a3 = sigmoid(z3);

sq_err = get_squared_error(a3, data);
weight_decay = 0.5 * lambda * (sum_sq_W(W1) + sum_sq_W(W2));
sparsity_penalty = beta * sparsityPenalty(a2, sparsityParam);

cost = sq_err + weight_decay + sparsity_penalty;

% derivative_of_sigmoid(zi) = ai * (1 - ai)

delta_layer_3 = -(data - a3) .* (a3 .* (1 - a3));
delta_layer_2 = (W2' * delta_layer_3 + ...
    beta * repmat(sparsityDerivative(a2, sparsityParam), 1, numSamples) ...
                    ) .* (a2 .* (1 - a2));

W2grad = (delta_layer_3 * a2'  ) / numSamples + lambda * W2;
b2grad = (delta_layer_3 * ones(numSamples, 1)) / numSamples;
W1grad = (delta_layer_2 * data') / numSamples + lambda * W1;
b1grad = (delta_layer_2 * ones(numSamples, 1)) / numSamples;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

function spd = sparsityDerivative(a, rho)
    rho_hat = mean(a, 2);
    spd = -(rho ./ rho_hat) + (1-rho) ./ (1-rho_hat);
end

function sp = sparsityPenalty(a, rho)
    rho_hat = mean(a, 2);
    sp = sum(rho * log(rho ./ rho_hat) + (1 - rho) * log((1 - rho) ./ (1 - rho_hat)));
end


%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function sqe = get_squared_error(h, y)
    err = h - y;
    sqe = sum(err .* err, 1); % sum of squared errors each example
    sqe = sum(sqe / 2) / size(y, 2); % 1/2 avg sq err
end

function ssw = sum_sq_W(W)
    ssw = sum(sum(W .* W, 1), 2);
end

