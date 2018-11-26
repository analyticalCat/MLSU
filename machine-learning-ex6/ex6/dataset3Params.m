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

Csigmaset = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
setLength = length(Csigmaset);

for i = 1:setLength
    Ctry = Csigmaset(i);
    for j = 1:setLength
        sigmatry = Csigmaset(j);
        model= svmTrain(X, y, Ctry, @(x1, x2) gaussianKernel(x1, x2, sigmatry)); %x1 and x2 are dummy parameters here
        predictions = svmPredict(model, Xval);
        tp = mean(double(predictions ~= yval));
        if (i == 1 && j == 1) || perror > tp
            perror = tp;
            C = Ctry;
            sigma = sigmatry;
        endif
    endfor
endfor






% =========================================================================

end
