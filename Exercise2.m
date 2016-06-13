%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Handwritten digits classification using Baysian classifier
% Principal Component Analysis & Maximum Likelihood Classifier
% 
% Input:    dmax (which is 60 in this exercise)
% Output:   a plot of classification errors (from d = 1 to dmax), optimal 
%           value of d and its classification error and the confusion matrix.
%
% Author: Jianyu Zhao
% Last revised: 13.06.2016
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars;

dmax = 10;

TestLb = loadMNISTLabels('t10k-labels.idx1-ubyte');
num = size(TestLb,1);

%% Change values to d to find the optimal one
ClsErr = zeros(dmax,1);
PreLb = zeros(num,dmax);
for l=1:dmax
    [ClsErr(l),PreLb(:,l)] = BaysianClassifier(l);
end

%% Visualization
figure;
plot(1:dmax,100*ClsErr,'LineWidth',1);
xlabel('d value'); ylabel({'Classification Error','(in percentage)'});
title('Classification Error Plot');

%% Optimal value of d
[mClsErr,dopt] = min(ClsErr);
PreOpt = PreLb(:,dopt);
%[ErrOpt,PreOpt] = BaysianClassifier(dopt);
ConMt = confusionmat(TestLb,PreOpt);
helperDisplayConfusionMatrix(ConMt);

