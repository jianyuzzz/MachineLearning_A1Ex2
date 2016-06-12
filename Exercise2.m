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
% Last revised: 12.06.2016
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dmax = 60;
d = 15;

% load the data
images_org = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
testim = loadMNISTImages('t10k-images.idx3-ubyte');
testlb = loadMNISTLabels('t10k-labels.idx1-ubyte');

% simplify the data for testing(REMEMBER TO REMOVE THIS PART)
images_org = images_org(:,1:6000);
labels = labels(1:6000);
testim = testim(:,1:1000);
testlb = testlb(1:1000);

n = size(labels);
m = size(testim,2);

% test: visualize the ith image in the data set
%i = 20000;
%imshow(reshape(images_org(:,i),28,28));

%% Principal Components Analysis (PCA)
muim = mean(images_org');% mean vec of the images from all training images
images_norm = images_org' - repmat(muim,[n,1]);
covim = cov(images_norm);% covariance matrix of the zero-mean data
[eigv,lambda] = eig(covim);% the columns of eigv are the eigenvectors
basis = eigv(:,1:d)';
images = (basis*images_org)';% !!multiply the data before normalization

%% Multivariate gaussian distribution
% image sorting
img_cl = cell(1,10);
for i=1:n
    switch labels(i)
        case 1
            img_cl{1} = [img_cl{1};images(i,:)];
        case 2
            img_cl{2} = [img_cl{2};images(i,:)];
        case 3
            img_cl{3} = [img_cl{3};images(i,:)];
        case 4
            img_cl{4} = [img_cl{4};images(i,:)];
        case 5
            img_cl{5} = [img_cl{5};images(i,:)];
        case 6
            img_cl{6} = [img_cl{6};images(i,:)];
        case 7
            img_cl{7} = [img_cl{7};images(i,:)];
        case 8
            img_cl{8} = [img_cl{8};images(i,:)];
        case 9
            img_cl{9} = [img_cl{9};images(i,:)];
        case 0
            img_cl{10} = [img_cl{10};images(i,:)];
    end
end

% calculate the mean and variance of each class
mucl = zeros(10,d);% mean vector of each class
covcl = zeros(10,d,d);% covariance of each class
for j=1:10
    mucl(j,:) = mean(img_cl{j});
    covcl(j,:,:) = cov(img_cl{j});
end

%% For a novel test input
mutest = mean(testim');
testim = testim' - repmat(mutest,[m,1]); 
testin = basis*testim';

%% Likelihood calculation and class determination
likelh = zeros(m,10);
label_pre = zeros(m,1);% multivariate normal probability density fct
for k=1:m
    for cls=1:10
        likelh(k,cls) = mvnpdf(testin(:,k),mucl(cls,:)',squeeze(covcl(cls,:,:)));
    end
    [mlikelh,label_pre(k)] = max(squeeze(likelh(k,:)));
end