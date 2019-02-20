% This script extracts the feature vector from googlenet
% for each image, and saves it to a file
addpath('../Functions');
GN=googlenet;
filepath='~/Desktop/parsingDataset/RSICD_images/';
ds=readCaptions();

features=zeros(1000,1);
F=zeros(1,1,1000);
featureSet=zeros(1000,10921);
for i=1:10921
    filename=strcat(filepath,ds.images(i).filename);
    I=imread(filename);
    F=activations(GN,I,'loss3-classifier');
    % change vector dimension from (1x1x1000) to (1000x1)
    features(:)=F(1,1,:);
    featureSet(:,i)=features;
    % progress update
    if mod(i,50)==0
        progress=100*i/10921;
        fprintf('Computing features for image %d\t Progress:%.4f%%\n',i,progress);
    end
end