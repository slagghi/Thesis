load('../../ScenarioAnalysis/GNfeatures.mat')
load('YTrain_firstCaption.mat')
load('trainIndices.mat')
addpath('../../Functions')

% this code loads the image features for the train images
DS=readCaptions();
trainFeatures=[];
for i=1:8734
    trainIndex=trainIndices(i);
    features=featureSet(:,trainIndex);
    trainFeatures=[trainFeatures,features];
end
    