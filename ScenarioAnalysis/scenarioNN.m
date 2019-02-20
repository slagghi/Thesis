addpath('../Functions');
addpath('Functions');
DS=readCaptions();
load('scenarioDictionary.mat');
load('GNfeatures.mat')

% This code assembles the feature set and divides it in 
% training, test and evaluation set

labeledImages={};
labels=zeros(10000,1);

for i=1:10000
    filename=DS.images(i).filename;
    n=scenarioNr(filename,scenarioDict);
    if isempty(n)
        continue
    end
    labeledImages{i,1}=featureSet(:,i);
    labeledImages{i,2}=n;
    labeledImages{i,3}=filename;
    labeledImages{i,4}=DS.images(i).split;
    labels(i)=n;
end

% encode labels matrix in one-hot
labelsOneHot=zeros(30,10000);
for i=1:size(labels)
    n=labels(i);
    oneHot=zeros(30,1);
    oneHot(n)=1;
    labelsOneHot(:,i)=oneHot;
end

% divide in train, test, eval
trainSet=zeros(1000,7946); trainLabels=zeros(30,7946);
trainNames=strings(7946,1);
testSet=zeros(1000,1027); testLabels=zeros(30,1027);
testNames=strings(1027,1);
valSet=zeros(1000,1027); valLabels=zeros(30,1027);
valNames=strings(1027,1);

testCtr=0;trainCtr=0;valCtr=0;

for i=1:10000
    flag=labeledImages{i,4};
    if strcmp("train",flag)
        trainCtr=trainCtr+1;
        trainSet(:,trainCtr)=labeledImages{i,1};
        trainLabels(:,trainCtr)=labelsOneHot(:,i);
        trainNames(trainCtr)=DS.images(i).filename;
    elseif strcmp("val",flag)
        valCtr=valCtr+1;
        valSet(:,valCtr)=labeledImages{i,1};
        valLabels(:,valCtr)=labelsOneHot(:,i);
        valNames(valCtr)=DS.images(i).filename;
    elseif strcmp("test",flag)
        testCtr=testCtr+1;
        testSet(:,testCtr)=labeledImages{i,1};
        testLabels(:,testCtr)=labelsOneHot(:,i);
        testNames(testCtr)=DS.images(i).filename;
        testNames(testCtr)=DS.images(i).filename;
    end
    if mod(i,100)==0
        progress=100*i/10921;
        fprintf("Processing image %d, progress:%.4f%%\n",i,progress);
    end
end
trainValSet=[trainSet,valSet];
trainValLabels=[trainLabels,valLabels];