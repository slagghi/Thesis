addpath('../Functions');
DS=readCaptions();
load('scenarioDictionary.mat');
load('GNfeatures.mat')

labeledImages={};
labels=zeros(10921,1);

for i=1:10921
    filename=DS.images(i).filename;
    n=scenarioNr(filename,scenarioDict);
    labeledImages{i,1}=featureSet(:,i);
    labeledImages{i,2}=n;
    labeledImages{i,3}=filename;
    labeledImages{i,4}=DS.images(i).split;
    if isempty(n)
        n=31;
    end
    labels(i)=n;
end

% encode labels matrix in one-hot
labelsOneHot=zeros(31,10921);
for i=1:size(labels)
    n=labels(i);
    oneHot=zeros(31,1);
    oneHot(n)=1;
    labelsOneHot(:,i)=oneHot;
end
