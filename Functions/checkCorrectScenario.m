function [check] = checkCorrectScenario(filename,NNoutput,dict)
% This function checks whether the scenarioNN yealded the correct
% prediction

% convert input to char
filename=char(filename);

prediction=predictedScenario(dict,NNoutput);
if filename(1)=='0'
    % the image is a generic one
    filename='image_0.jpg';
end
groundTruth=deleteTail(filename);
check=strcmp(prediction,groundTruth);
end

