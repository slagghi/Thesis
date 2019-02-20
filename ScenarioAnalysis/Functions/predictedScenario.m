function [output] = predictedScenario(dict,v)
% This function reads the scenarioNN output and returns
% the predicted class
M=max(v);
index=find(v==M);
output=dict(index);
end

