function [sNr] = scenarioNr(s,sDict)
% This code read an image name and returns the scenario number

% remove extension, file number etc.
s=deleteTail(s);
binaryVect=sDict==s;
sNr=find(binaryVect==1);
end

