% This code gives a numeric index to the different scenarios
addpath('../Functions');
DS=readCaptions();
load('scenarioIndex.mat');

scenarioDict=strings(50,1);
sctr=0;
new=0;
for i=1:10921
    s=sIndex(i);
    % check if s is in the dictionary or if it is new
    if strncmp(s,'0',1)
        s="image";
    end
    new=1;
    for j=1:size(scenarioDict)
        if strcmp(s,scenarioDict(j))
        % if the string is found in the dictionary, the scenario isn't new
            new=0;
            break;
        end
    end
    if new
        % new scenario: update dictionary
        sctr=sctr+1;
        fprintf('New scenario: %s\n',s);
        scenarioDict(sctr)=s;
    end
end