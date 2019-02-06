function [captions] = readCaptions()
% This function reads the json file containing the captions
% and returns the decoded struct
fid=fopen("~/Desktop/parsingDataset/dataset_rsicd.json");
raw=fread(fid,inf);
str=char(raw');
captions=jsondecode(str);
end

