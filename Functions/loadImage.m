function [out] = loadImage(imgName)
% This function takes a filename as input and
% reads the corresponding image from the dataset
path="~/Desktop/parsingDataset/RSICD_images/";
out=imread(path+imgName);
end

