function [out] = loadImage(imgName)
path="~/Desktop/parsingDataset/RSICD_images/";
out=imread(path+imgName);
end

