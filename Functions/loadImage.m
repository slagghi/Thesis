function [out] = loadImage(id)
% multimodal function for reading dataset images
% reads image from either image id or filename
path="~/Desktop/parsingDataset/RSICD_images/";
%% If the identifier is a number, it is the file id in the dataset
if isa(id,'double') || isa(id,'int64') || isa(id,'uint8')
    caps=readCaptions();
    name=caps.images(id).filename;
    out=imread(path+name);
end
%% If the identifier is a string, it is the image filename
if isa(id,'string') || isa(id,'char')
    out=imread(path+id);
end
end

