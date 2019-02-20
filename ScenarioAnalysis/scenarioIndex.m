% This code divides the images by scenarios and creates a "scenario index"
addpath("../Functions");
addpath("Functions");
DS=readCaptions();
sIndex=strings(10921,1);
for i=1:10921
    if mod(i,250)==0
        fprintf('image %d\n',i);
    end
    s=DS.images(i).filename;
    s=deleteTail(s);
    sIndex(i)=s;
end