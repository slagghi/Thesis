addpath('../Functions');
dataset=readCaptions();

% possible networks:
% googlenet, alexnet, vgg16, vgg19
net=googlenet;

%cycle for training images
% load random image for starters
r=int64(rand*10921);
inImage=loadImage(r);
targetCaption=dataset.images(r).sentences(2).tokens;
