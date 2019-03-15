YTrain_firstCaption=cell(1,8734);
ctr=0;
for i=1:43670
    if mod(i,5)==1
        ctr=ctr+1;
        YTrain_firstCaption(ctr)=YTrain(i);
        fprintf("%d out of 43670\n",i);
    end
end