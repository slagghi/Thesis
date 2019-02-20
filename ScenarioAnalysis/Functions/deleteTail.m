function s=deleteTail(s)
undVector=s=='_';
%count how many zeros are before the 1
zeroCtr=0;
for si=1:length(s)
    if undVector(si)==0
        zeroCtr=zeroCtr+1;
    else
        break
    end
end
s=s(1:zeroCtr);
end