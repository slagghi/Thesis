function [normA] = normalise(A)
% This function normalises the input matrix
m=min(min(A));
M=max(max(A));
normA=A;
normA(:,:)=(A(:,:)-m)/(M-m);
end

