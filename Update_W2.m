function W = Update_W2(F,Z,Y2,miu )
n=size(Z,1);
UUUU=zeros(n,n);
for i=1:n
    for j=1:n
        UUUU(i,j)=(1/2)*norm((F(i,:)-F(j,:)),2)^2;
    end
end

b=(Z+(Y2/miu));

W=[];
for k1=1:n
    xss=b(:,k1)-(1./miu)*UUUU(:,k1);
    W=[W xss];
end

%W = W ./ repmat(sqrt(sum(W .* W )),[size(W ,1),1]);