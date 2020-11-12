function E = Update_E(X,Z,Y1,miu,beta)
temp=X-X*Z+(Y1/miu);
ppa=(2*beta)/miu;
E=temp;
for i=1:size(temp,2)
    E(:,i)=solve_12(temp(:,i),ppa);
end

function [xpx]=solve_12(www,ppa)
nw=norm(www);
if nw>ppa
    xpx=(nw-ppa)*www/nw;
else
    xpx=zeros(length(www),1);
end

