function[ Z,svp] = Update_Z( X,Z,Y1,Y2,miu,lambda,E,W )
temp1=X-X*Z-E+(Y1/miu);
temp2=Z-W+(Y2/miu);
norm2X = norm(X,2);
sq_X= norm2X*norm2X*1.02;
% sq_X= norm2X*norm2X*2;
temp3=Z-(1/sq_X)*(-X'*temp1+temp2);

[Uu,sigma_Z,Vv]=svd(temp3,'econ');
sigma_Z=diag(sigma_Z);
svp=length(find(sigma_Z>(1/((sq_X*miu)/lambda))));
if svp>1
    sigma_Z=sigma_Z(1:svp)-1/((sq_X*miu)/lambda);
else
    svp=1;
    sigma_Z=0;
end
Z=Uu(:,1:svp)*diag(sigma_Z)*Vv(:,1:svp)';




