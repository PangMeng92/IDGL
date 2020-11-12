function [ F,Z,W,E, recErrRecord] = SSLRR(X,U,Y,lambda,beta)
addpath(genpath('.\YALL1_v1.3'));  
[dd nn] = size(X);
%%------------------------------------------------------------------------
normfX = norm(X,'fro');
tol1 = 1e-6;%threshold for the error in constraint  (1e-6)
tol2 = 1e-6;%threshold for the change in the solutions  (1e-6)
maxIter = 400;  % 400
max_miu = 10^6;
miu=0.1;  % 0.1
rho=1.1;  % 1.1
%----------------------Initializing variables------------
Z=zeros(nn,nn);
F=zeros(nn,size(Y,2));
E=zeros(dd,nn);
W=zeros(nn,nn);
Y1=zeros(dd,nn);
Y2=zeros(nn,nn);



% recErrRecord=[];
% relChgRecord=[];

recErrRecord=zeros(1,maxIter);

%--------------End of initializing variables----------------
iter=0;
while iter<maxIter
    Term1=0;
    Term2=0;
    Term3=0;
    Term4=0;
    iter=iter+1;
    Ek=E;
    Zk=Z;
    Wk=W;
    Fk=F;
    [Z,svp]= Update_Z( X,Z,Y1,Y2,miu,lambda,E,W);
    JJ=(W+W')/2;
    L=diag(sum(JJ,2))-JJ;
    F = Update_F(L,U,Y );
    E = Update_E(X,Z,Y1,miu,beta);
    W = Update_W2(F,Z,Y2,miu );
    %W = Update_W(F,Z,Y2,miu );
%    W=Z;
    %%
    recErr=norm(X-X*Z-E,'fro')/normfX;
    relChgZ=norm(Zk-Z,'fro')/normfX;
    relChgE=norm(Ek-E,'fro')/normfX;
    relChgW=norm(Wk-W,'fro')/normfX;
    relChgF=norm(Fk-F,'fro')/normfX;
    rel=[];
    rel=[rel relChgZ];rel=[rel relChgE];rel=[rel relChgW];rel=[rel relChgF];
    %%
    relChg=max(abs(rel));
%     recErrRecord=[recErrRecord, recErr];
%     relChgRecord=[relChgRecord, relChg];
    convergence=recErr<tol1&&relChg<tol2;
    Y1=Y1+miu*(X-X*Z-E);
    Y2=Y2+miu*(Z-W);
    %% Objective function
        for i=1:nn
            for j=1:nn
         Term1=Term1+sum((F(i,:)-F(j,:)).^2*Z(i,j));
            end
        end
        
        Term2=trace((F-Y)'*U*(F-Y));
        
        
        Term3=sum(svd(Z));
        
        for i=1:dd 
                Term4=Term4+sqrt(sum(E(i,:).^2));
        end
        
        Term5=miu/2*(norm(X-X*Z-E+Y1/miu, 'fro')+norm(Z-W+Y2/miu, 'fro'));
        
        Term6=1/(2*miu)*(norm(Y1, 'fro')+norm(Y2, 'fro'));
        
        recErrRecord(iter)=Term1+Term2+lambda*Term3+beta*Term4;
%    recErrRecord(iter)=recErr;    
    if miu*max(abs(rel))<tol2
        miu=min(max_miu,miu*rho);
    end
    
    if iter>2
        if convergence
            break
        end
    end
end

