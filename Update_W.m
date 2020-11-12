function W = Update_W(F,Z,Y2,miu )
n=size(Z,1);
UUUU=zeros(n,n);
for i=1:n
    for j=1:n
        UUUU(i,j)=(1/2)*norm((F(i,:)-F(j,:)),2)^2;
    end
end
II=eye(size(Z,1));b=(Z+(Y2/miu));
W=[];
for k1=1:n
    opts.tol=5e-3;
    opts.rho=1/miu;
    %opts.rho=1;
    opts.nonneg=1;
    opts.weight=UUUU(:,k1);
    [xss,out]=yall1(II,b(:,k1),opts);
    W=[W xss];
end


% The follwoing is to solve a non-negative least squares paoblem

% function W = Update_W(F,Z,Y2,miu )
% n=size(Z,1);
% UUUU=zeros(n,n);
% for i=1:n
%     for j=1:n
%         UUUU(i,j)=(1/2)*norm((F(i,:)-F(j,:)),2)^2;
%     end
% end
% W = [];
% A = (Z+(Y2/miu));
% JKK = A-(UUUU/miu);
% IIO = eye(size(JKK,1));
% for jlo = 1:size(JKK,2)
%     xio = lsqnonneg(IIO,JKK(:,jlo));
%     W = [W xio];   
% end

