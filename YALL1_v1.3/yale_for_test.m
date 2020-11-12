%%%yale_for_test%%%%
clc,clear;
load USPS;
 for hk=1:length(gnd)
 fea(hk,:)= fea(hk,:)/norm(fea(hk,:));         %训练样本归一化
 end
A=fea;
A=orth(A);
A=A';
MM=A*A';

% A=orth(X);
b=A(:,1);
w=10*rand(9298,1);
n=9298;
%-----------------------------------------------
opts.tol = 5e-3; 

opts.rho = 5e-5;
opts.nonneg = 1;


opts.weights = w; 
% opts.nonorth = 1; 

%----------------------------------------------
tic; [x,Out] = yall1(A, b, opts); toc
%  relerr = norm(x-xs)/norm(xs);
fprintf('iter = %4i',Out.iter)
% plot(1:n,xs,'b-',1:n,x,'r:'); 
% legend('Original','Recovered','location','best')