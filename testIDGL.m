clc;
clear all;


load AR_TrainRandom_50percent_SSPP50b_5.mat

Dim=size(Iv,1); 
 Num_PerClass=size(Iv,2);
ClassNum=size(Iv,3);


%%  Generate X
elltrain=1; % training sample;
elltest=6; % test sampe;
elltestnew=6;  % new test sample

TotalNum=(elltrain+elltest)*ClassNum;
      U=zeros(TotalNum,TotalNum);

TotalTrain = ClassNum * elltrain;
TotalTest = ClassNum * elltest;
TotalTestnew = ClassNum * elltestnew;


% fea: Rows of vectors of data points. Each row is x_i
Itrain = zeros(Dim,TotalTrain);
 for i=1:ClassNum
    for j=1:elltrain
         Itrain(:,j+(i-1)*elltrain) = Iv(:,j,i);  
    end
 end

Itrain1=Itrain;



%% Inductive Setting
%RandomSet=randperm(Num_PerClass-1);
%RandomSet=[1:Num_PerClass-1];
RandomSet=[6     2     7    10     8     1     5     9    11    12     3    4];  % AR

testset=RandomSet(1:elltest);
testsetnew=RandomSet(elltest+1:Num_PerClass-1);


Itest = zeros(Dim,TotalTest);
for i=1:ClassNum
    for j=1:elltest
        Itest(:,j+(i-1)*elltest) = Iv(:,elltrain+testset(j),i);
    end
end

Itestnew = zeros(Dim,TotalTestnew);
for i=1:ClassNum
    for j=1:elltestnew
        Itestnew(:,j+(i-1)*elltestnew) = Iv(:,elltrain+testsetnew(j),i);
    end
end

X_original=[Itrain, Itest];

%% Skinny SVD
%rank_X_original=rank(X_original);
rank_X_original=200;
[w,d,h]=svds(X_original,rank_X_original);

X=d*h';


%% Ground Truth label
Y_g=zeros(TotalNum,ClassNum);

for j=1:ClassNum
    Y_g(j,j)=1;
end

for k=1:ClassNum
    Y_g(ClassNum+1+elltest*(k-1):ClassNum+elltest+elltest*(k-1),k)=1;    
end


trainlabels=constructlabel(ClassNum,elltrain);
testlabels=constructlabel(ClassNum,elltest);
testlabelsnew=constructlabel(ClassNum,elltestnew);

%% AR: lambda1=15, gamma=1, beta=2, Maxiteration=400; E-YaleB: lambda1=15, gamma=1, beta=2, Maxiteration=400;
%% CAS-PEAL: lambda1=15, gamma=1, beta=2, Maxiteration=400; FERET: lambda1=15, gamma=1, beta=2, Maxiteration=400;  
%% Multi-PIE: lambda1=15, gamma=1, beta=2, Maxiteration=400;
%% LFW-LightendCNN: lambda1=15, gamma=1, beta=0.5

lambda1=15;
gamma=1;
beta=2;

%% Initialize U
for i=1:TotalNum
    U(i,i)=lambda1;
end

%% Initialize Y
Y=zeros(TotalNum,ClassNum);


for j=1:ClassNum
    Y(j,j)=1;
end

for T=1:4

fprintf('The iteration time is %d \n', T);    

[F,Z,W,E,recErrRecord] = SSLRR(X,U,Y,gamma,beta);


Y_New=zeros(TotalNum,ClassNum);

for jj=1:TotalNum
    [maxr,index]=max(F(jj,:));
    Y_New(jj,index)=1;
end


CorrectNum=0;

%% Prototype Leanring

P=[];
for ii=1:TotalTrain
    idx=find(Y_New(:,ii)==1);
    P_Temp=X(:,idx)*Z(idx,ii);
    P=[P,P_Temp];
end

P=w*P;
P = P ./ repmat(sqrt(sum(P .* P )),[size(P ,1),1]); % unit norm 2

MIP=P;
indC=[1:ClassNum];
%       Itrain(:,indC)=MIP(:,indC);
Itrain=MIP;


%% P+V model recognition
load AR_Variance_Generic_LRR50.mat

par.dim = Dim;
par.tr_num = TotalTrain;  % The number of training samples
par.tt_num = TotalTest;   % The number of testing samples

r = 200;  %  r is number of basis vectorsg

Igeneric=Itrain_Variance;

% Itrain = Itrain ./ repmat(sqrt(sum(Itrain .* Itrain )),[size(Itrain ,1) 1]); % unit norm 2
% Itest = Itest ./ repmat(sqrt(sum(Itest .* Itest)),[size(Itest ,1) 1]); % unit norm 2
% Igeneric = Igeneric ./ repmat(sqrt(sum(Igeneric .* Igeneric)),[size(Igeneric ,1) 1]); % unit norm 2
%Itestnew = Itestnew ./ repmat(sqrt(sum(Itestnew .* Itestnew)),[size(Itest ,1) 1]); % unit norm 2


Itrain = Itrain';
Itest = Itest';
Igeneric=Igeneric';
%Itestnew=Itestnew';


%% PCA
  %
    options = [];
    I=[];
    I=[Itrain;Igeneric];
    options.ReducedDim = r;
  [WProj, eigvalue] = PCA(I, options);
  %}     

FeaTrain = Itrain * WProj;
FeaTrain = FeaTrain';
FeaTest = Itest * WProj;
FeaTest = FeaTest';
FeaGeneric=Igeneric*WProj;
FeaGeneric=FeaGeneric'; 


lambda=0.001;
Distance_mark='L2';

[Miss_NUMESRC1, Miss_NUMESRC2, Y_f] =  ExtendedSRC(FeaTrain,FeaGeneric, trainlabels,FeaTest,testlabels,lambda,Distance_mark, WProj);
Recognition_rateExtendedSRC_top1(T)=(par.tt_num-Miss_NUMESRC1)/par.tt_num
Recognition_rateExtendedSRC_top5(T)=(par.tt_num-Miss_NUMESRC2)/par.tt_num

%Y=Y_New.*Y_f;
Y=Y_f;
beta=beta./1.2;

Itest = Itest';
%Itrain=Itrain';
end


%% Recognition of new test sample based on learned P + learned V model
par.ttnew_num = TotalTestnew;   % The number of testing samples
FeaTestnew = WProj'*Itestnew;
[Miss_NUM1, Miss_NUM5] =  ExtendedSRC(FeaTrain,FeaGeneric, trainlabels,FeaTestnew,testlabelsnew,lambda,Distance_mark, WProj);
Recognition_rate_top1=(par.ttnew_num-Miss_NUM1)/par.ttnew_num;
Recognition_rate_top5=(par.ttnew_num-Miss_NUM5)/par.ttnew_num;

fprintf('Top-1 recognition rate for new testing samples %f \n', Recognition_rate_top1); 
fprintf('Top-5 recognition rate for new testing samples %f \n', Recognition_rate_top5); 
