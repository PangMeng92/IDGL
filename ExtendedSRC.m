function [Miss_NUM1, Miss_NUM2, Y_f] = ExtendedSRC(trainset,genericset,traingnd,testset,testgnd,lambda,Distance_mark, WProj)
if ~exist('lambda','var')
    lambda = 0;
end

[DIM, Class_Train_NUM, Class_NUM]=size(trainset);
[DIM, Class_Test_NUM, Class_NUM]=size(testset);
[DIM, Class_Variance_NUM, Class_NUM]=size(genericset);
EClass_Train_NUM=Class_Train_NUM/Class_NUM;
Miss_NUM1=0;
Miss_NUM2=0;
% Normalize the training data 
for t=1:Class_NUM
   for s=1:Class_Train_NUM 
      X=trainset(:,s,t);  
      X=X/norm(X);
      trainset(:,s,t)=X;
   end
end
for t=1:Class_NUM
   for s=1:Class_Test_NUM 
      X=testset(:,s,t);  
      X=X/norm(X);
      testset(:,s,t)=X;
   end
end

for t=1:Class_NUM
   for s=1:Class_Variance_NUM 
      X=genericset(:,s,t);  
      X=X/norm(X);
      genericset(:,s,t)=X;
   end
end

[dim,trainnum] = size(trainset);
[dim,testnum] = size(testset);
[dim,genericnum] = size(genericset);
nClass = length(unique(traingnd));

Y_t=zeros(testnum,nClass);
Y_f=[];

for i = 1:testnum

    [label] = Extended_SRCDist(trainset,genericset,traingnd,testset(:,i),lambda,Distance_mark, WProj);
     
    Y_t(i,label(1))=1;
     if  ~ismember(testgnd(i),label(1))
         Miss_NUM1=Miss_NUM1+1;
    end
    if  ~ismember(testgnd(i),label)
          Miss_NUM2=Miss_NUM2+1;
    end
end

TempZ=eye(nClass);
Y_f=[TempZ;Y_t];