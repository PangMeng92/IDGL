function [label,dist] = Extended_SRCDist(trainset,genericset,traingnd,img,lambda,Distance_mark,WProj)
if ~exist('lambda','var')
    lambda = 0;
end
Labels = unique(traingnd);
nClass = length(Labels);
dist = 10^20;
label = -1;
TrainNum=length(traingnd);
Distance=zeros(nClass,1);
X=[];
for i = 1:nClass
%      i=66
    Xbi = trainset(:,traingnd==Labels(i));
    
    [kk,mm]=size(Xbi);
    distance_V=zeros(mm,1); 
    for t=1:mm
           
            V=img-Xbi(:,t);
            
             switch Distance_mark
                 case {'Euclidean', 'L2'}
                      distm=norm(V,2); % Euclid (L2) distance
                 case 'L1'
                      distm=norm(V,1); % L1 distance
                 case 'Cos'
                      distm=acos(Test'*Train/(norm(Test,2)*norm(Train,2)));     % cos distance
                 otherwise
                      distm=norm(V,2); % Default distance
              end

            distance_V(t)=distm;
    end
    [sorted,neighborhood] = sort(distance_V,'ascend' );
       
    Xi= Xbi(:,neighborhood(1:mm));
    X=[X,Xi];
end
X=[X,genericset];
[W_all, gamma_x, total_iter, total_time] = BPDN_homotopy_function(X, img, lambda, 200);

Num=size(X,2);
%% using residual
for i=1:nClass  
      idx=find(traingnd==Labels(i));
      Temp=zeros(Num,1);
      Temp(TrainNum+1:Num,:)=1;
      Temp(idx,:)=1;
      W=W_all.*Temp;
     imghat = X*W;
    tmpdist = norm(img-imghat,2);
    Distance(i)=Distance(i)+tmpdist;
end
   [sorted,neighborhood] = sort(Distance,'ascend' );
    label=neighborhood(1:5);
    
%% using coefficient
%    W_all = W_all ./ repmat(sqrt(sum(W_all .* W_all)),[size(W_all ,1) 1]); % unit norm 2
%    Distance=W_all(1:nClass);
%    [sorted,neighborhood] = sort(Distance,'descend' );
%     label=neighborhood(1:5);
