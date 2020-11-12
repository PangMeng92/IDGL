function F = Update_F(L,U,Y )
F=pinv(L+U+eye(size(L,1))*0.01)*U*Y;

% F=F';
% F = F ./ repmat(sqrt(sum(F .* F )),[size(F ,1),1]); % unit norm 2
% F=F';


% [Q,R] = qr(A);
% InvR =  inv(R'*R)*R';
% qrInvA =InvR*Q';