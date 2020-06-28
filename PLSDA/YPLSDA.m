function Y = YPLSDA(nb,ref)
% This function create a Y matrix for PLS-DA.
% Y cell take a value of 1 in the column of the group number, 
% for the other(s) it is 0.
% Input: 
%	nb: number of group
%   ref : reference vector
% Output:
%   Y: matrix for PLS-DA

[n, ~] =size(ref); % Size of the reference vector
% Initialisation
y=zeros(n,1); % y vector
Y=zeros(n,nb); % Y matrix
    for i=1:nb % loop for the group
        for j=1:n % loop for the sample
            if ref(j,:)==i % y take a value of 1 if the reference value 
                           % have the same value as the column number
                y(j,:)=1;
            else % otherwise y take a value of 0
                y(j,:)=0;
            end
        end
    Y(:,i)=y; % Concatenation of the different y vector into Y matrix
    end
end