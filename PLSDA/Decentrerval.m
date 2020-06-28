function [ Xc ] = Decentrerval( Xval, Xcal )
% Uncentered the set with the mean reference.

% Size of the set to uncentered
[nval, ~]=size(Xval);

% Uncentered
Xc=Xval+repmat(mean(Xcal),nval,1);

end

