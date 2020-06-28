function [ Xc ] = Centrerval( Xval, Xcal )
% Function that centered the validation set on the calibration set.

% Size of the validation set
[nval, ~]=size(Xval);

% Centering
if size(Xcal,1)>1
    Xc=Xval-repmat(mean(Xcal),nval,1);
else
    Xc=Xval-repmat(Xcal,nval,1);
end

end