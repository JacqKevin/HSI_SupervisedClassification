function pred=predPLSDA(Scal, Sval, ref_cal, ref_val, group, model, para, fig)
% Creation of a model with PLS-DA algorithm. The model is calculate with a
% calibration set, after the validation set is predict. 
% To verify the quality of the model, a confusion matrix and some
% coefficients are define.

% Input:    Scal,Sval: calibration and validation spectra
%           ref_cal, ref_val: calibration and validation references
%           group: number of group
%           model: indicate if you create a model (0) or if you use a
%                   existant model (1)
%           para:  if model=1, you have to define a vector with the
%                   parameters of the model 
%           fig:    number of the figure if the user want to create
%                   graphical output 

% Output:   pred structure:
%               T,P,U,Q,B,W: parameters of the model
%               Ypred: validation prediction
%               Yval: validation references
%               Cf: confusion matrix
%               Precision, sensibility, specificity, R²: model performance

% Centering of the spectra
Sc_cal=Centrer(Scal);
Sc_val=Centrerval(Sval,Scal);

% Definition of the Y PLS-DA
Y_cal = YPLSDA(group,ref_cal);
Y_val = YPLSDA(group,ref_val);

% Saving the references values
Y_calb=Y_cal;
Y_valb=Y_val;

% Centering of the calibration references
Y_cal=Centrer(Y_cal);

if model==1 % If a model is already created
    T=para.PLSDA.T;
    P=para.PLSDA.P;
    Q=para.PLSDA.Q;
    B=para.PLSDA.B;
    W=para.PLSDA.W;
    Wstar=para.PLSDA.Wstar;
    R2X=para.PLSDA.R2X;
    R2Y=para.PLSDA.R2Y;
else % Creation of a model
    [B,Wstar,T,P,Q,W,R2X,R2Y]=plsnipals(Sc_cal,Y_cal);
    R2X=1-R2X;
    R2Y=1-R2Y;

end

% CALIBRATION

Y0 = Sc_cal*Wstar*Q; % Calibration prediction
Y0 = Decentrerval( Y0, Y_calb ); % Uncentering of the prediction

Max=max(Y0,[],2); % Maximum for all the sample prediction

% If the value of the cell is the maximum of the raw, the sample is in this
% group (1), otherwise it is not (0)
[e, r]=size(Y0);
Y0p=zeros(e,r);
for i=1:e
    for j=1:r
        if Y0(i,j)==Max(i,1)
            Y0p(i,j)=1;
        else
            Y0p(i,j)=0;
        end
    end
end

% Confusion matrix
a=round(sum(Y_calb))';
Cf0=abs(Y0p'*Y_calb);
[n, m]=size(Cf0);
for i=1:n
    for j=1:m
        if Cf0(i,j)<0.0001
            Cf0(i,j)=0;
        end
        Cf0(i,j)=round(Cf0(i,j));
    end
end

err0=zeros(group,1);
for i=1:group
   err0(i,:)=a(i,:)-Cf0(i,i);
end

error0=zeros(group,1);
for i=1:group
    if a(i,1)==0
        error0(i,1)=0;
    else
        error0(i,1)=err0(i,1)/a(i,1)*100;
    end
end

[n, ~]=size(Cf0);
z=1:n;
Eff1=zeros(1,n);Eff2=zeros(1,n);
for i=1:n
    if i>1&&i<n
        z1=[z(1:(i-1)) z((i+1):n)];
    else if i==1
           z1=z((i+1):n); 
        else
            z1=z(1:(i-1));
        end
    end            
Eff1(1,i)=sum(a(z1,1));
z2=diag(Cf0);
Eff2(1,i)=sum(z2(z1,1));
end

% Sensibility
Sensibility0=zeros(1,n);
for i=1:n
    if Eff1(i)==0&&Eff2(i)==0
        Sensibility0(i)=0;
    else
        Sensibility0(i)=Eff2(i)/Eff1(i)*100;
    end
end

[b, ~]=size(ref_cal);

% Accuracy
Precision0=sum(diag(Cf0))/b*100;

% Specificity
Specificity0=(100-error0)';

Cf0=[a' ;Cf0 ; Specificity0; Sensibility0]; % Confusion matrix

% VALIDATION

Y1 = Sc_val*(Wstar*Q); % Validation prediction
Y1 = Decentrerval( Y1, Y_calb ); % Uncenturing validation by the calibration

Max=max(Y1,[],2); % Maximum for all the sample prediction

% If the value of the cell is the maximum of the raw, the sample is in this
% group (1), otherwise it is not (0)
[e, r]=size(Y1);
Y1p=zeros(e,r);
for i=1:e
    for j=1:r
        if Y1(i,j)==Max(i,1)
            Y1p(i,j)=1;
        else
            Y1p(i,j)=0;
        end
    end
end

% Confusion matrix
a=round(sum(Y_valb))';
Cf=abs(Y1p'*Y_valb);

[n, m]=size(Cf);
for i=1:n
    for j=1:m
        if Cf(i,j)<0.0001
            Cf(i,j)=0;
        end
        Cf(i,j)=round(Cf(i,j));
    end
end

err=zeros(group,1);
for i=1:group
   err(i,:)=a(i,:)-Cf(i,i);
end

error=zeros(group,1);
for i=1:group
    if a(i,1)==0
        error(i,1)=0;
    else
        error(i,1)=err(i,1)/a(i,1)*100;
    end
end

[n, ~]=size(Cf);
z=1:n;
for i=1:n
    if i>1&&i<n
        z1=[z(1:(i-1)) z((i+1):n)];
    else if i==1
           z1=z((i+1):n); 
        else
            z1=z(1:(i-1));
        end
    end  
Eff1(1,i)=sum(a(z1,1));
z2=diag(Cf);
Eff2(1,i)=sum(z2(z1,1));
end

% Sensibility
Sensibility=zeros(1,n);
for i=1:n
    if Eff1(i)==0&&Eff2(i)==0
        Sensibility(i)=0;
    else
        Sensibility(i)=Eff2(i)/Eff1(i)*100;
    end
end

[b, ~]=size(ref_val);

% Accuracy
Precision=sum(diag(Cf))/b*100;

% Specificity
Specificity=(100-error)';

Cf=[a' ;Cf ; Specificity; Sensibility]; % COnfusion matrix

% GRAPHICAL OUTPUT
if fig>0 % If the user want to see the confusion matrix in a figure
    f = figure(fig);
    set(f,'Position',[440 500 461 146]);
    d = Cf;

    % Create the column and row names in cell arrays 
    A=zeros(1,group);
    for i=1:group
    A(1,i)={strcat('G',num2str(i))};
    end
    A1=['Effectifs' A 'Sensibilité(%)' 'Specificité(%)'];
    cnames = strcat(A);
    rnames = strcat(A1);

    % Create the uitable
    t = uitable(f,'Data',d,...
                'ColumnName',cnames,... 
                'RowName',rnames);
    t.Position(3) = t.Extent(3);
    t.Position(4) = t.Extent(4);
end

% OUTPUT STRUCTURE

pred.Confcal=Cf0;
pred.Confval=Cf;
pred.Precisioncal=Precision0;
pred.Precisionval=Precision;
pred.Specificitycal=Specificity0;
pred.Specificityval=Specificity;
pred.Sensibilitycal=Sensibility0;
pred.Sensibilityval=Sensibility;
pred.R2X=R2X;
pred.R2Y=R2Y;
pred.Ypred=Y1p;
pred.Yval=Y_valb;
pred.T=T;
pred.P=P;
pred.W=W;
pred.Wstar=Wstar;
pred.Q=Q;
pred.B=B;
[~, m]=size(T);
pred.VL=m;

end