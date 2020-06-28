function pred = PredDA(X_cal, X_val, Ref_cal, Ref_val, group,para, model,para2, fig)
% Creation of a model with DA algorithm. The model is calculate with a
% calibration set, after the validation set is predict. 
% To verify the quality of the model, a confusion matrix and some
% coefficients are define.

% Input:    X_cal,X_val: calibration and validation PCA scores
%           ref_cal, ref_val: calibration and validation references
%           group: number of group
%           para: A REVOIIIIIIIRRRRRR
%           model: indicate if you create a model (0) or if you use a
%                   existant model (1)
%           para2:  if model=1, you have to define a vector with the
%                   parameters of the model 
%           fig:    number of the figure if the user want to create
%                   graphical output 

% Output:   pred structure:
%               model: parameters of the model
%               label: prediction
%               Cf: confusion matrix
%               Precision: Accuracy of the model
%               score: other parameters

if model==1 % if a model is already created
    modelDA=para2.LDA.model;
else % creation of a model
    modelDA = fitcdiscr(X_cal, Ref_cal,'DiscrimType',para.Distance{1});
end

% CALIBRATION

[label,~] = predict(modelDA,X_cal); % calibration prediction

% Confusion Matrix
labelc = Conf(label, group);
refc = Conf(Ref_cal, group);
a=round(sum(refc))';
Confus0=abs(labelc'*refc);
err0=zeros(group,1);
for i=1:group
   err0(i,:)=a(i,:)-Confus0(i,i);
end

error0=zeros(group,1);
for i=1:group
    if a(i,1)==0
        error0(i,1)=0;
    else
        error0(i,1)=err0(i,1)/a(i,1)*100;
    end
end

[n, ~]=size(Confus0);
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
z2=diag(Confus0);
Eff2(1,i)=sum(z2(z1,1));
end
Sensibility0=Eff2./Eff1*100; % Sensibility parameter

[b, ~]=size(Ref_cal);

Precision0=sum(diag(Confus0))/b*100; % Accuracy parameter
Specificity0=(100-error0)'; % Specificity parameter

Cf0=[a' ;Confus0 ; Specificity0; Sensibility0]; % Confusion matrix

% VALIDATION

[label,score] = predict(modelDA,X_val); % validation prediction

% Confusion matrix         
labelc = Conf(label, group);
refc = Conf(Ref_val, group);
a=round(sum(refc))';
Confus=abs(labelc'*refc);
err=zeros(group,1);
for i=1:group
   err(i,:)=a(i,:)-Confus(i,i);
end

error=zeros(group,1);
for i=1:group
    if a(i,1)==0
        error(i,1)=0;
    else
        error(i,1)=err(i,1)/a(i,1)*100;
    end
end

[n, ~]=size(Confus);
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
z2=diag(Confus);
Eff2(1,i)=sum(z2(z1,1));
end
Sensibility=Eff2./Eff1*100; % Sensibility parameter

[b, ~]=size(Ref_val);

Precision=sum(diag(Confus))/b*100; % Accuracy parameter
Specificity=(100-error)'; % Specificity parameter

Cf=[a' ;Confus ; Specificity; Sensibility]; % Confusion matrix

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
pred.model=modelDA;
pred.label=label;
pred.score=score;
end