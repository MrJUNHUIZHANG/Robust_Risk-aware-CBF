clear;
clc;

% =====Grid State Space=====train neural network=======conformal
% prediction=======


%------------Sample grid points------------
x = -1.0:0.02:1.0;
y = -1.0:0.02:1.0;

[X, Y] = meshgrid(x, y);
X_points = X(:);
Y_points = Y(:);
L_p=1;

figure;
plot(X_points, Y_points, '.', 'MarkerSize', 3); 

hold on
theta = linspace(0, 2*pi, 100); 
x_circle = cos(theta);
y_circle = sin(theta);
plot(x_circle, y_circle, 'b', 'LineWidth', 2); 

xlabel('X-axis');
ylabel('Y-axis');
title('Grid Points for State Space');
axis equal; 
grid on;
hold off;

%----------------train neural network------------------
Noise_x=[];
Noise_y=[];

N_total=1010;

for i=1:N_total
noise_x=exprnd(1/200,length(X_points),1);
Noise_x=[Noise_x noise_x];
noise_y=exprnd(1/200,length(Y_points),1);
Noise_y=[Noise_y noise_y];
end

Train_input_X=[];
Train_input_Y=[];
Train_output_X=[];
Train_output_Y=[];
N_train=100;

for i=1:N_train
Train_input_x=L_p*(X_points+Noise_x(:,i));
Train_input_X=[Train_input_X;Train_input_x];
Train_input_y=L_p*Y_points+Noise_y(:,i);
Train_input_Y=[Train_input_Y;Train_input_y];

Train_output_X=[Train_output_X;X_points];
Train_output_Y=[Train_output_Y;Y_points];
end

% Train_input=[Train_input_X Train_input_Y];
% Train_output=[Train_output_X Train_output_Y]';

[input1, inputStr] = mapminmax(Train_input_X');            % Normalize input
[output1, outputStr]= mapminmax(Train_output_X');          % Normalize output

fitper=0; 

for i=1:1

    net = feedforwardnet([1]);
    net = train(net,input1,output1);

    NN_output = net(input1);
    NN_output=mapminmax('reverse',NN_output,outputStr);     % Reverse normalization

    cost_func = 'NRMSE';
    fit = goodnessOfFit(NN_output',Train_output_X,cost_func);  
    fitper_new=1-fit

    if fitper_new>fitper
        fitper= fitper_new
        netopt2=net;
    end
end

save('mynetopt2.mat','netopt2')

mynetopt2=load('mynetopt2.mat');
netopt2=mynetopt2.netopt2;

Test_input_X=[];
Test_input_Y=[];
Test_output_X=[];
Test_output_Y=[];

for i=N_train+1:N_total
Test_input_x=X_points+Noise_x(:,i);
Test_input_X=[Test_input_X;Test_input_x];
Test_input_y=Y_points+Noise_y(:,i);
Test_input_Y=[Test_input_Y;Test_input_y];

Test_output_X=[Test_output_X;X_points];
Test_output_Y=[Test_output_Y;Y_points];
end

% Test_input=[Test_input_X Test_input_Y];
% Test_output=[Test_output_X Test_output_Y]';

[inputX2, inputStrX] = mapminmax(Test_input_X');            % Normalize input
[outputX2, outputStrX]= mapminmax(Test_output_X');           % Normalize output

[inputY2, inputStrY] = mapminmax(Test_input_Y');            % Normalize input
[outputY2, outputStrY]= mapminmax(Test_output_Y');           % Normalize output

% Get output from NN
NN_X_est = netopt2(inputX2);
NN_Y_est = netopt2(inputY2);

% Reverse Normalization
NN_X_est=mapminmax('reverse',NN_X_est,outputStrX);
NN_Y_est=mapminmax('reverse',NN_Y_est,outputStrY);

N_points=length(X_points);
N_test=N_total-N_train;

[a,b]=size(Test_output_X);

R=[];

%-------Conformal predition------------

for i=1:N_points
NN_X_est_tem=NN_X_est(i:N_points:a);
NN_Y_est_tem=NN_Y_est(i:N_points:a);
NN_est_tem=[NN_X_est_tem;NN_Y_est_tem];

Test_out_tem=[Test_output_X(i:N_points:a)';Test_output_Y(i:N_points:a)'];
R_tem=Conformal_prediction(Test_out_tem,NN_est_tem);
R=[R R_tem];
end

max(R)

w1 = netopt2.iw{1,1}                   % Weight from input layer to hidden layer
b1 = netopt2.b{1}                      % bias at hidden layer
w2 = netopt2.lw{2,1}                   % weight from hidden layer to output layer
b2 = netopt2.b{2}                      % bias

L_NN=norm(w1)*norm(w2);          % Lips constants

epsion=sqrt(2)/2*0.02;

a_c2=max(R)+(L_NN*L_p+1)*epsion; % Estimation error bound for any point at state space

% save('a_c2.mat','a_c2');
% 
% save('Test_input_X.mat','Test_input_X');
% save('Test_output_X.mat','Test_output_X');
% save('Test_input_Y.mat','Test_input_Y');
% save('Test_output_Y.mat','Test_output_Y');
