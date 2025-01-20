clear;
clc;

% ====Grid state space=====NN training====conformal prediction===

%--------- Grid state space -----
x = -2:0.05:2;
y = -2:0.05:2;
z = -2:0.05:2;

X_points = x(:);
Y_points = y(:);
Z_points = z(:);

[X, Y, Z] = ndgrid(x, y, z);

X_points = x(:);
Y_points = y(:);
Z_points = z(:);

L_p=1;

scatter3(X(:), Y(:), Z(:), 'filled');
xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D Grid Points');
grid on;

hold on
%Generate points on the surface of the sphere
[U, V, W] = sphere(50); % 50 specifies the grid resolution

% Set the radius and center of the sphere
radius = 0.2; % Radius
center = [0, 0, 0]; % Sphere center coordinates

% Scale the sphere and adjust its position
U = radius * U + center(1);
V = radius * V + center(2);
W = radius * W + center(3);

% Plot the sphere
surf(U, V, W, 'FaceColor', 'k', 'EdgeColor', 'none'); % Cyan surface with no grid lines
axis equal; % Keep the axes proportional
camlight; % Add lighting
lighting gouraud; % Apply lighting effect
axis([-1 1 -1 1 -1 1])
grid on
hold on
[U, V, W] = sphere(50); % 50 specifies the grid resolution

% Set the radius and center of the sphere
radius = 0.1; % Radius
center = [-0.6;0;0.7]; % Sphere center coordinates

% Scale the sphere and adjust its position
U = radius * U + center(1);
V = radius * V + center(2);
W = radius * W + center(3);

% Plot the sphere
surf(U, V, W, 'FaceColor', 'k', 'EdgeColor', 'none'); % Cyan surface with no grid lines
axis equal; % Keep the axes proportional
camlight; % Add lighting
lighting gouraud; % Apply lighting effect
axis([-1 1 -1 1 -1 1])

Noise_x=[];
Noise_y=[];
Noise_z=[];

N_total=1010;

%------- NN training ---------
for i=1:N_total

    mu = 0;        % Mean
    sigma=0.02;

    noise_x=sigma*randn(length(X_points),1)+mu;
    Noise_x=[Noise_x noise_x];
    noise_y=sigma*randn(length(X_points),1)+mu;
    Noise_y=[Noise_y noise_y];
    noise_z=sigma*randn(length(Z_points),1)+mu;
    Noise_z=[Noise_z noise_z];
end

Train_input_X=[];
Train_input_Y=[];
Train_output_X=[];
Train_output_Y=[];
Train_input_Z=[];
Train_output_Z=[];
N_train=100;

for i=1:N_train
    Train_input_x=L_p*(X_points+Noise_x(:,i));
    Train_input_X=[Train_input_X;Train_input_x];
    Train_input_y=L_p*(Y_points+Noise_y(:,i));
    Train_input_Y=[Train_input_Y;Train_input_y];
    Train_input_z=L_p*(Z_points+Noise_z(:,i));
    Train_input_Z=[Train_input_Z;Train_input_z];

    Train_output_X=[Train_output_X;X_points];
    Train_output_Y=[Train_output_Y;Y_points];
    Train_output_Z=[Train_output_Z;Z_points];
end

[input1, inputStr] = mapminmax(Train_input_X');           % Normalize input
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

% save('mynetopt2.mat','netopt2')
% 
% mynetopt2=load('mynetopt2.mat');
% netopt2=mynetopt2.netopt2;

Test_input_X=[];
Test_input_Y=[];
Test_input_Z=[];
Test_output_X=[];
Test_output_Y=[];
Test_output_Z=[];

for i=N_train+1:N_total
    Test_input_x=X_points+Noise_x(:,i);
    Test_input_X=[Test_input_X;Test_input_x];
    Test_input_y=Y_points+Noise_y(:,i);
    Test_input_Y=[Test_input_Y;Test_input_y];
    Test_input_z=Z_points+Noise_z(:,i);
    Test_input_Z=[Test_input_Z;Test_input_z];

    Test_output_X=[Test_output_X;X_points];
    Test_output_Y=[Test_output_Y;Y_points];
    Test_output_Z=[Test_output_Z;Z_points];
end

% Test_input=[Test_input_X Test_input_Y];
% Test_output=[Test_output_X Test_output_Y]';

[inputX2, inputStrX] = mapminmax(Test_input_X');            % Normalize input
[outputX2, outputStrX]= mapminmax(Test_output_X');           % Normalize output

[inputY2, inputStrY] = mapminmax(Test_input_Y');            % Normalize input
[outputY2, outputStrY]= mapminmax(Test_output_Y');           % Normalize output

[inputZ2, inputStrZ] = mapminmax(Test_input_Z');            % Normalize input
[outputZ2, outputStrZ]= mapminmax(Test_output_Z');           % Normalize output

% Get output from NN
NN_X_est = netopt2(inputX2);
NN_Y_est = netopt2(inputY2);
NN_Z_est = netopt2(inputZ2);

% Reverse Normalization
NN_X_est=mapminmax('reverse',NN_X_est,outputStrX);
NN_Y_est=mapminmax('reverse',NN_Y_est,outputStrY);
NN_Z_est=mapminmax('reverse',NN_Z_est,outputStrZ);

N_points=length(X_points);
N_test=N_total-N_train;

[a,b]=size(Test_output_X);

R=[];

%=============conformal prediction======

for i=1:N_points
    NN_X_est_tem=NN_X_est(i:N_points:a);
    NN_Y_est_tem=NN_Y_est(i:N_points:a);
    NN_Z_est_tem=NN_Z_est(i:N_points:a);
    NN_est_tem=[NN_X_est_tem;NN_Y_est_tem;NN_Z_est_tem];

    Test_out_tem=[Test_output_X(i:N_points:a)';Test_output_Y(i:N_points:a)';Test_output_Z(i:N_points:a)'];
    R_tem=Conformal_prediction(Test_out_tem,NN_est_tem);
    R=[R R_tem];
end

max(R)

w1 = netopt2.iw{1,1}                   % Weight from input layer to hidden layer
b1 = netopt2.b{1}                      % bias at hidden layer
w2 = netopt2.lw{2,1}                   % weight from hidden layer to output layer
b2 = netopt2.b{2}                      % bias

L_NN=norm(w1)*norm(w2);          % Lips constants

epsion=sqrt(2)/2*0.05;

a_c2=max(R)+(L_NN*L_p+1)*epsion; % Estimation error

% save('a_c2.mat','a_c2');

% save('Test_input_X.mat','Test_input_X');
% save('Test_output_X.mat','Test_output_X');
% save('Test_input_Y.mat','Test_input_Y');
% save('Test_output_Y.mat','Test_output_Y');
% save('Test_input_Z.mat','Test_input_Z');
% save('Test_output_Z.mat','Test_output_Z');
