clc
clear

% =================================================
% Sampled-data Meaurement-robust, Risk-aware CBF
% =================================================

%----Load neural network and estimation error by conformal prediction-------

load('Test_input_X.mat');
load('Test_output_X.mat');
load('Test_input_Y.mat');
load('Test_output_Y.mat');
load('a_c2.mat');
mynetopt=load('mynetopt.mat');
netopt2=mynetopt.netopt2;
alpha_p=0.005;

[inputX2, inputStrX] = mapminmax(Test_input_X');             % Normalize input
[outputX2, outputStrX]= mapminmax(Test_output_X');           % Normalize output
[inputY2, inputStrY] = mapminmax(Test_input_Y');             % Normalize input
[outputY2, outputStrY]= mapminmax(Test_output_Y');           % Normalize output
%------------------------------------------------------------------------------
count_failure=0;
dis_save=[];

for count=1:1:2

    %----Scenerio 1: Initial position [-1/sqrt(2);0] -------
    if count<=1
        x=[-1/sqrt(2);0];
        gamma=0.5;
        rho_d=0.1599;
    else
        %----Scenerio 2: Initial position [0;-2/sqrt(5)]-------
        x=[0;-2/sqrt(5)];
        gamma=0.8;
        rho_d=0.45;
    end

    x_g=[1/sqrt(2);1/sqrt(2)];        % Target

    dt=0.001;                         % Time step
    T=0.6;                            % Time horizon
    N=601;                            % The amount of total time step

    u_max=5;                          % The maximal limits of u
    u=[];                             % Real control for robot
    k=10;                             % Norminal control parameter

    %Bz=x^2+y^2;
    G=3*u_max*dt;                     % Strength of noise

    %% ======================Designed parameters===================

    lamda=0.05;

    eta=sqrt(2)*2*G;
    alpha=200;                            % linear K-class function
    delta=10;                             % Every delta sample once


    rho_d_bound=1-erf((1-gamma)/(sqrt(2)*T*eta));

    %%=======================Bounded parameters======================
    F=sqrt(2)*u_max;
    eps_f=0;

    M=sqrt(2)*G;
    eps_q=4*G^2;

    L_lfb=0;
    L_lgb=2;
    L_q=0;

    belta=F*delta*dt+a_c2+lamda;
    belta=0.25;

    b1_min=-1/sqrt(2);
    b1_max=1.5;

    b2_min=0;
    b2_max=1.5;

    eps=1-gamma-sqrt(2)*eta*T*erfinv(1-rho_d);

    ii=0;         % sampling instants
    u_sum=0;

    x_est=[];

    %======================Simulation Process============================
    for i=0:1:N-1

        x_cur=x(:,end);
        y_cur=x_cur+exprnd(1/200,2,1);
       
        normalized_new_input1 = mapminmax('apply', y_cur(1), inputStrX);
        normalized_new_input2 = mapminmax('apply', y_cur(2), inputStrY);
        NN_out_1=netopt2(normalized_new_input1);
        NN_out_2=netopt2(normalized_new_input2);
        x_hat1=mapminmax('reverse', NN_out_1, outputStrX);
        x_hat2=mapminmax('reverse', NN_out_2, outputStrY);

        x_hat=[x_hat1;x_hat2];
        x_est=[x_est, x_hat];

        if mod(i,delta)==0                                        % sample time
            %% CBF_constraint: nonlinear constraint
            % A_1u+A_2||u||+A_3<=0

            if ~isempty(u)
                F=sqrt(sum(u(:,end).^2));
                belta=F*delta*dt+a_c2+lamda;
            end
            LgB=2*x_hat';
            mu_a=alpha*delta*dt*LgB;

            A1=mu_a+LgB;

            mu_b=(alpha*delta*dt+1)*L_lgb*belta;
            mu_c=alpha*delta*dt*(0.5*eps_q+0.5*L_q*belta)+(L_lfb+0.5*L_q)*belta;

            A2=mu_b;

            if ii==0
                A3=0.5*eps_q+mu_c-alpha*eps;
            else
                x_step0=x_est(:,end-1);
                A = [];
                b = [];
                Aeq = [];
                beq = [];
                lb=[-1,-1];
                ub=[1,1];
                fun = @(x_step)-2*x_step(1)*u(1,end)-2*x_step(2)*u(2,end);
                nonlcon = @(x_step)Robust_constraint(x_est(:,end-1),belta,x_step);
                [x_step_opt, fval] = fmincon(fun,x_step0,A,b,Aeq,beq,lb,ub,nonlcon);
                u_sum=u_sum-fval;
                A3=0.5*eps_q+mu_c-alpha*(eps-(eps_f+0.5*eps_q)*dt*(i)-u_sum*delta*dt);
            end

            u_norm=-k*(x_hat-x_g);

            u0=u_norm';    % Initial value for searching control;
            A = [];
            b = [];
            Aeq = [];
            beq = [];
            lb=[-u_max,-u_max];
            ub=[u_max,u_max];
            fun = @(u)(u(1)-u_norm(1))^2 +(u(2)-u_norm(2))^2;
            nonlcon = @(u)cbf_constraint(A1,A2,A3,u);
            u_qp = fmincon(fun,u0,A,b,Aeq,beq,lb,ub,nonlcon);

            u_cur=u_qp';
            ii=ii+1;
        end

        x_next=step_dynamics(x_cur, u_cur, G, dt);
        x=[x x_next];
        u=[u u_cur];
    end

    pro=(1-alpha_p)^ii*(1-M^2*delta*dt/lamda^2)^ii;
    rho=rho_d+1-pro;


   %====================plot figures====================
    theta = linspace(0, 2*pi, 100);
    x_circle = cos(theta);
    y_circle = sin(theta);
    plot(x_circle, y_circle, 'color', '[0, 0.5, 1]', 'LineWidth', 1.5);
    hold on;
    plot(x(1,1), x(2,1), 'o', 'MarkerFaceColor', '[0.8500 0.3250 0.0980]', 'MarkerSize', 10,'MarkerEdgeColor', '[0.9290 0.6940 0.1250]');
    p.Annotation.LegendInformation.IconDisplayStyle = 'off';

    %--------------------plot target--------------------
    center_x = 1/sqrt(2);
    center_y = 1/sqrt(2);
    p=plot(center_x, center_y, 'k*', 'MarkerSize', 10);
    axis equal;
    hold on
    %--------------------plot trajectories-----------------------------

    plot(x(1,1:3:end),x(2,1:3:end),'o', 'MarkerSize', 3, 'MarkerFaceColor', '[0.8500 0.3250 0.0980]', 'MarkerEdgeColor', '[0.9290 0.6940 0.1250]');
    hold on

    xlabel('x_1', 'Fontname','Times New Roman','FontSize', 13);
    ylabel('x_2', 'Fontname','Times New Roman','FontSize', 13);

    legend('Boundary of Safe Region', ...
        'Initial Position','Trajectories', 'Location', 'best', 'Fontname','Times New Roman','FontSize', 12);

    hold on;

    text(x_g(1)+0.1, x_g(2)+0.1, 'Target', 'HorizontalAlignment', 'center', 'Color', 'k', 'Fontname','Times New Roman','FontSize', 13);
    hold on

    dis=sqrt(sum((x(:,end)-x_g).^2));
    dis_save=[dis_save dis];

    if min(abs(sum(x.^2)-1))<0.005
        count_failure=count_failure+1;
    end
    count
end

%sum(dis_save)/count


%%==============Dynamics=============

function xdot= fsi(x, u) % Single intergter
xdot=[u(1); u(2)];
end

function x_next = step_dynamics(x, u, G, dt)
dw=sqrt(dt)*randn(2,1);
x_next = x+dt*fsi(x, u)+G*dw;
end




