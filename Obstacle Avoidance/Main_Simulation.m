clc
clear

%======================================================
 % Sampled-data, measurement-robust, risk-aware CBF
 % Obstacle avoidance
%=======================================================
% Load NN and estimation error

load('Test_input_X.mat');
load('Test_output_X.mat');
load('Test_input_Y.mat');
load('Test_output_Y.mat');
load('Test_input_Z.mat');
load('Test_output_Z.mat');

load('a_c2.mat');

mynetopt2=load('mynetopt2.mat');
netopt2=mynetopt2.netopt2;

alpha_p=0.005;

[inputX2, inputStrX] = mapminmax(Test_input_X');             % Normalize input
[outputX2, outputStrX]= mapminmax(Test_output_X');           % Normalize output

[inputY2, inputStrY] = mapminmax(Test_input_Y');             % Normalize input
[outputY2, outputStrY]= mapminmax(Test_output_Y');           % Normalize output

[inputZ2, inputStrZ] = mapminmax(Test_input_Z');             % Normalize input
[outputZ2, outputStrZ]= mapminmax(Test_output_Z');           % Normalize output

count_failure=0;

%==== Simulation =======
for count=1:1:15
    r=0.2;
    r2=0.1;

    c=0.8;  % B=e^(-ch)
    a=0.2;


    x=[-0.7;-0.3;-0.5];                 % initial position of state
    x_g=[0.9;0;0.9];                    % State goal

    dt=0.001;                            % Time step
    T=0.75;                              % Time horizon
    N=801;                               % The amount total time step

    u_max=6;                             % The maximal limits of u
    u=[];                                % real control for robot
    k=10;                                % norminal control parameter

    %Bz=x^2+y^2;
    G=6*u_max*dt;                        % Strength of noise

    %% ======================Designed parameters===================

    lamda=0.08;

    x_o=[0;0;0];
    h0=sum((x-x_o).^2)-r^2;
    gamma=exp(-h0);                            % the region of initial state

    x_o2=[-0.6;0;0.7];
    h02=sum((x-x_o2).^2)-r2^2;
    gamma2=exp(-h02);

    %rho_d=0.4209;
    % rho_d=0.0364;
    %rho_d=0.45;
    rho_d=0.0653;

    eta=sqrt(3)*2*G;

    alpha=50;                            % linear K-class function
    delta=15;                             % Every delta sample once


    rho_d_bound=1-erf((1-gamma)/(sqrt(2)*T*eta));
    %=======================Bounded parameters======================

    F=sqrt(3)*(a+u_max);
    eps_f=0;

    M=sqrt(3)*G;
    % result = exp(-(x-50)^2 - (y-50)^2 - (z-50)^2 + r^2) * ...
    %   (-6 + 4*((x-50)^2 + (y-50)^2 + (z-50)^2));

    %========parameters for B1===========
    eps_q=10*G^2;
    L_lgb=2*c^2*r^2;
    L_lfb=a*L_lgb;
    L_q=11*G^2;

    %==========parameters for B2=======
    eps_q2=11*G^2;
    L_lgb2=2*c^2*r^2;
    L_lfb2=a*L_lgb2;
    L_q2=8*G^2;

    belta=F*delta*dt+a_c2+lamda;

    eps=1-gamma-sqrt(2)*eta*T*erfinv(1-rho_d);

    ii=0; % sampling instants
    u_sum=0;
    u_sum2=0;

    x_est=[];

    %======================Simulation process==========================
    for i=0:1:N-1

        x_cur=x(:,end);

        mu = 0;        % Mean
        sigma = 0.02;   % Standard deviation

        noise=sigma*randn(3,1)+mu;
        y_cur=x_cur+noise;

        normalized_new_input1 = mapminmax('apply', y_cur(1), inputStrX);
        normalized_new_input2 = mapminmax('apply', y_cur(2), inputStrY);
        normalized_new_input3 = mapminmax('apply', y_cur(3), inputStrZ);

        NN_out_1=netopt2(normalized_new_input1);
        NN_out_2=netopt2(normalized_new_input2);
        NN_out_3=netopt2(normalized_new_input3);

        x_hat1=mapminmax('reverse', NN_out_1, outputStrX);
        x_hat2=mapminmax('reverse', NN_out_2, outputStrY);
        x_hat3=mapminmax('reverse', NN_out_3, outputStrZ);

        x_hat=[x_hat1;x_hat2;x_hat3];
        x_est=[x_est, x_hat];

        if mod(i,delta)==0                                        % sample time
            %% CBF_constraint: nonlinear constraint

            % A_1u+A_2||u||+A_3<=0
            if ~isempty(u)
                F=sqrt(sum(u(:,end).^2));
                belta=F*delta*dt+a_c2+lamda;
            end
            %=====parameters for B1=======
            LfB=-B(x_hat,x_o,r,c)*2*c*(x_hat-x_o)'*a*[tanh(x_hat(1));tanh(x_hat(2));tanh(x_hat(3))];
            LgB=-B(x_hat,x_o,r,c)*2*c*(x_hat-x_o)';
            mu_a=alpha*delta*dt*LgB;

            A1=mu_a+LgB;

            mu_b=(alpha*delta*dt+1)*L_lgb*belta;

            belta_2=belta;
            mu_c=alpha*delta*dt*(LfB+L_lfb*belta_2+0.5*q(x_hat,x_o,r,G,c)+0.5*L_q*belta_2)+(L_lfb+0.5*L_q)*belta_2;

            A2=mu_b;
            %=====parameters for B2===========
            LfB2=-B(x_hat,x_o2,r2,c)*2*c*(x_hat-x_o2)'*a*[tanh(x_hat(1));tanh(x_hat(2));tanh(x_hat(3))];
            LgB2=-B(x_hat,x_o2,r2,c)*2*c*(x_hat-x_o2)';
            mu_a2=alpha*delta*dt*LgB2;

            A12=mu_a2+LgB2;

            mu_b2=(alpha*delta*dt+1)*L_lgb2*belta;

            belta_2=belta;
            mu_c2=alpha*delta*dt*(LfB2+L_lfb*belta_2+0.5*q(x_hat,x_o2,r2,G,c)+0.5*L_q2*belta_2)+(L_lfb2+0.5*L_q2)*belta_2;

            A22=mu_b2;

            % ========B1========
            if ii==0
                A3=LfB+0.5*eps_q+mu_c-alpha*eps;
            else
                %u_sum=u_sum+max(b1_min*u(1,end),b1_max*u(1,end))+max(b2_min*u(2,end),b2_max*u(2,end));
                %u_sum=0;
                x_step0=x_est(:,end-1);
                A = [];
                b = [];
                Aeq = [];
                beq = [];
                lb=[-1,-1,-1];
                ub=[1,1,1];

                fun=@(x_step)B(x_step,x_o,r,c)*2*c*(x_step-x_o)'*a*[tanh(x_step(1));tanh(x_step(2));tanh(x_step(3))]+B(x_step,x_o,r,c)*2*c*((x_step(1)-x_o(1))'*u(1,end)+(x_step(2)-x_o(2))'*u(2,end)...
                    +(x_step(3)-x_o(3))'*u(3,end));

                nonlcon = @(x_step)Robust_constraint(x_est(:,end-1),belta,x_step);
                [x_step_opt, fval] = fmincon(fun,x_step0,A,b,Aeq,beq,lb,ub,nonlcon);
                

                u_sum=u_sum-fval;
                A3=LfB+0.5*q(x_hat,x_o,r,G,c)+mu_c-alpha*(eps-0.5*eps_q*dt*(i)-u_sum*delta*dt);
            end


            % ========B2========
            if ii==0
                A32=LfB2+0.5*eps_q2+mu_c2-alpha*eps;
            else
                x_step0=x_est(:,end-1);
                A = [];
                b = [];
                Aeq = [];
                beq = [];
                lb=[-1,-1,-1];
                ub=[1,1,1];
     
                fun=@(x_step)B(x_step,x_o2,r2,c)*2*c*(x_step-x_o2)'*a*[tanh(x_step(1));tanh(x_step(2));tanh(x_step(3))]+B(x_step,x_o2,r2,c)*2*c*((x_step(1)-x_o2(1))'*u(1,end)+(x_step(2)-x_o2(2))'*u(2,end)...
                    +(x_step(3)-x_o2(3))'*u(3,end));

                nonlcon = @(x_step)Robust_constraint(x_est(:,end-1),belta,x_step);
                [x_step_opt, fval2] = fmincon(fun,x_step0,A,b,Aeq,beq,lb,ub,nonlcon);

                u_sum2=u_sum2-fval2;
                A32=LfB2+0.5*q(x_hat,x_o2,r2,G,c)+mu_c2-alpha*(eps-0.5*eps_q2*dt*(i)-u_sum2*delta*dt);
            end
            %=====================================================================================================
            u_norm=-k*(x_cur-x_g);

            u0=u_norm'; % Initial for searching;
            A = [];
            b = [];
            Aeq = [];
            beq = [];
            lb=[-u_max,-u_max,-u_max];
            ub=[u_max,u_max,u_max];
            fun = @(u)(u(1)-u_norm(1))^2 +(u(2)-u_norm(2))^2+(u(3)-u_norm(3))^2;
            nonlcon = @(u)cbf_constraint(A1,A2,A3,A12,A22,A32,u);
            u_qp = fmincon(fun,u0,A,b,Aeq,beq,lb,ub,nonlcon);
            u_cur=u_qp';
            ii=ii+1;
        end
        x_next=step_dynamics(a, x_cur, u_cur, G, dt);
        x=[x x_next];
        u=[u u_cur];
    end

    pro=(1-alpha_p)^ii*(1-M^2*delta*dt/lamda^2)^ii;
    rho_sd=rho_d+1-pro;

    %===================plot figures========================================

    % Generate points on the surface of the sphere
    [U, V, W] = sphere(50); % 50 specifies the grid resolution

    % Set the radius and center of the sphere
    radius = r; % Radius
    center = x_o; % Sphere center coordinates

    % Scale the sphere and adjust its position
    U = radius * U + center(1);
    V = radius * V + center(2);
    W = radius * W + center(3);

    % Plot the sphere
    surf(U, V, W, 'FaceColor', 'k', 'EdgeColor', 'none'); % Cyan surface with no grid lines
    axis equal; % Keep the axes proportional
    camlight; % Add lighting
    lighting gouraud; % Apply lighting effect
    hold on

    % Generate points on the surface of the sphere
    [U, V, W] = sphere(50); % 50 specifies the grid resolution

    % Set the radius and center of the sphere
    radius = r2; % Radius
    center = x_o2; % Sphere center coordinates

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

    xlabel('x_1', 'Fontname','Times New Roman','FontSize', 13);
    ylabel('x_2', 'Fontname','Times New Roman','FontSize', 13);
    zlabel('x_3', 'Fontname','Times New Roman','FontSize', 13);

    hold on

    plot3(x(1,:),x(2,:),x(3,:),'r', 'LineWidth', 0.5);
    hold on
end

plot3(x_g(1),x_g(2),x_g(3), 'p', 'MarkerSize', 10, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'y');
hold on
plot3(x(1),x(2),x(3), 'o', 'MarkerSize', 5, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'b');
grid on;
text(x(1,1)-0.1, x(2,1) - 0.1, x(3,1)-0.1, 'Initial Position', 'HorizontalAlignment', 'center', 'Color', 'k', 'Fontname','Times New Roman','FontSize', 13);
text(x_g(1)-0.1, x_g(2) - 0.1, x_g(3)+0.1, 'Goal', 'HorizontalAlignment', 'center', 'Color', 'k', 'Fontname','Times New Roman','FontSize', 13);


%======Functions====
function xdot= fsi(a, x, u) % Single intergter
xdot=a*[tanh(x(1));tanh(x(2));tanh(x(3))]+[u(1); u(2); u(3)];
end

function x_next = step_dynamics(a, x, u, G, dt)
dw=sqrt(dt)*randn(3,1);
x_next = x+dt*fsi(a, x, u)+G*dw;
end

function B_cal = B(x,x_o,r,c) % Single intergter
h=sum((x-x_o).^2)-r^2;
B_cal=exp(-c*h);
end
function q_cal= q(x,x_o,r,G,c)
q_cal=exp(-c*(sum((x-x_o).^2)-r^2))*(-6*c + 2*c^2*(sum((x-x_o).^2)))*G^2;
end
