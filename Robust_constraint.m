function [c,ceq] = Robust_constraint(x_hat,belta,x_step)
% A_1u+A_2||u||+A_3<=0

c=(x_step(1)-x_hat(1))^2+(x_step(2)-x_hat(2))^2-belta^2;
ceq = [];
end