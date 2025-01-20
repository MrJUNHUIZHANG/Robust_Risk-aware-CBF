function [c,ceq] = cbf_constraint(A1,A2,A3,u)
% A_1u+A_2||u||+A_3<=0

c= A1*[u(1);u(2)]+A2*sqrt(u(1)^2+u(2)^2)+A3;
ceq = [];
end