function [c,ceq] = cbf_constraint(A1,A2,A3,A12,A22,A32,u)
% A_1u+A_2||u||+A_3<=0

c(1)= A1*[u(1);u(2);u(3)]+A2*sqrt(u(1)^2+u(2)^2+u(3)^2)+A3;
c(2)= A12*[u(1);u(2);u(3)]+A22*sqrt(u(1)^2+u(2)^2+u(3)^2)+A32;
ceq = [];
end