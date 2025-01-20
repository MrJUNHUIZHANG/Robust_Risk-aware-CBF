function R_tem=Conformal_prediction(Cal_data_out,Cal_data_est)

alpha=0.005;                           % failure probability
[~,N_cal]=size(Cal_data_out);          % size of calibration dataset

% [~,b]=size(calibration_set);

calibration_residual=sqrt(sum((Cal_data_out-Cal_data_est).^2));

[calibration_residual_sort,I]=sort(calibration_residual);

position=ceil((N_cal+1)*(1-alpha));

if position>N_cal
   position=N_cal;    
end
R_tem=calibration_residual_sort(position);
end