clc;
clear all;
close all;
S=load('/media/darcy/Documents/code/shenzhen_task_6_mine_all/personal_conditional_data_mat/hongxi_day_2/person_3_hongxi_day_2_data.mat');
BP=S.data(:,1);
ECG=S.data(:,2);
PPG=S.data(:,3);
figure;
fs = 200;
x=PPG;
%x=BP
%x=ECG
N=size(x,1);
Q = 1; r = 3; J = 9;
w = tqwt_radix2(x,Q,r,J);
PlotSubbands(x,w,Q,r,1,J+1,fs);
w_new=cell(1,10);
for i=[1:1:9]
    w_new{1,i}=w{1,i};
end
w_new{1,10}=zeros(size(w{1,10}));
x_recon = itqwt_radix2(w_new,Q,r,N);
figure;
plot(x_recon);
w_2 = tqwt_radix2(x_recon,Q,r,J);
PlotSubbands(x_recon,w_2,Q,r,1,J+1,fs);
figure;
plot(x);
hold on;
plot(x_recon);


