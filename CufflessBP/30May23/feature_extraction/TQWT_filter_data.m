% this program is to convert the original signal to the filtered signal
% using TQWT to remove the baseline wandering, the length of data has not
% been changed

clc;
clear all;
close all;

directory='/media/darcy/Documents/code/shenzhen_task_6_mine_all';
from_dir='/personal_conditional_data_mat';
to_dir='/filtered_personal_conditional_data_mat';
name_cell0={'shenzixiao','funan_rest','funan_sport','hongxi_day_1','hongxi_day_2','hongxi_day_3','hongxi_day_4'};

mkdir(strcat(directory,to_dir));
for k=[1:1:size(name_cell0,2)]
    mkdir(strcat(directory,to_dir,'/',name_cell0{1,k}));
end
for(p=[1:1:7])  % source
    if(p==1)
        q_max=62;
    elseif(p==2 | p==3)
        q_max=45;
    elseif(p>=4 & p<=7)
        q_max=12;
    end
    for(q=1:1:q_max) % person 
        S=load(strcat(directory,from_dir,'/',name_cell0{1,p},'/','person_',num2str(q),'_',name_cell0{1,p},'_data.mat'));
        data=S.data;
        BP=data(:,1);
        ECG=data(:,2);
        PPG=data(:,3);
        fs=200; % here original frequency is 200 Hz
        % filter BP, ECG, PPG
        BP_ECG_PPG={BP,ECG,PPG};
        for k=[2:1:3] % here we dont process BP
            x=BP_ECG_PPG{1,k};
            N=size(x,1);
            Q = 1; r = 3; J = 9;
            w = tqwt_radix2(x,Q,r,J);
            w_new=cell(1,10);
            for i=[1:1:9]
                w_new{1,i}=w{1,i};
            end
            w_new{1,10}=zeros(size(w{1,10}));
            x_recon = itqwt_radix2(w_new,Q,r,N);
            BP_ECG_PPG{1,k}=x_recon.'; % note that we transpose to column vector
        end
        data=horzcat(BP_ECG_PPG{1,1},BP_ECG_PPG{1,2},BP_ECG_PPG{1,3});
        filter_data_mat=strcat(directory,to_dir,'/',name_cell0{1,p},'/','person_',num2str(q),'_',name_cell0{1,p},'_data.mat');
        save(filter_data_mat, 'data');
    end
end




