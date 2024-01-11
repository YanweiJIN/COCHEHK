clc;
clear all;
close all;

% directory='/media/darcy/Documents/code/shenzhen_task_1';
% experiment_name='MultiLSTM_lstm_size_128_num_step_64_num_layers_4_2018-08-23';
% step_name='step_100000';
% name_collection={'test_day_1','test_day_2','test_day_3','test_day_4','test_day_1_2_3_4'};

channel_collection={'SBP','DBP','MBP'};
%for(i=[2:1:size(name_collection,2)-1])
%for(i=[1:1:size(name_collection,2)-1])
data_origin=load('/media/darcy/Documents/code/shenzhen_task_6_mine_all/experiments/2019_4_11_16len_my_513_waveform_based_1_bi_1_uni_lstm_personally_mix_remove_BP_outlier_good_overlap_experiment/result/step-30000-epoch-84/all_data_train_origin_real_tgt.mat');
data_origin=data_origin.data;
data_predictions=load('/media/darcy/Documents/code/shenzhen_task_6_mine_all/experiments/2019_4_11_16len_my_513_waveform_based_1_bi_1_uni_lstm_personally_mix_remove_BP_outlier_good_overlap_experiment/result/step-30000-epoch-84/all_data_train_prediction_real_tgt.mat');
data_predictions=data_predictions.data;
 
%     data_outlier=load('/media/darcy/Documents/code/shenzhen_task_6_mine_all/experiments/2019_4_11_16len_my_513_waveform_based_1_bi_1_uni_lstm_personally_mix_remove_BP_outlier_good_overlap_experiment/data/outlier_data.mat')
%     data_outlier=data_outlier.data;
%     data_origin=data_outlier(:,:,514:516);
%     data_predictions=data_origin;

data_origin=data_origin(1:min(1000, size(data_origin,1)),:,:);
data_predictions=data_predictions(1:min(1000,size(data_origin,1)),:,:);
for(k=[1:1:size(data_origin,1)])
    if(k==1)
        data_origin_flat=squeeze(data_origin(k,:,:));
        data_predictions_flat=squeeze(data_predictions(k,:,:));
    else
        data_origin_flat=vertcat(data_origin_flat, squeeze(data_origin(k,:,:)));
        data_predictions_flat=vertcat(data_predictions_flat, squeeze(data_predictions(k,:,:)));
    end
end
% display the true and predicted data with time axis 
figure;
for (j=[1:1:3])
    sub(j)=subplot(3,1,j);
    plot([1:1:size(data_origin_flat,1)], data_origin_flat(:,j),'b')
    hold on;
    plot([1:1:size(data_predictions_flat,1)], data_predictions_flat(:,j),'r')
    hold on;
    for(s=[16.5:16:size(data_origin_flat,1)])
        plot([s, s],ylim,'g')
    end
    ylabel(channel_collection{1,j});
    xlabel('beat');
    legend({'original','predictions'})
    %title(strrep(name_collection{1,i},'_','-'));
end
linkaxes(sub,'x');
