clc;
clear all;
close all;

data_origin=load('/media/darcy/Documents/code/shenzhen_task_6_mine_all/experiments/2019_4_11_16len_my_513_waveform_based_1_bi_1_uni_lstm_personally_mix_remove_BP_outlier_good_overlap_experiment/result/step-30000-epoch-84/all_data_train_origin_real_src.mat');
data_origin=data_origin.data;
data_predictions=load('/media/darcy/Documents/code/shenzhen_task_6_mine_all/experiments/2019_4_11_16len_my_513_waveform_based_1_bi_1_uni_lstm_personally_mix_remove_BP_outlier_good_overlap_experiment/result/step-30000-epoch-84/all_data_train_prediction_real_src.mat');
data_predictions=data_predictions.data;

data_origin=data_origin(1:min(100, size(data_origin,1)),:,:);
data_predictions=data_predictions(1:min(100,size(data_origin,1)),:,:);
for(k=[1:1:size(data_origin,1)])
    if(k==1)
        data_origin_flat=squeeze(data_origin(k,:,:));
        data_predictions_flat=squeeze(data_predictions(k,:,:));
    else
        data_origin_flat=vertcat(data_origin_flat, squeeze(data_origin(k,:,:)));
        data_predictions_flat=vertcat(data_predictions_flat, squeeze(data_predictions(k,:,:)));
    end
end

for(k=[1:1:size(data_origin_flat,1)])
    if(k==1)
        data_origin_flat_flat=data_origin_flat(k,:).';
        data_predictions_flat_flat=data_predictions_flat(k,:).';
    else
        data_origin_flat_flat=vertcat(data_origin_flat_flat,data_origin_flat(k,:).');
        data_predictions_flat_flat=vertcat(data_predictions_flat_flat, data_predictions_flat(k,:).');
    end
end
% display the true and predicted data with time axis 
figure;
plot(data_origin_flat_flat);
hold on;
plot(data_predictions_flat_flat,'r');
