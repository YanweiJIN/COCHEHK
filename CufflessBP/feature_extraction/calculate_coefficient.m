clc;
clear all;
close all;

origin_src=load('/media/darcy/Documents/code/shenzhen_task_6_mine_all/experiments/2019_4_11_16len_my_513_waveform_based_1_bi_1_uni_lstm_personally_mix_remove_BP_outlier_good_overlap_experiment/result/step-30000-epoch-84/all_data_train_origin_norm_src.mat');
origin_tgt=load('/media/darcy/Documents/code/shenzhen_task_6_mine_all/experiments/2019_4_11_16len_my_513_waveform_based_1_bi_1_uni_lstm_personally_mix_remove_BP_outlier_good_overlap_experiment/result/step-30000-epoch-84/all_data_train_origin_norm_tgt.mat');
prediction_src=load('/media/darcy/Documents/code/shenzhen_task_6_mine_all/experiments/2019_4_11_16len_my_513_waveform_based_1_bi_1_uni_lstm_personally_mix_remove_BP_outlier_good_overlap_experiment/result/step-30000-epoch-84/all_data_train_prediction_norm_src.mat');
prediction_tgt=load('/media/darcy/Documents/code/shenzhen_task_6_mine_all/experiments/2019_4_11_16len_my_513_waveform_based_1_bi_1_uni_lstm_personally_mix_remove_BP_outlier_good_overlap_experiment/result/step-30000-epoch-84/all_data_train_prediction_norm_tgt.mat');

origin_src=origin_src.data;
origin_tgt=origin_tgt.data;
prediction_src=prediction_src.data;
prediction_tgt=prediction_tgt.data;
error_src=squeeze(squeeze(sum(sum((origin_src-prediction_src).^2,2),3)));

% load('/media/darcy/Documents/code/shenzhen_task_6_mine_all/experiments/2019_4_10_16len_my_513_waveform_based_1_bi_1_uni_lstm_personally_mix_remove_BP_outlier_experiment/data/nearby_error.mat');
% error_src=error(:,2);

error_tgt=squeeze(squeeze(sum(sum((origin_tgt-prediction_tgt).^2,2),3)));
coefficient_origin=corrcoef(error_src, error_tgt);
mean_error_src=mean(error_src);
std_error_src=std(error_src);
outlier_error_src_index=find(error_src>mean_error_src+std_error_src);
mean_error_tgt=mean(error_tgt);
std_error_tgt=std(error_tgt);
outlier_error_tgt_index=find(error_tgt>mean_error_tgt+std_error_tgt);
outlier_all_index=unique([outlier_error_src_index;outlier_error_tgt_index]);
error_src_part=error_src;
error_src_part(outlier_all_index)=[];
error_tgt_part=error_tgt;
error_tgt_part(outlier_all_index)=[];
coefficient_filtered=corrcoef(error_src_part, error_tgt_part);
figure;
scatter(error_src, error_tgt);
figure;
scatter(error_src_part, error_tgt_part);

%coefficient_judge=corrcoef(error_tgt, error)