clc;
clear all;
close all;

origin_src=load('/media/darcy/Documents/code/shenzhen_task_6_mine_all/experiments/2019_4_10_16len_my_513_waveform_based_1_bi_1_uni_lstm_personally_mix_remove_BP_outlier_experiment/result/step-5418-epoch-43/all_data_test_origin_norm_src.mat');
origin_tgt=load('/media/darcy/Documents/code/shenzhen_task_6_mine_all/experiments/2019_4_10_16len_my_513_waveform_based_1_bi_1_uni_lstm_personally_mix_remove_BP_outlier_experiment/result/step-5418-epoch-43/all_data_test_origin_norm_tgt.mat');
prediction_src=load('/media/darcy/Documents/code/shenzhen_task_6_mine_all/experiments/2019_4_10_16len_my_513_waveform_based_1_bi_1_uni_lstm_personally_mix_remove_BP_outlier_experiment/result/step-5418-epoch-43/all_data_test_prediction_norm_src.mat');
prediction_tgt=load('/media/darcy/Documents/code/shenzhen_task_6_mine_all/experiments/2019_4_10_16len_my_513_waveform_based_1_bi_1_uni_lstm_personally_mix_remove_BP_outlier_experiment/result/step-5418-epoch-43/all_data_test_prediction_norm_tgt.mat');
origin_src=origin_src.data;
origin_tgt=origin_tgt.data;
prediction_src=prediction_src.data;
prediction_tgt=prediction_tgt.data;
infer_data=load('/media/darcy/Documents/code/shenzhen_task_6_mine_all/experiments/2019_4_10_16len_my_513_waveform_based_1_bi_1_uni_lstm_personally_mix_remove_BP_outlier_experiment/data/test_data/all_data_test.mat');
infer_data=infer_data.data;
load('/media/darcy/Documents/code/shenzhen_task_6_mine_all/experiments/2019_4_10_16len_my_513_waveform_based_1_bi_1_uni_lstm_personally_mix_remove_BP_outlier_experiment/data/nearby_array.mat');
x=32;
t=1;
infer_input_waveform=infer_data(x,t,1:513);
origin_input_waveform=origin_src(x,t,1:513);
infer_input_waveform_flat=squeeze(squeeze(infer_input_waveform));
origin_input_waveform_flat=squeeze(squeeze(origin_input_waveform));
nearby_input_waveform=squeeze(all_nearby_array(x, :,:,:))
figure;
% plot(infer_input_waveform_flat, 'k');
plot(origin_input_waveform_flat, 'k');
legend('origin')
hold on;
for i=[1:1:size(nearby_input_waveform,1)]
    plot(squeeze(nearby_input_waveform(i,t,1:513)));
    hold on;
end
% here we use SBP to see the nearby output
figure;
% plot(infer_input_waveform_flat, 'k');
origin_output_BP_flat=squeeze(origin_tgt(x,:,3));
plot(origin_output_BP_flat, 'k');
legend('origin')
hold on;
% prediction_output_BP_flat=squeeze(prediction_tgt(x,:,3));
% plot(prediction_output_BP_flat, 'r');
% legend('prediction')
% hold on;
for i=[1:1:size(nearby_input_waveform,1)]
    plot(squeeze(nearby_input_waveform(i,:,516)));
    hold on;
end
