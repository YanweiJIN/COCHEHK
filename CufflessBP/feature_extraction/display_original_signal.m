clc;
clear all;
close all;
S=load('/media/darcy/Documents/code/shenzhen_task_6_mine_all/TQWT_filtered_feature_data_Api_9th_waveform2BP_overlap/funan_rest/person_10_sample_106_funan_rest_feature.mat');
src_feature_size=513
seq_len=16
SBP=S.feature(:,src_feature_size+1);
DBP=S.feature(:,src_feature_size+2);
MBP=S.feature(:,src_feature_size+3);
SBP_change=S.feature(2:seq_len,src_feature_size+1)-S.feature(1:seq_len-1,src_feature_size+1);
DBP_change=S.feature(2:seq_len,src_feature_size+2)-S.feature(1:seq_len-1,src_feature_size+2);
MBP_change=S.feature(2:seq_len,src_feature_size+3)-S.feature(1:seq_len-1,src_feature_size+3);
figure;
sub(1)=subplot(3,1,1);
plot(SBP);
sub(2)=subplot(3,1,2);
plot(DBP);
sub(3)=subplot(3,1,3);
plot(MBP);
linkaxes(sub,'x');
figure;
sub(1)=subplot(3,1,1);
plot(SBP_change);
sub(2)=subplot(3,1,2);
plot(DBP_change);
sub(3)=subplot(3,1,3);
plot(MBP_change);
linkaxes(sub,'x');