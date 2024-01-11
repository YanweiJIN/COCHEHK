import os
import scipy.io as scio
import numpy as np


def judge(data_dir, src_feature_size, tgt_feature_size, nearby_num):
    for root, _, name in os.walk(os.path.join(data_dir, 'test_data')):
        for x in name:
            if x.split('.')[-1] == 'mat' and x.split('.')[-2].split('_')[-1] == 'test':
                infer_data_path = os.path.join(root, x)
    print('infer_data_path')
    print(infer_data_path)
    infer_data = scio.loadmat(infer_data_path)['data']
    print('infer_data.shape[0]')
    print(infer_data.shape[0])
    for root, _, name in os.walk(os.path.join(data_dir, 'train_data')):
        for x in name:
            if x.split('.')[-1] == 'mat':
                train_data_path = os.path.join(root, x)
    train_data = scio.loadmat(train_data_path)['data']
    print('train_data_path')
    print(train_data_path)
    error_array = np.zeros(
        (infer_data.shape[0], 2))  # the first for calculation of src difference, the second is for target
    all_nearby_array = np.zeros(
        (infer_data.shape[0], nearby_num, infer_data.shape[1], src_feature_size + tgt_feature_size))
    for i in range(infer_data.shape[0]):
        current_infer_src = infer_data[i, :, :src_feature_size]
        diff_src_data = np.zeros((train_data.shape[0], train_data.shape[1], src_feature_size))
        for j in range(train_data.shape[0]):
            diff_src_data[j, :, :src_feature_size] = train_data[j, :, :src_feature_size] - current_infer_src
        diff_src_data = np.power(diff_src_data, 2)
        diff_src_data = np.sum(diff_src_data, axis=(1, 2))
        nearby_index = diff_src_data.argsort()[:nearby_num]
        nearby_array = train_data[nearby_index]
        #
        nearby_src = nearby_array[:, :, :src_feature_size]
        nearby_src_mean = np.mean(nearby_src, axis=0)
        nearby_src_error = nearby_src - np.repeat(np.expand_dims(nearby_src_mean, axis=0), nearby_num, axis=0)
        nearby_src_error = np.mean(np.power(nearby_src_error, 2))
        error_array[i, 0] = nearby_src_error
        nearby_tgt = nearby_array[:, :, src_feature_size:src_feature_size + tgt_feature_size]
        nearby_tgt_mean = np.mean(nearby_tgt, axis=0)
        nearby_tgt_error = nearby_tgt - np.repeat(np.expand_dims(nearby_tgt_mean, axis=0), nearby_num, axis=0)
        nearby_tgt_error = np.mean(np.power(nearby_tgt_error, 2))
        error_array[i, 1] = nearby_tgt_error
        all_nearby_array[i, :, :] = nearby_array
        print(i)
    scio.savemat(os.path.join(data_dir, 'nearby_array.mat'), {'all_nearby_array': all_nearby_array})
    scio.savemat(os.path.join(data_dir, 'nearby_error.mat'), {'error': error_array})


if __name__ == '__main__':
    data_dir = '/media/darcy/Documents/code/shenzhen_task_6_mine_all/experiments/2019_4_10_16len_my_513_waveform_based_1_bi_1_uni_lstm_personally_mix_remove_BP_outlier_experiment/data'
    src_feature_size = 513
    tgt_feature_size = 3
    nearby_num = 10
    judge(data_dir, src_feature_size, tgt_feature_size, nearby_num)
