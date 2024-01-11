from __future__ import print_function

import os

import scipy.io as scio
import numpy as np
import tensorflow as tf


def initialize_dir_filter():
    """create data directory filter"""
    all_dir = ['shenzixiao', 'funan_rest', 'funan_sport',
               'hongxi_day_1', 'hongxi_day_2', 'hongxi_day_3', 'hongxi_day_4',
               'summer_day_1_1', 'summer_day_1_2', 'summer_day_1_3', 'summer_day_1_4', 'summer_day_1_5',
               'summer_day_2', 'summer_day_3', 'summer_day_4']
    rest_filter = ['shenzixiao', 'funan_rest', 'hongxi_day_1', 'hongxi_day_2', 'hongxi_day_3', 'hongxi_day_4',
                   'summer_day_1_1', 'summer_day_1_2', 'summer_day_1_3', 'summer_day_1_4', 'summer_day_1_5',
                   'summer_day_2', 'summer_day_3', 'summer_day_4']  # summer all may have some not rest data
    sport_filter = ['funan_sport']
    shenzhen_filter = ['shenzixiao', 'funan_rest', 'funan_sport', 'hongxi_day_1',
                       'hongxi_day_2', 'hongxi_day_3', 'hongxi_day_4']
    summer_filter = ['summer_day_1_1', 'summer_day_1_2', 'summer_day_1_3', 'summer_day_1_4', 'summer_day_1_5',
                     'summer_day_2', 'summer_day_3', 'summer_day_4']
    day1_filter = ['shenzixiao', 'funan_rest', 'hongxi_day_1',
                   'summer_day_1_1', 'summer_day_1_2', 'summer_day_1_3', 'summer_day_1_4', 'summer_day_1_5']
    day2_filter = ['hongxi_day_2', 'summer_day_2']
    day3_filter = ['hongxi_day_3', 'summer_day_3']
    day4_filter = ['hongxi_day_4', 'summer_day_4']
    return all_dir, rest_filter, sport_filter, shenzhen_filter, summer_filter, day1_filter, \
           day2_filter, day3_filter, day4_filter


def get_filelist(data_source_dir, data_dir):
    '''structure: data_source_dir -- data_dir -- data_files
            this generate the file list and load from them to create the data array'''
    filelist = []
    for x in data_dir:
        for root, _, files in os.walk(os.path.join(data_source_dir, x)):
            for name in files:
                filelist.append(os.path.join(root, name))
    return filelist


def get_data(filelist):
    # here we use data block method to speed up the loading process
    block_size = 1000
    data_block_list = []
    for i in range(len(filelist)):
        sample_data = scio.loadmat(filelist[i])['feature']  # this is rank-2
        sample_data = np.expand_dims(sample_data, axis=0)  # this is rank-3
        if i % block_size == 0:
            if i != 0:
                data_block_list.append(data)
            data = sample_data
        else:
            data = np.concatenate((data, sample_data), axis=0)  # this is rank-3
        if i == len(filelist) - 1 and len(filelist) % block_size != 0:
            data_block_list.append(data)
    for j in range(len(data_block_list)):
        if j == 0:
            data = data_block_list[j]
        else:
            data = np.concatenate((data, data_block_list[j]), axis=0)  # this is rank-3
    return data


def shuffle_filelist(filelist, seed):
    """shuffle the filelist"""
    index = np.arange(len(filelist))
    np.random.seed(seed)
    np.random.shuffle(index)
    filelist = [filelist[i] for i in index]
    return filelist


def get_info(file):
    data_group = file.split('/')[-2]
    detail = file.split('/')[-1]
    person_num = int(detail[0:detail.find('_sample')].strip('person_'))
    sample_num = int(detail[detail.find('sample'):].split('_')[1])
    return data_group, person_num, sample_num


def sort_filelist(filelist):
    # the filename is like "/media/darcy/Documents/code/shenzhen_task_1/feature_data_fill_Jan_3rd_overlap/summer_day_2/person_10_sample_100_summer_day_2_feature.mat"
    for file in filelist:
        filelist = sorted(filelist, key=lambda file: get_info(file))
    return filelist


def split_filelist(filelist, seed, train_ratio, eval_ratio, batch_size, split_by_person):
    if split_by_person:
        person_filelist = []
        for file in filelist:
            person_filelist.append(file[:file.find('sample')])
        person_filelist = list(set(person_filelist))
        person_filelist.sort()
        train_filelist = []
        eval_filelist = []
        test_filelist = []
        for i in person_filelist:
            single_person_filelist = []
            for j in filelist:
                if i in j:
                    single_person_filelist.append(j)
            single_person_filelist = shuffle_filelist(single_person_filelist, seed)
            train_split_index = int(np.floor(len(single_person_filelist) * train_ratio))
            eval_split_index = int(np.floor(len(single_person_filelist) * (train_ratio + eval_ratio)))
            train_filelist.extend(single_person_filelist[0: train_split_index])
            eval_filelist.extend(single_person_filelist[train_split_index:eval_split_index])
            test_filelist.extend(single_person_filelist[eval_split_index:])

        # make the training sample num the multiple of batch size (easy for backwrd training
        # move the margin data from train set to test set
        real_train_split_index = int(np.floor(len(train_filelist) / batch_size) * batch_size)
        train_filelist = train_filelist[0:real_train_split_index]
        test_filelist.extend(train_filelist[real_train_split_index:])
    else:
        # the filelist needs to be shuflled fist to ensure random select
        filelist = shuffle_filelist(filelist, seed)
        # split data
        train_split_index = int(np.floor(len(filelist) * hparams.train_ratio / hparams.batch_size) * hparams.batch_size)
        eval_split_index = int(np.floor(len(filelist) * (hparams.train_ratio + hparams.eval_ratio)))
        train_filelist = filelist[0:train_split_index]
        eval_filelist = filelist[train_split_index:eval_split_index]
        test_filelist = filelist[eval_split_index:]
    return train_filelist, eval_filelist, test_filelist


def split_overlap_filelist(filelist, seed, train_ratio, eval_ratio, batch_size, shift_sequence_num):
    segment_len = 36
    person_filelist = []
    for file in filelist:
        person_filelist.append(file[:file.find('sample')])
    person_filelist = list(set(person_filelist))
    person_filelist.sort()
    train_filelist = []
    eval_filelist = []
    test_filelist = []
    for i in person_filelist:
        single_person_filelist = []
        for j in filelist:
            if i in j:
                single_person_filelist.append(j)

        def get_sample_num(file):
            sample_num = int(file[file.find('sample'):].split('_')[1])
            return sample_num

        single_person_filelist.sort(key=get_sample_num)

        segment_num = int(np.floor(len(single_person_filelist) / segment_len))
        for k in range(segment_num):
            current_segment = single_person_filelist[k * segment_len:(k + 1) * segment_len - 1]
            train_filelist.extend(current_segment[shift_sequence_num:int(np.floor(segment_len * train_ratio))])
            eval_filelist.extend(current_segment[int(np.floor(segment_len * train_ratio)) + shift_sequence_num:int(
                np.floor(segment_len * (train_ratio + eval_ratio)))])
            test_filelist.extend(current_segment[int(np.floor(segment_len * (train_ratio + eval_ratio))):])
        if len(single_person_filelist) % segment_len != 0:
            residual_filelist = single_person_filelist[segment_num * segment_len:]
            residual_len = len(residual_filelist)
            if int(np.floor(residual_len * train_ratio)) > shift_sequence_num and (
                    int(np.floor(residual_len * (train_ratio + eval_ratio))) > int(
                    np.floor(residual_len * train_ratio)) + shift_sequence_num ):
                train_filelist.extend(residual_filelist[shift_sequence_num:int(np.floor(residual_len * train_ratio))])
                eval_filelist.extend(residual_filelist[int(np.floor(residual_len * train_ratio)) + shif_sequence_num:int(
                    np.floor(residual_len * (train_ratio + eval_ratio)))])
                test_filelist.extend(residual_filelist[int(np.floor(residual_len * (train_ratio + eval_ratio))):])
    # make the training sample num the multiple of batch size (easy for backwrd training
    # move the margin data from train set to test set
    real_train_split_index = int(np.floor(len(train_filelist) / batch_size) * batch_size)
    train_filelist = train_filelist[0:real_train_split_index]
    return train_filelist, eval_filelist, test_filelist


def store_filelist(data_dir, filelist, filelist_name):
    with open(os.path.join(data_dir, filelist_name), 'w') as filelist_fo:
        for x in filelist:
            filelist_fo.write('%s\n' % x)


def remove_BP_outlier(filelist, hparams):
    all_data = get_data(filelist)
    all_data_BP = all_data[:, :, hparams.src_feature_size:hparams.src_feature_size + hparams.tgt_feature_size]
    all_data_BP_change_1 = all_data_BP[:, 1:, :] - all_data_BP[:, :-1, :]  # this mean the change of adjacent BP value
    all_data_BP_change_2 = all_data_BP[:, 2:, :] - all_data_BP[:, :-2, :]  # this mean the change of adjacent BP value
    all_data_BP_change_3 = all_data_BP[:, 3:, :] - all_data_BP[:, :-3, :]  # this mean the change of adjacent BP value
    mean_array = np.array(
        [[np.mean(all_data_BP[:, :, 0]), np.mean(all_data_BP[:, :, 1]), np.mean(all_data_BP[:, :, 2])],
         [np.mean(all_data_BP_change_1[:, :, 0]), np.mean(all_data_BP_change_1[:, :, 1]),
          np.mean(all_data_BP_change_1[:, :, 2])],
         [np.mean(all_data_BP_change_2[:, :, 0]), np.mean(all_data_BP_change_2[:, :, 1]),
          np.mean(all_data_BP_change_2[:, :, 2])],
         [np.mean(all_data_BP_change_3[:, :, 0]), np.mean(all_data_BP_change_3[:, :, 1]),
          np.mean(all_data_BP_change_3[:, :, 2])]])
    std_array = np.array([[np.std(all_data_BP[:, :, 0]), np.std(all_data_BP[:, :, 1]), np.std(all_data_BP[:, :, 2])],
                          [np.std(all_data_BP_change_1[:, :, 0]), np.std(all_data_BP_change_1[:, :, 1]),
                           np.std(all_data_BP_change_1[:, :, 2])],
                          [np.std(all_data_BP_change_2[:, :, 0]), np.std(all_data_BP_change_2[:, :, 1]),
                           np.std(all_data_BP_change_2[:, :, 2])],
                          [np.std(all_data_BP_change_3[:, :, 0]), np.std(all_data_BP_change_3[:, :, 1]),
                           np.std(all_data_BP_change_3[:, :, 2])]])
    outlier_list = []
    print('start')
    print(all_data.shape[0])
    for i in range(all_data.shape[0]):
        print(i)
        is_outlier_seq = 0
        for k in range(3):  # refer to three kind of BP
            for j in range(all_data_BP.shape[1]):
                if all_data_BP[i, j, k] > mean_array[0, k] + 3 * std_array[0, k] or all_data_BP[i, j, k] < mean_array[
                    0, k] - 3 * std_array[0, k]:
                    is_outlier_seq = 1
                    break
            for j in range(all_data_BP_change_1.shape[1]):
                if all_data_BP_change_1[i, j, k] > mean_array[1, k] + 3 * std_array[1, k] or all_data_BP_change_1[
                    i, j, k] < mean_array[1, k] - 3 * std_array[1, k]:
                    is_outlier_seq = 1
                    break
            for j in range(all_data_BP_change_2.shape[1]):
                if all_data_BP_change_2[i, j, k] > mean_array[2, k] + 3 * std_array[2, k] or all_data_BP_change_2[
                    i, j, k] < mean_array[2, k] - 3 * std_array[2, k]:
                    is_outlier_seq = 1
                    break
            for j in range(all_data_BP_change_3.shape[1]):
                if all_data_BP_change_3[i, j, k] > mean_array[3, k] + 3 * std_array[3, k] or all_data_BP_change_3[
                    i, j, k] < mean_array[3, k] - 3 * std_array[3, k]:
                    is_outlier_seq = 1
                    break
            if is_outlier_seq == 1:
                break
        if is_outlier_seq == 1:
            outlier_list.append(i)
    outlier_list = list(tuple(outlier_list))
    all_index_list = list(range(all_data.shape[0]))
    normal_filelist = [filelist[x] for x in all_index_list if x not in outlier_list]
    outlier_filelist = [filelist[x] for x in outlier_list]
    return normal_filelist, outlier_filelist


def preprocess(hparams, data_dir):
    data_source_dir = hparams.data_source_dir
    (all_dir, rest_filter, sport_filter, shenzhen_filter, summer_filter, day1_filter,
     day2_filter, day3_filter, day4_filter) = initialize_dir_filter()

    # group selection
    if hparams.source_group == 'all':
        dir_list = all_dir
    elif hparams.source_group == 'shenzhen':
        dir_list = [x for x in all_dir if x in shenzhen_filter]
    elif hparams.source_group == 'summer':
        dir_list = [x for x in all_dir if x in summer_filter]
    # status selection
    if hparams.source_status == 'rest':
        dir_list = [x for x in dir_list if x in rest_filter]
    elif hparams.source_status == 'sport':
        dir_list = [x for x in dir_list if x in sport_filter]

    day1_data_dir = [x for x in dir_list if x in day1_filter]
    day2_data_dir = [x for x in dir_list if x in day2_filter]
    day3_data_dir = [x for x in dir_list if x in day3_filter]
    day4_data_dir = [x for x in dir_list if x in day4_filter]
    all_data_dir = dir_list

    assert hparams.dataset_sheme in ['normal', 'refined', 'mix']
    if hparams.dataset_sheme in ['normal', 'refined']:
        # under these two sheme, the day1 data for train, eval and test, while day2, day3, day4 data only for test
        if hparams.dataset_sheme == 'refined':
            day1_data_dir = ['hongxi_day_1', 'summer_day_1_1']
        day1_filelist = get_filelist(data_source_dir, day1_data_dir)
        day2_filelist = get_filelist(data_source_dir, day2_data_dir)
        day3_filelist = get_filelist(data_source_dir, day3_data_dir)
        day4_filelist = get_filelist(data_source_dir, day4_data_dir)

        # train, eval and test set split

        if not hparams.split_by_person:
            day1_filelist_train, day1_filelist_eval, day1_filelist_test = \
                split_filelist(day1_filelist, hparams.random_seed, hparams.train_ratio, hparams.eval_ratio,
                               hparams.batch_size, split_by_person=False)
        else:
            day1_filelist_train, day1_filelist_eval, day1_filelist_test = \
                split_filelist(day1_filelist, hparams.random_seed, hparams.train_ratio, hparams.eval_ratio,
                               hparams.batch_size, split_by_person=True)
        # the filelist is randomed after split, sort the eval and test filelsit for better display afterwards when testing
        day1_filelist_eval = sort_filelist(day1_filelist_eval)
        day1_filelist_test = sort_filelist(day1_filelist_test)
        # load data
        day1_data_train = get_data(day1_filelist_train)
        day1_data_eval = get_data(day1_filelist_eval)
        day1_data_test = get_data(day1_filelist_test)
        if not hparams.day1_only:
            day2_filelist_test = sort_filelist(day2_filelist)
            day3_filelist_test = sort_filelist(day3_filelist)
            day4_filelist_test = sort_filelist(day4_filelist)
            day2_data_test = get_data(day2_filelist_test)
            day3_data_test = get_data(day3_filelist_test)
            day4_data_test = get_data(day4_filelist_test)

        # normalization
        # here the normalization is restricted in each sample for each channel
        # data_mean, data_std are both in the form of (sample_num, channel)
        assert hparams.src_tgt_feature_normalize_group in ['train', 'all'], 'normalize group not recognized'
        if hparams.src_tgt_feature_normalize_group == 'all':
            if not hparams.day1_only:
                all_data = np.concatenate(
                    (day1_data_train, day1_data_eval, day1_data_test, day2_data_test, day3_data_test, day4_data_test),
                    axis=0)
            else:
                all_data = np.concatenate((day1_data_train, day1_data_eval, day1_data_test), axis=0)
            data_mean = np.mean(np.mean(all_data, axis=0), axis=0)
            data_std = np.std(all_data, axis=(0, 1))
            if hparams.customized_waveform_normalize == True:
                curve_feature_mean = np.mean(all_data[:, :, :hparams.src_curve_feature_size])
                data_mean[:, :hparams.src_curve_feature_size] = curve_feature_mean
                curve_feature_std = np.std(all_data[:, :, :hparams.src_curve_feature_size])
                data_std[:, :hparams.src_curve_feature_size] = curve_feature_std
        elif hparams.src_tgt_feature_normalize_group == 'train':
            data_mean = np.mean(np.mean(day1_data_train, axis=0), axis=0)
            data_std = np.std(day1_data_train, axis=(0, 1))
            if hparams.customized_waveform_normalize == True:
                curve_feature_mean = np.mean(all_data_mean[:, :, :hparams.src_curve_feature_size])
                data_mean[:hparams.src_curve_feature_size] = curve_feature_mean
                curve_feature_std = np.std(all_data_mean[:, :, :hparams.src_curve_feature_size])
                data_std[:hparams.src_curve_feature_size] = curve_feature_std
        # data_soft_min, data_soft_max are the boundary, may used to carry the sof_minmax_normalization
        data_soft_min = data_mean[:hparams.src_feature_size + hparams.tgt_feature_size] - 3 * data_std[
                                                                                              :hparams.src_feature_size + hparams.tgt_feature_size]
        data_soft_max = data_mean[:hparams.src_feature_size + hparams.tgt_feature_size] + 3 * data_std[
                                                                                              :hparams.src_feature_size + hparams.tgt_feature_size]

        if hparams.src_feature_normalize_method == 'normal':
            for i in range(hparams.src_feature_size):
                day1_data_train[:, :, i] = ((day1_data_train[:, :, i] - data_mean[i]) / data_std[i])
                day1_data_eval[:, :, i] = ((day1_data_eval[:, :, i] - data_mean[i]) / data_std[i])
                day1_data_test[:, :, i] = ((day1_data_test[:, :, i] - data_mean[i]) / data_std[i])
                if not hparams.day1_only:
                    day2_data_test[:, :, i] = ((day2_data_test[:, :, i] - data_mean[i]) / data_std[i])
                    day3_data_test[:, :, i] = ((day3_data_test[:, :, i] - data_mean[i]) / data_std[i])
                    day4_data_test[:, :, i] = ((day4_data_test[:, :, i] - data_mean[i]) / data_std[i])
        if hparams.src_feature_normalize_method == 'soft_minmax':
            for i in range(hparams.src_feature_size):
                day1_data_train[:, :, i] = (
                        (day1_data_train[:, :, i] - data_soft_min[i]) / (data_soft_max[i] - data_soft_min[i]))
                day1_data_eval[:, :, i] = (
                        (day1_data_eval[:, :, i] - data_soft_min[i]) / (data_soft_max[i] - data_soft_min[i]))
                day1_data_test[:, :, i] = (
                        (day1_data_test[:, :, i] - data_soft_min[i]) / (data_soft_max[i] - data_soft_min[i]))
                if not hparams.day1_only:
                    day2_data_test[:, :, i] = (
                            (day2_data_test[:, :, i] - data_soft_min[i]) / (data_soft_max[i] - data_soft_min[i]))
                    day3_data_test[:, :, i] = (
                            (day3_data_test[:, :, i] - data_soft_min[i]) / (data_soft_max[i] - data_soft_min[i]))
                    day4_data_test[:, :, i] = (
                            (day4_data_test[:, :, i] - data_soft_min[i]) / (data_soft_max[i] - data_soft_min[i]))

        if hparams.tgt_feature_normalize_method == 'normal':
            for i in range(hparams.src_feature_size, hparams.src_feature_size + hparams.tgt_feature_size):
                day1_data_train[:, :, i] = ((day1_data_train[:, :, i] - data_mean[i]) / data_std[i])
                day1_data_eval[:, :, i] = ((day1_data_eval[:, :, i] - data_mean[i]) / data_std[i])
                day1_data_test[:, :, i] = ((day1_data_test[:, :, i] - data_mean[i]) / data_std[i])
                if not hparams.day1_only:
                    day2_data_test[:, :, i] = ((day2_data_test[:, :, i] - data_mean[i]) / data_std[i])
                    day3_data_test[:, :, i] = ((day3_data_test[:, :, i] - data_mean[i]) / data_std[i])
                    day4_data_test[:, :, i] = ((day4_data_test[:, :, i] - data_mean[i]) / data_std[i])
        if hparams.tgt_feature_normalize_method == 'soft_minmax':
            for i in range(hparams.src_feature_size, hparams.src_feature_size + hparams.tgt_feature_size):
                day1_data_train[:, :, i] = (
                        (day1_data_train[:, :, i] - data_soft_min[i]) / (data_soft_max[i] - data_soft_min[i]))
                day1_data_eval[:, :, i] = (
                        (day1_data_eval[:, :, i] - data_soft_min[i]) / (data_soft_max[i] - data_soft_min[i]))
                day1_data_test[:, :, i] = (
                        (day1_data_test[:, :, i] - data_soft_min[i]) / (data_soft_max[i] - data_soft_min[i]))
                if not hparams.day1_only:
                    day2_data_test[:, :, i] = (
                            (day2_data_test[:, :, i] - data_soft_min[i]) / (data_soft_max[i] - data_soft_min[i]))
                    day3_data_test[:, :, i] = (
                            (day3_data_test[:, :, i] - data_soft_min[i]) / (data_soft_max[i] - data_soft_min[i]))
                    day4_data_test[:, :, i] = (
                            (day4_data_test[:, :, i] - data_soft_min[i]) / (data_soft_max[i] - data_soft_min[i]))

        # here integrate 4 days test data (just for display)
        if not hparams.day1_only:
            day1_day2_day3_day4_data_test = np.concatenate(
                (day1_data_test, day2_data_test, day3_data_test, day4_data_test),
                axis=0)
            day1_day2_day3_day4_filelist_test = [*day1_filelist_test, *day2_filelist_test,
                                                 *day3_filelist_test, *day4_filelist_test]

        # store data shape
        with open(os.path.join(data_dir, 'data_shape.txt'), 'w') as data_shape_fo:
            data_shape_fo.write('%s.shape:%s\n' % ('day1_data_train', day1_data_train.shape))
            data_shape_fo.write('%s.shape:%s\n' % ('day1_data_eval', day1_data_eval.shape))
            data_shape_fo.write('%s.shape:%s\n' % ('day1_data_test', day1_data_test.shape))
            if not hparams.day1_only:
                data_shape_fo.write('%s.shape:%s\n' % ('day2_data_test', day2_data_test.shape))
                data_shape_fo.write('%s.shape:%s\n' % ('day3_data_test', day3_data_test.shape))
                data_shape_fo.write('%s.shape:%s\n' % ('day4_data_test', day4_data_test.shape))
                data_shape_fo.write(
                    '%s.shape:%s\n' % ('day1_day2_day3_day4_data_test', day1_day2_day3_day4_data_test.shape))

        # store mean and std, soft_min and soft_max
        if hparams.src_tgt_feature_normalize_group == 'all':
            scio.savemat(os.path.join(data_dir, 'all_data_mean.mat'), {'data': data_mean})
            scio.savemat(os.path.join(data_dir, 'all_data_std.mat'), {'data': data_std})
            scio.savemat(os.path.join(data_dir, 'all_data_soft_min.mat'), {'data': data_soft_min})
            scio.savemat(os.path.join(data_dir, 'all_data_soft_max.mat'), {'data': data_soft_max})
        elif hparams.src_tgt_feature_normalize_group == 'train':
            scio.savemat(os.path.join(data_dir, 'train_data_mean.mat'), {'data': data_mean})
            scio.savemat(os.path.join(data_dir, 'train_data_std.mat'), {'data': data_std})
            scio.savemat(os.path.join(data_dir, 'train_data_soft_min.mat'), {'data': data_soft_min})
            scio.savemat(os.path.join(data_dir, 'train_data_soft_max.mat'), {'data': data_soft_max})

        # store data
        train_data_dir = os.path.join(data_dir, 'train_data')
        if not tf.gfile.Exists(train_data_dir):
            tf.gfile.MakeDirs(train_data_dir)
        eval_data_dir = os.path.join(data_dir, 'eval_data')
        if not tf.gfile.Exists(eval_data_dir):
            tf.gfile.MakeDirs(eval_data_dir)
        test_data_dir = os.path.join(data_dir, 'test_data')
        if not tf.gfile.Exists(test_data_dir):
            tf.gfile.MakeDirs(test_data_dir)
        scio.savemat(os.path.join(train_data_dir, 'day1_data_train.mat'), {'data': day1_data_train})
        if day1_data_eval.shape[0] != 0:
            scio.savemat(os.path.join(eval_data_dir, 'day1_data_eval.mat'), {'data': day1_data_eval})
        if day1_data_test.shape[0] != 0:
            scio.savemat(os.path.join(test_data_dir, 'day1_data_test.mat'), {'data': day1_data_test})
        if not hparams.day1_only:
            scio.savemat(os.path.join(test_data_dir, 'day2_data_test.mat'), {'data': day2_data_test})
            scio.savemat(os.path.join(test_data_dir, 'day3_data_test.mat'), {'data': day3_data_test})
            scio.savemat(os.path.join(test_data_dir, 'day4_data_test.mat'), {'data': day4_data_test})
            scio.savemat(os.path.join(test_data_dir, 'day1_day2_day3_day4_data_test.mat'),
                         {'data': day1_day2_day3_day4_data_test})

        # store filelist
        store_filelist(train_data_dir, day1_filelist_train, 'day1_filelist_train')
        if day1_data_eval.shape[0] != 0:
            store_filelist(eval_data_dir, day1_filelist_eval, 'day1_filelist_eval')
        if day1_data_test.shape[0] != 0:
            store_filelist(test_data_dir, day1_filelist_test, 'day1_filelist_test')
        if not hparams.day1_only:
            store_filelist(test_data_dir, day2_filelist_test, 'day2_filelist_test')
            store_filelist(test_data_dir, day3_filelist_test, 'day3_filelist_test')
            store_filelist(test_data_dir, day4_filelist_test, 'day4_filelist_test')
            store_filelist(test_data_dir, day1_day2_day3_day4_filelist_test, 'day1_day2_day3_day4_filelist_test')
    elif hparams.dataset_sheme == 'mix':
        # under these sheme, the day1 ,day2, day3, day4 data are all for train and test
        all_filelist = get_filelist(data_source_dir, all_data_dir)
        if hparams.remove_BP_outlier:
            all_filelist, outlier_filelist = remove_BP_outlier(all_filelist, hparams)
        if hparams.split_by_person:
            if hparams.is_overlap == True:
                all_filelist_train, all_filelist_eval, all_filelist_test = \
                    split_overlap_filelist(all_filelist, hparams.random_seed, hparams.train_ratio, hparams.eval_ratio,
                                           hparams.batch_size, hparams.shift_sequence_num)
            else:
                all_filelist_train, all_filelist_eval, all_filelist_test = \
                    split_filelist(all_filelist, hparams.random_seed, hparams.train_ratio, hparams.eval_ratio,
                                   hparams.batch_size, split_by_person=True)
        else:  # here the not split_by_person but is_overlap part is not writtern
            all_filelist_train, all_filelist_eval, all_filelist_test = \
                split_filelist(all_filelist, hparams.random_seed, hparams.train_ratio, hparams.eval_ratio,
                               hparams.batch_size, split_by_person=False)
        # the filelist is randomed after split, sort the eval and test filelsit for better display afterwards when testing
        all_filelist_eval = sort_filelist(all_filelist_eval)
        all_filelist_test = sort_filelist(all_filelist_test)
        # load data
        all_data_train = get_data(all_filelist_train)
        all_data_eval = get_data(all_filelist_eval)
        all_data_test = get_data(all_filelist_test)
        if hparams.remove_BP_outlier:
            outlier_data = get_data(outlier_filelist)

        # normalization
        # here the normalization is restricted in each sample for each channel
        # data_mean, data_std are both in the form of (sample_num, channel)
        assert hparams.src_tgt_feature_normalize_group in ['train', 'all'], 'normalize group not recognized'
        if hparams.src_tgt_feature_normalize_group == 'all':
            all_data = np.concatenate((all_data_train, all_data_eval, all_data_test), axis=0)
            data_mean = np.mean(np.mean(all_data, axis=0), axis=0)
            data_std = np.std(all_data, axis=(0, 1))
            if hparams.customized_waveform_normalize == True:
                curve_feature_mean = np.mean(all_data[:, :, :hparams.src_curve_feature_size])
                data_mean[:hparams.src_curve_feature_size] = curve_feature_mean
                curve_feature_std = np.std(all_data[:, :, :hparams.src_curve_feature_size])
                data_std[:hparams.src_curve_feature_size] = curve_feature_std
        elif hparams.src_tgt_feature_normalize_group == 'train':
            data_mean = np.mean(np.mean(all_data_train, axis=0), axis=0)
            data_std = np.std(all_data_train, axis=(0, 1))
            if hparams.customized_waveform_normalize == True:
                curve_feature_mean = np.mean(all_data_train[:, :, :hparams.src_curve_feature_size])
                data_mean[:hparams.src_curve_feature_size] = curve_feature_mean
                curve_feature_std = np.std(all_data_train[:, :, :hparams.src_curve_feature_size])
                data_std[:hparams.src_curve_feature_size] = curve_feature_std
        # data_soft_min, data_soft_max are the boundary, may used to carry the sof_minmax_normalization
        data_soft_min = data_mean[:hparams.src_feature_size + hparams.tgt_feature_size] - 3 * data_std[
                                                                                              :hparams.src_feature_size + hparams.tgt_feature_size]
        data_soft_max = data_mean[:hparams.src_feature_size + hparams.tgt_feature_size] + 3 * data_std[
                                                                                              :hparams.src_feature_size + hparams.tgt_feature_size]
        if hparams.customized_waveform_normalize == True:
            curve_feature_mean = np.mean(data_mean[:hparams.src_curve_feature_size])
            data_mean[:hparams.src_feature_size]
        if hparams.src_feature_normalize_method == 'normal':
            for i in range(hparams.src_feature_size):
                all_data_train[:, :, i] = ((all_data_train[:, :, i] - data_mean[i]) / data_std[i])
                all_data_eval[:, :, i] = ((all_data_eval[:, :, i] - data_mean[i]) / data_std[i])
                all_data_test[:, :, i] = ((all_data_test[:, :, i] - data_mean[i]) / data_std[i])
        if hparams.tgt_feature_normalize_method == 'soft_minmax':
            for i in range(hparams.src_feature_size):
                all_data_train[:, :, i] = (
                        (all_data_train[:, :, i] - data_soft_min[i]) / (data_soft_max[i] - data_soft_min[i]))
                all_data_eval[:, :, i] = (
                        (all_data_eval[:, :, i] - data_soft_min[i]) / (data_soft_max[i] - data_soft_min[i]))
                all_data_test[:, :, i] = (
                        (all_data_test[:, :, i] - data_soft_min[i]) / (data_soft_max[i] - data_soft_min[i]))
        if hparams.tgt_feature_normalize_method == 'normal':
            for i in range(hparams.src_feature_size, hparams.src_feature_size + hparams.tgt_feature_size):
                all_data_train[:, :, i] = ((all_data_train[:, :, i] - data_mean[i]) / data_std[i])
                all_data_eval[:, :, i] = ((all_data_eval[:, :, i] - data_mean[i]) / data_std[i])
                all_data_test[:, :, i] = ((all_data_test[:, :, i] - data_mean[i]) / data_std[i])
        if hparams.tgt_feature_normalize_method == 'soft_minmax':
            for i in range(hparams.src_feature_size, hparams.src_feature_size + hparams.tgt_feature_size):
                all_data_train[:, :, i] = (
                        (all_data_train[:, :, i] - data_soft_min[i]) / (data_soft_max[i] - data_soft_min[i]))
                all_data_eval[:, :, i] = (
                        (all_data_eval[:, :, i] - data_soft_min[i]) / (data_soft_max[i] - data_soft_min[i]))
                all_data_test[:, :, i] = (
                        (all_data_test[:, :, i] - data_soft_min[i]) / (data_soft_max[i] - data_soft_min[i]))

        # store data shape
        with open(os.path.join(data_dir, 'data_shape.txt'), 'w') as data_shape_fo:
            data_shape_fo.write('%s.shape:%s\n' % ('all_data_train', all_data_train.shape))
            data_shape_fo.write('%s.shape:%s\n' % ('all_data_eval', all_data_eval.shape))
            data_shape_fo.write('%s.shape:%s\n' % ('all_data_test', all_data_test.shape))

        # store mean and std, soft_min and soft_max
        if hparams.src_tgt_feature_normalize_group == 'all':
            scio.savemat(os.path.join(data_dir, 'all_data_mean.mat'), {'data': data_mean})
            scio.savemat(os.path.join(data_dir, 'all_data_std.mat'), {'data': data_std})
            scio.savemat(os.path.join(data_dir, 'all_data_soft_min.mat'), {'data': data_soft_min})
            scio.savemat(os.path.join(data_dir, 'all_data_soft_max.mat'), {'data': data_soft_max})
        elif hparams.src_tgt_feature_normalize_group == 'train':
            scio.savemat(os.path.join(data_dir, 'train_data_mean.mat'), {'data': data_mean})
            scio.savemat(os.path.join(data_dir, 'train_data_std.mat'), {'data': data_std})
            scio.savemat(os.path.join(data_dir, 'train_data_soft_min.mat'), {'data': data_soft_min})
            scio.savemat(os.path.join(data_dir, 'train_data_soft_max.mat'), {'data': data_soft_max})

        # store data
        train_data_dir = os.path.join(data_dir, 'train_data')
        if not tf.gfile.Exists(train_data_dir):
            tf.gfile.MakeDirs(train_data_dir)
        eval_data_dir = os.path.join(data_dir, 'eval_data')
        if not tf.gfile.Exists(eval_data_dir):
            tf.gfile.MakeDirs(eval_data_dir)
        test_data_dir = os.path.join(data_dir, 'test_data')
        if not tf.gfile.Exists(test_data_dir):
            tf.gfile.MakeDirs(test_data_dir)
        scio.savemat(os.path.join(train_data_dir, 'all_data_train.mat'), {'data': all_data_train})
        if all_data_eval.shape[0] != 0:
            scio.savemat(os.path.join(eval_data_dir, 'all_data_eval.mat'), {'data': all_data_eval})
        if all_data_test.shape[0] != 0:
            scio.savemat(os.path.join(test_data_dir, 'all_data_test.mat'), {'data': all_data_test})
        if hparams.remove_BP_outlier:
            if outlier_data.shape[0] != 0:
                scio.savemat(os.path.join(data_dir, 'outlier_data.mat'), {'data': outlier_data})
        # store filelist
        store_filelist(train_data_dir, all_filelist_train, 'all_filelist_train')
        if all_data_eval.shape[0] != 0:
            store_filelist(eval_data_dir, all_filelist_eval, 'all_filelist_eval')
        if all_data_test.shape[0] != 0:
            store_filelist(test_data_dir, all_filelist_test, 'all_filelist_test')
        if hparams.remove_BP_outlier:
            if outlier_data.shape[0] != 0:
                store_filelist(data_dir, outlier_filelist, 'outlier_filelist')
