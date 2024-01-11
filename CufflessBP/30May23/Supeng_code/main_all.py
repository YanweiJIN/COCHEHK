from __future__ import print_function

import argparse
import os
import codecs
import json

import tensorflow as tf

import utils
import process
import preprocess
#import judge


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # network
    parser.add_argument("--architecture", type=str, default="peng", help="'peng's rnn'")
    ## layer(for peng's model)
    parser.add_argument("--num_bi_layers", type=int, default=1,
                        help="number of bidirectional layers, 1 layer acually represent 2 directions")
    parser.add_argument("--num_uni_layers", type=int, default=1,
                        help="number of uni layers stacked on the bi layers, "
                             "with uni_residual_layer=uni_layers-1 setting as default .")
    parser.add_argument("--projector_type", type=str, default='linear',
                        help="the type of the last layer,"
                             "should be one of ['sigmoid', 'linear']")
    ## cell
    parser.add_argument("--unit_type", type=str, default="lstm", help="lstm")
    parser.add_argument("--num_units", type=int, default=128, help="the hidden size of each cell.")
    parser.add_argument("--forget_bias", type=float, default=1.0,
                        help="Forget bias for BasicLSTMCell.")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout rate (not keep_prob)")
    ## initializer (uniform initializer as default)
    parser.add_argument("--init_weight", type=float, default=0.1,
                        help=("for uniform init_op, initialize weights "
                              "between [-this, this]."))
    ## optimizer
    parser.add_argument("--optimizer", type=str, default="adam", help="the type of optimizer")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate. Adam: 0.001 | 0.0001")

    # data
    ## data source
    parser.add_argument('--data_source_dir', type=str,
                        default='/media/darcy/Documents/code/shenzhen_task_6_mine_all/TQWT_filtered_feature_data_Api_2nd_feature2BP',
                        help='the path to load original data for train data extraction')
    parser.add_argument('--source_group', type=str, default='all',
                        help='''the standard to select data for training and testing,
                             should be one of 'all','shenzhen','summer' ''')
    parser.add_argument('--source_status', type=str, default='rest',
                        help='''the standard to select data for training and testing,
                             should be one of 'rest','sport' ''')
    parser.add_argument('--dataset_sheme', type=str, default='mix',
                        help='''the sheme for split of dataset into train, eval, test set, 
                             should be one of ['normal', 'mix', 'refined'],
                             'normal' mean all day 1 data to be train ,eval, test set and day 2,3,4 to be test set,
                             'refined' is similar to 'normal', except that day 1 data refined to hongxi and summer_day1_1
                             'mix' mean mix all data, shuffle and divide into train eval and test set,
                             'mix' option not available yet''')
    parser.add_argument('--day1_only', type="bool", default=False,
                        help='''if true, then we only have data of data 1 and disregard the multiday effect' ''')
    parser.add_argument('--split_by_person', type="bool", default=True,
                        help='''if true, the split of train, eval, test set is according to person''')
    parser.add_argument('--is_overlap', type="bool", default=False,
                        help='''if true, the record is overlap partly''')
    parser.add_argument("--shift_sequence_num", type=int, default=3,
                        help="number of sequences that should be discarded to avoid the common data in both training and testing set,"
                             "only applied when --is_overlap==True")
    ## data normalization
    ## should note that the normalization of data according to person is carred out (if have) by matlab preprocessing program
    parser.add_argument('--remove_BP_outlier', type="bool", default=False,
                        help="if true, some record with BP outlier is removed, and be put int o outlier data")
    parser.add_argument('--customized_waveform_normalize', type="bool", default=False,
                            help="for waveform feature, the normalize is kind of different,"
                                 "because the so-called feature is actually a curve,"
                                 "and they should be scale down and back with same factor")
    parser.add_argument('--src_feature_normalize_method', type=str, default='normal',
                        help="""the normalization method of src feature
                                should be one of ['normal', 'none'],
                                'normal' mean normalize into the form of (0,1) normal distribution,
                                'none' mean don't do anything""")
    parser.add_argument('--tgt_feature_normalize_method', type=str, default='normal',
                        help="""the normalization method of tgt feature, namely SBP, DBP, MBP
                            should be one of ['soft_minmax', 'normal', 'none'],
                            'soft_minmax' mean normalize into the form of [0,1] range by min and max value, 
                            here the min and max are mean+(-)3std respectively,
                            'normal' mean normalize into the form of (0,1) normal distribution,
                            'none' mean don't do anything
                            """)
    parser.add_argument('--src_tgt_feature_normalize_group', type=str, default='train',
                        help="""the data base for calculating mean and std and scaling back,
                               should be one of ['all', 'train'],
                               'all' mean the calculation is based on all data,
                               'train' mean the calculation is based on only traning set data,
                                in practice, only 'train' principle appies,
                                while in research, we may use 'all' principle""")
    parser.add_argument('--remove_outlier_from_loss_in_test', type="bool", default=False,
                        help="""if true, the point with ground truth BP that is not in the reasonable range,
                            (here the [mean(BP)-3*std(BP), mean(BP) +3*stf(BP)]),
                            will be removed from calculation of loss,
                            this only apply to test set by now""")
    ## data detail
    parser.add_argument("--src_len", type=int, default=16,
                        help="length of src sequences during training. "
                             "the input are all in same length")
    parser.add_argument("--tgt_len", type=int, default=16,
                        help="length of tgt sequences during training.")
    parser.add_argument("--src_feature_size", type=int, default=8,
                        help="feature size of src")
    parser.add_argument("--tgt_feature_size", type=int, default=3,
                        help="feature size of tgt")
    parser.add_argument("--src_curve_feature_size", type=int, default=512,
                        help="feature size of curve in the src, should be normalized as whole,"
                             "only applied when --customized_waveform_normalize==True" )


    # expriment
    parser.add_argument("--exprs_dir", type=str,
                        default='/media/darcy/Documents/code/shenzhen_task_6_mine_all/experiments',
                        help="the dir to store experiment.")
    parser.add_argument("--expr_name", type=str,
                        default='2019_4_14_16len_my_8_feature_based_1_bi_1_uni_lstm_personally_mix_experiment',
                        help='the name of dir to store config, model, data, figure, result')
    parser.add_argument('--train_ratio', default=0.8, type=float,
                        help='the ratio of data for trainning')
    parser.add_argument('--eval_ratio', default=0.1, type=float,
                        help='the ratio of data for validation')
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--random_seed", type=int, default=1234,
                        help="Random seed (>0, set a specific seed).")
    parser.add_argument("--num_train_steps", type=int, default=100000, help="Num steps to train.")
    parser.add_argument('--backward_enable', default=False, type="bool",
                        help="if enable the backward training")
    parser.add_argument('--forward_directions', default=["src2tgt", "tgt2src"], type=list,
                        help="the directions for forward phase of traning"
                             " can only be 'src2tgt', 'tgt2src', 'src2src', 'tgt2tgt'")
    parser.add_argument('--backward_directions', default=["src2tgt2src", "tgt2src2tgt"], type=list,
                        help="the directions for backward phase of traning"
                             " can only be 'src2tgt2src', 'tgt2src2tgt'")
    parser.add_argument("--steps_per_stats", type=int, default=1,
                        help=("How many training steps to do per stats logging."
                              "Save checkpoint every epoch"))
    parser.add_argument("--steps_per_test", type=int, default=2500,
                        help=("How many training steps to do per testing."
                              "before each time test will also save the checkpoint"))

def create_dirs(hparams):
    """create directories"""
    assert hparams.exprs_dir
    exprs_dir = hparams.exprs_dir
    if not tf.gfile.Exists(exprs_dir):
        tf.gfile.MakeDirs(exprs_dir)
    utils.print_out("# experiments directory: %s " % exprs_dir)
    expr_dir = os.path.join(exprs_dir, hparams.expr_name)
    if not tf.gfile.Exists(expr_dir):
        tf.gfile.MakeDirs(expr_dir)
    utils.print_out("# -experiment directory: %s " % expr_dir)
    config_dir = os.path.join(expr_dir, 'config')
    if not tf.gfile.Exists(config_dir):
        tf.gfile.MakeDirs(config_dir)
    utils.print_out("# --config directory: %s " % config_dir)
    log_dir = os.path.join(expr_dir, 'log')
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
    utils.print_out("# --log directory: %s " % log_dir)
    data_dir = os.path.join(expr_dir, 'data')
    if not tf.gfile.Exists(data_dir):
        tf.gfile.MakeDirs(data_dir)
    utils.print_out("# --data directory: %s " % data_dir)
    model_dir = os.path.join(expr_dir, 'model')
    if not tf.gfile.Exists(model_dir):
        tf.gfile.MakeDirs(model_dir)
    utils.print_out("# --model directory: %s " % model_dir)
    figure_dir = os.path.join(expr_dir, 'figure')
    if not tf.gfile.Exists(figure_dir):
        tf.gfile.MakeDirs(figure_dir)
    utils.print_out("# --figure directory: %s " % figure_dir)
    result_dir = os.path.join(expr_dir, 'result')
    if not tf.gfile.Exists(result_dir):
        tf.gfile.MakeDirs(result_dir)
    utils.print_out("# --result directory: %s " % result_dir)
    return expr_dir, config_dir, log_dir, data_dir, model_dir, figure_dir, result_dir


def check_and_save_hparams(out_dir, hparams):
    """Save hparams."""
    hparams_file = os.path.join(out_dir, "hparams")
    if tf.gfile.Exists(hparams_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
            origin_hparams = json.load(f)
            origin_hparams = tf.contrib.training.HParams(**origin_hparams)
        wrong_keys = []
        keys = set(list(hparams.values().keys()) + (list(origin_hparams.values().keys())))
        for key in keys:
            if (hparams.values().get(key, None) != origin_hparams.values().get(key, None) or
                    hparams.values().get(key, None) == None or
                    hparams.values().get(key, None) == None):
                wrong_keys.append(key)
        try:
            assert origin_hparams.values() == hparams.values()
            utils.print_out("using the same hparams of old %s" % hparams_file)
        except:
            utils.print_out('new hparams not the same with the existed one')
            for wrong_key in wrong_keys:
                utils.print_out(" keys: %s, \norigin_value: %s, \nnew_value: %s\n" % (
                    wrong_key, origin_hparams.values()[wrong_key], hparams.values()[wrong_key]))
            raise ValueError
    else:
        utils.print_out("  not old hparams found, create new hparams to %s" % hparams_file)
        with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
            f.write(hparams.to_json(indent=4))


def main():
    parser = argparse.ArgumentParser(description='peng lstm')
    add_arguments(parser)
    args = parser.parse_args()
    print(args)
    hparams = tf.contrib.training.HParams(**vars(args))
    # check GPU device
    utils.print_out("# Devices visible to TensorFlow: %s" % repr(tf.Session().list_devices()))
    #  create dirs
    expr_dir, config_dir, log_dir, data_dir, model_dir, figure_dir, result_dir = create_dirs(hparams)
    # save hyperameter
    check_and_save_hparams(config_dir, hparams)

    stage = 'train_eval_test'  # preprocess','train_eval', 'test', or 'judge'
    assert stage in ['preprocess', 'train_eval', 'test', 'train_eval_test'], 'stage not recognized'
    utils.print_out('stage: %s' % stage)
    if stage == 'preprocess':
        preprocess.preprocess(hparams, data_dir)
        # the data are stored in the data_dir for the training step
    if stage == 'train_eval':
        process.train_eval(hparams, data_dir, model_dir, log_dir)
    if stage == 'test':
        process.infer(hparams, data_dir, model_dir, result_dir)
    if stage == "train_eval_test":
        while (1):
            global_step=process.train_eval(hparams, data_dir, model_dir, log_dir)
            # at certain step, the program will return
            if global_step<hparams.num_train_steps:
                process.infer(hparams, data_dir, model_dir, result_dir)
    if stage=='judge':
        judge.judge(hparams, data_dir)


if __name__ == '__main__':
    main()
