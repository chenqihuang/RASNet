# -*- coding:utf-8 -*-
# !/ussr/bin/env python2
__author__ = "QiHuangChen"

class Config:
    def __init__(self):

        self.pos_pair_range = 100
        self.num_pairs = 5.32e4
        self.val_ratio = 0.1
        self.num_epoch = 70
        self.batch_size = 8
        self.examplar_size = 127
        self.instance_size = 255
        self.sub_mean = 0
        self.train_num_workers = 12  # number of threads to load data when training
        self.val_num_workers = 8
        self.stride = 8
        self.rPos = 16
        self.rNeg = 0
        self.label_weight_method = "balanced"

        self.lr = 1e-2               # learning rate of SGD
        self.momentum = 0.9          # momentum of SGD
        self.weight_decay = 5e-4     # weight decay of optimizator
        self.step_size = 1           # step size of LR_Schedular
        self.gamma = 0.8685          # decay rate of LR_Schedular
        self.model_save_path = " "