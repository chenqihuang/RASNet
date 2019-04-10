import torch
import numpy as np
def create_logisticloss_label(label_size, rPos, rNeg):
    """
    construct label for logistic loss (same for all pairs)
    """
    label_side = int(label_size[0])
    logloss_label = torch.zeros(label_side, label_side)
    label_origin = np.array([np.ceil(label_side / 2), np.ceil(label_side / 2)])
    for i in range(label_side):
        for j in range(label_side):
            dist_from_origin = np.sqrt((i - label_origin[0]) ** 2 + (j - label_origin[1]) ** 2)
            # print dist_from_origin
            if dist_from_origin <= rPos:
                logloss_label[i, j] = +1
            else:
                if dist_from_origin <= rNeg:
                    logloss_label[i, j] = 0

    return logloss_label


def create_label(fixed_label_size, config, use_gpu):
    """
    create label with weight
    """
    rPos = config.rPos / config.stride
    rNeg = config.rNeg / config.stride

    half = int(np.floor(fixed_label_size[0] / 2) + 1)

    if config.label_weight_method == "balanced":
        fixed_label = create_logisticloss_label(fixed_label_size, rPos, rNeg)


        instance_weight = torch.ones(fixed_label.shape[0], fixed_label.shape[1])
        tmp_idx_P = np.where(fixed_label == 1)
        sumP = tmp_idx_P[0].size
        tmp_idx_N = np.where(fixed_label == 0)
        sumN = tmp_idx_N[0].size
        instance_weight[tmp_idx_P] = 0.5 * instance_weight[tmp_idx_P] / sumP
        instance_weight[tmp_idx_N] = 0.5 * instance_weight[tmp_idx_N] / sumN
      
        fixed_label = torch.reshape(fixed_label, (1, 1, fixed_label.shape[0], fixed_label.shape[1]))
        # copy label to match batchsize
        fixed_label = fixed_label.repeat(config.batch_size, 1, 1, 1)

        # reshape weight
        instance_weight = torch.reshape(instance_weight, (1, instance_weight.shape[0], instance_weight.shape[1]))

    if use_gpu:
        return fixed_label.cuda(), instance_weight.cuda()
    else:
        return fixed_label, instance_weight
