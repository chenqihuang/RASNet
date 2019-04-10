# -*- coding:utf-8 -*-
# !/ussr/bin/env python2
__author__ = "QiHuangChen"


import time
from train_config import *
from data.VIDDataset import *
from torch.utils.data import DataLoader
from models.RASNet import *
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from utils.create_label import *
import torchvision.transforms as transforms
from data.DataAugmentation import *

config = Config()

def train(data_dir, train_imdb):


    center_crop_size = config.instance_size - config.stride
    random_crop_size = config.instance_size - 2 * config.stride
    train_z_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((config.examplar_size, config.examplar_size)),
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((center_crop_size, center_crop_size)),
        RandomCrop((random_crop_size, random_crop_size)),
        ToTensor()
    ])
    train_dataset = RASNet_VIDDataset(train_imdb, data_dir, config, train_z_transforms, train_x_transforms)
    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.train_num_workers, drop_last=True)

    net = RASNet()
    net.cuda()

    optimizer = torch.optim.SGD(net.parameters(), config.lr, config.momentum, config.weight_decay)
    scheduler = StepLR(optimizer, config.step_size, config.gamma)



    for i in range(config.num_epoch):
        scheduler.step()
        net.train()

        train_loss = []
        for j, data in enumerate(train_loader):
            exemplar_imgs, instance_imgs, penalty = data
            exemplar_imgs = exemplar_imgs.cuda()
            instance_imgs = instance_imgs.cuda()
            output = net.forward(Variable(exemplar_imgs), Variable(instance_imgs))

            train_response_flag = True
            response_size = output.shape[2:4] # 灵活使用，根据最终输出得到response map的大小，构造label的大小。
            train_eltwise_label, train_instance_weight = create_label(response_size, config, use_gpu)

            loss = net.weight_loss(output, train_eltwise_label, train_instance_weight, penalty)

            loss.backward()
            optimizer.step()

            loss_train = loss.to('cpu').squeeze().data.numpy()
            train_loss.append(loss_train)


        # 保存模型
        if not os.path.exists(config.model_save_path):
            os.makedirs(config.model_save_path)
        torch.save(net.state_dict(), config.model_save_path + "SiamFC_dict_" + str(i + 1) + "_model.pth")

        print ("Epoch %d   training loss: %f" % (i + 1, np.mean(train_loss)))


if __name__ == "__main__":
    data_dir = " "
    train_imdb = " "
    val_imdb = " "
    model_save_path = ""

    print time.strftime('start %Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    train(data_dir, train_imdb)
    print time.strftime('end %Y-%m-%d %H:%M:%S', time.localtime(time.time()))

