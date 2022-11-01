import mindspore.nn as nn
from mindspore import Model, load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.common import set_seed
import random
import numpy as np
import os
from src import LossMonitor, ordered_yaml, build_dataset, build_dataloader, build_net, total_loss, CustomWithLossCell, preprocess_dataset, TrainClipGrad

######### argparse #######
import argparse
parser = argparse.ArgumentParser()
# Directories
parser.add_argument('--data_url', default=None, type=str)
parser.add_argument('--train_url', default=None, type=str)
parser.add_argument('--device_target', default='Ascend', type=str)
parser.add_argument('--root_src',default=None,type=str)
args = parser.parse_args()

if args.data_url is not None and args.train_url is not None:
    ######### 云脑训练 #########
    from src.ObsAndEnv import ObsToEnv, EnvToObs
    workroot = '/home/ma-user/modelarts/user-job-dir'
    data_dir = workroot + '/data' 
    train_dir = workroot + '/model'
    coderoot = workroot + '/code'
    ObsToEnv(args.data_url,data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
else:
    coderoot = os.path.abspath('.')
    data_dir = args.root_src
    train_dir = coderoot + '/output'

########## yaml ###################
import yaml
with open(os.path.join(coderoot,'config/train.yml'), mode='r') as f_yml:
    Loader, _ = ordered_yaml()
    opt = yaml.load(f_yml, Loader=Loader)

####### preprocess dataset #####
if not os.path.exists(opt['DATASET']['path']):
    preprocess_dataset(root_src = data_dir, root_dst = opt['DATASET']['path'])

########## seed setting ##########
random.seed(1)
set_seed(1)
np.random.seed(1)

########## build net and resume  ########
network = build_net(opt['MODEL_NAME'])
network.set_train(True)
if opt['TRAIN']['RESUME'] is not None:
    param_dict = load_checkpoint(opt['TRAIN']['RESUME'])
    load_param_into_net(network, param_dict)

########## dataset  ########
trainset = build_dataset(**opt['DATASET'])
dataloader = build_dataloader(trainset, **opt['DATALOADER'])
trainset_len = len(trainset)
steps_per_epoch = trainset_len // opt['DATALOADER']['batch_size'] if trainset_len % opt['DATALOADER']['batch_size'] == 0 else trainset_len // opt['DATALOADER']['batch_size'] + 1

##########  loss schedular optimizer #######
loss = total_loss()
loss_scale_manager = nn.wrap.loss_scale.DynamicLossScaleUpdateCell(2**24, 2, steps_per_epoch)

opt['SCHE']['step_per_epoch'] = steps_per_epoch
opt['SCHE']['total_step'] = opt['TRAIN']['NUMEPOCH'] * steps_per_epoch
learning_rate = nn.exponential_decay_lr(**opt['SCHE'])
optimizer = nn.Adam(params=network.trainable_params(), learning_rate=learning_rate, **opt['OPTIM'])

########## Callback ######
# ckpt
config_ck = CheckpointConfig(saved_network=network, **opt['CALLBACK']['ckpt_cfg'])
ckpt_cb = ModelCheckpoint(prefix="mimo-unet", directory = train_dir, config=config_ck)
# loss
loss_cb = LossMonitor(learning_rate, opt['CALLBACK']['print_freq'])

########## model train  ##########
network_with_loss = CustomWithLossCell(network, loss)
network_with_loss = TrainClipGrad(network_with_loss, optimizer, scale_sense=loss_scale_manager, clip_global_norm_value=0.01)
model = Model(network_with_loss)
model.train(opt['TRAIN']['NUMEPOCH'], dataloader, callbacks=[loss_cb,ckpt_cb], dataset_sink_mode=False) 

if args.data_url is not None and args.train_url is not None:
    EnvToObs(train_dir, args.train_url)