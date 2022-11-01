import time
import numpy as np
from PIL import Image
import mindspore
from mindspore import load_checkpoint, load_param_into_net, Model
import mindspore.ops as ops
import os
from src import build_dataset, build_dataloader, ordered_yaml, build_net, gse_in, gse_pr

import yaml
with open(os.path.join(os.path.abspath('.'), 'config/test.yml'), mode='r') as f_yml:
    Loader, _ = ordered_yaml()
    opt = yaml.load(f_yml, Loader=Loader)

######## load checkpoint #########
network = build_net(opt['MODEL_NAME'])
param_dict = load_checkpoint(opt['PRETRAINED'])
load_param_into_net(network, param_dict)
model = Model(network)

####### 定义数据集  ########
valset = build_dataset(**opt['DATASET'])
dataloader = build_dataloader(valset, **opt['DATALOADER'])
dataloader = dataloader.create_dict_iterator()

psnr_list = []
time_list = []

#####
PSNR = mindspore.nn.PSNR()

for iter_idx, data in enumerate(dataloader,1):
    
    input_img = data["input"]
    label_img = data["label"]
    
    ##### warm up #####
    if iter_idx == 1 :
        pre = model.predict(input_img)[2]

    tm = time.time()
    inp = gse_in(input_img)           
    pre_ = []
    for idx,i in enumerate(inp):
        pre_.append(model.predict(i)[2])
    pre = gse_pr(pre_)
    elapsed = time.time() - tm

    time_list.append(elapsed)

    pred_clip = ops.clip_by_value(pre, 0, 1)

    psnr = PSNR(pred_clip, label_img).asnumpy().item()
    
    psnr_list.append(psnr)
    print('%d iter PSNR: %.2f time: %f' % (iter_idx, psnr, elapsed))
    
    if opt["SAVE_IMG"] and opt["SAVED_PATH"] is not None:
        if not os.path.exists(opt["SAVED_PATH"]):
            os.makedirs(opt["SAVED_PATH"])
        pred_clip += 0.5 / 255
        pred_numpy = (pred_clip.asnumpy().squeeze().transpose(1, 2, 0)*255).astype(np.uint8)
        pred_img = Image.fromarray(pred_numpy, 'RGB')
        pred_img.save(os.path.join(opt["SAVED_PATH"], f"{iter_idx}_pred.png"))
        
total_images = len(psnr_list)
print('==========================================================')
print('The total images are %d' % total_images)
print('The average PSNR is %.2f dB' % (sum(psnr_list)/total_images))
print("Average time: %f" % (sum(time_list)/total_images))
