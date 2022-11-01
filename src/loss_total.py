import mindspore.nn as nn
import mindspore.ops as ops

class total_loss(nn.LossBase):
    def __init__(self):
        super(total_loss, self).__init__()
        self.L1_loss = nn.L1Loss()
    
    def interpolate_down(self, input, scale_factor):
        _, _, h, w = input.shape
        h = h // scale_factor
        w = w // scale_factor
        resize = ops.ResizeNearestNeighbor((h, w))
        return resize(input)
        
    def construct(self, pred_img, label_img):
        '''construct_loss'''
        label_img_2 = self.interpolate_down(label_img, scale_factor=2)
        label_img_4 = self.interpolate_down(label_img, scale_factor=4)

        l1 = self.L1_loss(pred_img[0], label_img_4)
        l2 = self.L1_loss(pred_img[1], label_img_2)
        l3 = self.L1_loss(pred_img[2], label_img)
        return l1+l2+l3
    
class CustomWithLossCell(nn.Cell):
    def __init__(self, network, loss):
        super(CustomWithLossCell, self).__init__()
        self.network = network
        self.loss = loss

    def construct(self, input_img, label_img):
        pred_img = self.network(input_img)              
        return self.loss(pred_img, label_img)  
