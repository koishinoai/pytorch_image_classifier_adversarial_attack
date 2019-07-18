import torch.nn as nn
import torch
import numpy as np

def PSNR(image1, image2,MAX_PIXEL_VALUE):
    loss = nn.MSELoss()
    # image[1/2] type : tensor
    if image1.shape != image2.shape:
        print('images shape do not match')
    return 10*np.log10(MAX_PIXEL_VALUE**2 / loss(image1, image2))

if __name__ == '__main__':
    data1 = torch.randn(size=(1,1,256,256))
    data2 = torch.randn(1,1,256,256)
    print(PSNR(data1,data2,MAX_PIXEL_VALUE=255))

