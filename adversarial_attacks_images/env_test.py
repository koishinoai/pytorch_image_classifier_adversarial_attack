import torch
import torchvision
import torchvision.transforms as transforms
from models import *
from copy import deepcopy
import cv2
import numpy as np
import matplotlib.pyplot as plt
# prepare raw data
from common.PSNR import PSNR
from common.SSIM import SSIM

transform_train = transforms.Compose([
    # noise train datasets
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # normalize each channel value .Normalize([channel1.mean, ..., ...], [channel1.std, ..., ...])
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

BATCH = 100
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=False, num_workers=2, drop_last = True)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False, num_workers=2, drop_last = True)
class_ifiar = [['vgg19']]

def load_classifiar(classifiar_name = 'vgg19'):
    if classifiar_name not in class_ifiar:
        TypeError('classifiar not exist')
    if classifiar_name == 'vgg19':
        net = torch.load('../checkpoint/vgg19_net.pkl')
    return net

class Env():
    def __init__(self, train_dataset = trainloader, test_dataset = testloader, mode = 'train', classifier_name = 'vgg19'):
        self.train_data = list(train_dataset)
        self.test_data = list(test_dataset)
        self._data_mode = mode
        self.classifier = load_classifiar(classifier_name)

    def reset(self):
        if self._data_mode == 'train':
            batch_index = np.random.randint(low=0,high=len(self.train_data))
            extract_index = np.random.randint(low=0,high=BATCH)
            self._state, self._ground_truth_label = self.train_data[batch_index]
            self._ground_truth_label = self._ground_truth_label[extract_index]
            self._state = self._state[extract_index]
            self.ground_truth_image = deepcopy(self._state)
            return self._state
        elif self._data_mode == 'test':
            batch_index = np.random.randint(low=0,high=len(self.test_data))
            extract_index = np.random.randint(low=0,high=BATCH)
            self._state,self._ground_truth_label = self.test_data[batch_index]
            self._ground_truth_label = self._ground_truth_label[extract_index]
            self._state = self._state[extract_index]
            self.ground_truth_image = deepcopy(self._state)
        else:
            ValueError('mode not exist')
        # return random minibatch

    def step(self, action):
        done = 0
        hight_index = action // self._state.size(1)
        width_index = action % self._state.size(1)
        new_state = self._state

        new_state[0,hight_index-1,width_index-1] = 1
        new_state[1, hight_index-1, width_index-1] = 1
        new_state[2, hight_index-1, width_index-1] = 1

        _, new_label = self.classifier(new_state.unsqueeze(0)).max(1)
        new_label = new_label.detach().cpu().data[0]
        disjudge_reward = 0
        if not bool(new_label == self._ground_truth_label):
            disjudge_reward = 1
            done = 1
        psnr_reward = PSNR(self.ground_truth_image, new_state, MAX_PIXEL_VALUE=1) - 40
        if bool(psnr_reward > 100):
            psnr_reward = 0


        reward = disjudge_reward + psnr_reward * 0.1
        self._state = new_state

        return new_state, reward, done

    def render(self):
        pass

def save_image(image,filename):
    if image.size(0) == 3:
        temp = transforms.ToPILImage()(image).convert('RGB')
        temp.save('./save/{}.png'.format(filename))
        return
    for i in range(image.size(0)):
        temp = transforms.ToPILImage()(image[i]).convert('RGB')
        temp.save('./save/{}{}.png'.format(filename,i))








if __name__ == '__main__':
    env = Env()
    state = env.reset()
    psnr = PSNR(state,state,MAX_PIXEL_VALUE=1)




