import torch
import torchvision
import torchvision.transforms as transforms

import cv2
import numpy as np
import matplotlib.pyplot as plt
# prepare raw data

transform_train = transforms.Compose([
    # noise train datasets
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True, num_workers=2, drop_last = True)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False, num_workers=2, drop_last = True)

class Env():
    def __init__(self, train_dataset = trainloader, test_dataset = testloader, mode = 'train', classifier = None):
        self.train_data = list(train_dataset)
        self.test_data = list(test_dataset)
        self._data_mode = mode
        if classifier == None:
            TypeError('classifier not exist')
        self.classifier = classifier

    def reset(self):
        if self._data_mode == 'train':
            batch_index = np.random.randint(low=0,high=len(self.train_data))
            extract_index = np.random.randint(low=0,high=BATCH)
            self._state,_= self.train_data[extract_index]
            self._state = self._state[0]
        elif self._data_mode == 'test':
            batch_index = np.random.randint(low=0, high=len(self.train_data))
            extract_index = np.random.randint(low=0, high=BATCH)
            self._state,_= self.train_data[extract_index]
            self._state = self._state[0]
        else:
            ValueError('mode not exist')
        # return random minibatch
        return self._state

    def step(self, action):
        hight_index = action // self._state.size(1)
        width_index = action % self._state.size(1)
        new_state = self._state
        new_state[0,hight_index,width_index] = 1
        new_state[1, hight_index, width_index] = 1
        new_state[2, hight_index, width_index] = 1
        # compute reward
        self._state = new_state
        # return state, reward, done, info

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
    data = env.reset()
    data = env.step(32*16 + 16)
    # image = data[0].unsqueeze(0)
    # save_image(image, 'test')
    # print(image.shape)
    # image[0,0,16,16] = 1
    # image[0,1, 16, 16] = 1
    # image[0,2, 16, 16] = 1
    # save_image(image,'test_raw')
    print(data.shape)
    print(env._state.shape)
    save_image(data,'test')
