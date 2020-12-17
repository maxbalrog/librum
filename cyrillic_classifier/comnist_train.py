import random
from random import shuffle

import cv2

from skimage.feature import hog

import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import tensorflow as tf

import torchvision

RANDOM_SEED = 112123

def fix_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class comnist_dataset(Dataset):
    def __init__(self, image_path, labels):
        self.image_path = glob.glob(image_path)
        shuffle(self.image_path)

        self.labels = {letter: i for i,letter in enumerate(labels)}

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        letter = img_path.split(os.sep)[-2]
        label = self.labels[letter]

        img = cv2.imread(img_path, -1)

        alpha = img[:,:,3]
        imgray = ~alpha
        imgray = 255 - imgray

        imgray_res = cv2.resize(imgray, (64,64))

        imgray_res = imgray_res.reshape((1,64,64))

        return (imgray_res, label)

class comnist_dataset_hog(Dataset):
    def __init__(self, image_path, labels, cell=(8,8)):
        self.image_path = glob.glob(image_path)
        shuffle(self.image_path)

        self.labels = {letter: i for i,letter in enumerate(labels)}

        self.cell = cell

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        letter = img_path.split(os.sep)[-2]
        label = self.labels[letter]

        img = cv2.imread(img_path, -1)

        alpha = img[:,:,3]
        imgray = ~alpha
        imgray = 255 - imgray

        imgray_res = cv2.resize(imgray, (64,64))

        fd, hog_image = hog(imgray_res, orientations=8, pixels_per_cell=self.cell,
                            cells_per_block=(1, 1), visualize=True, multichannel=False)

        hog_image = hog_image.reshape((1,64,64))

        return (hog_image, label)

def setup_experiment_name():
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    return train_summary_writer, test_summary_writer

class Model(nn.Module):
    def __init__(self, n_classes=34, n_filters=15):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, n_filters, 5, padding=0)
        self.conv2 = nn.Conv2d(n_filters, n_filters*2, 5, padding=0)
        self.conv3 = nn.Conv2d(n_filters*2, n_filters*4, 5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(n_filters*4 * 4 * 4, 160)
        self.fc2 = nn.Linear(160, 100)
        self.fc3 = nn.Linear(100, n_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def train(save_path='comnist_cls.pth'):

    fix_seed(RANDOM_SEED)

    batch_size = 100

    labels = '| а б в г д е ж з и к л м н о п р с т у ф х ц ч ш щ ъ ы ь э ю я'.split()
    dataset_train = comnist_dataset('comnist_splitted/train/*/*.png', labels)
    dataset_test = comnist_dataset('comnist_splitted/test/*/*.png', labels)
    print(len(dataset_train), len(dataset_test))

    trainloader = DataLoader(dataset=dataset_train,
                             batch_size=batch_size,
                             shuffle=True, drop_last=True)

    testloader = DataLoader(dataset=dataset_test,
                            batch_size=batch_size,
                            shuffle=True, drop_last=True)

    augment = torchvision.transforms.Compose([
                                        torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                        torchvision.transforms.RandomVerticalFlip(p=0.5),
#                                         torchvision.transforms.ToTensor(),
                                        ])

    model = Model().float()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.004)

    n_epochs = 40
    acc_best = 0
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = augment(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        acc = 0
        model.eval()
        for i, data in enumerate(testloader):
            inputs, labels = data

            inputs = augment(inputs)

            outputs = model(inputs.float())

            acc += torch.sum(torch.argmax(outputs,1) == labels)

        acc = acc / len(testloader) / batch_size
        print('Accuracy: {}'.format(acc*100))

        if acc > acc_best:
            acc_best = acc
            torch.save(model.state_dict(), save_path)


    print('Best accuracy: {}'.format(acc_best*100))

    print('Finished Training')
