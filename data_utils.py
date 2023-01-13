import logging

import torch
import os
import random
import json
from my_dataset import MyDataSet
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler


logger = logging.getLogger(__name__)

def read_split_data(root: str,isTrain=False):
    '''
    读取cub数据集
    '''
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    images_path = []  # 存储数据集的所有图片路径
    images_label = []  # 存储数据集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        #val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            images_path.append(img_path)
            images_label.append(image_class)

    print("{} images were found in the {} dataset.".format(sum(every_class_num),'train' if isTrain else 'test'))
    print("{} images for {}.".format(len(images_path),'training' if isTrain else 'testing'))
    return images_path, images_label

def get_loader(args):
    '''
    获取 dataloader
    '''
    
    transform_train = transforms.Compose([
        transforms.Resize((args.img_size,args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "mnist":
        trainset = datasets.MNIST(root="data/mnist",
                                    train=True,
                                    download=True,
                                    transform=transforms.Compose([
                                        transforms.Resize(args.img_size), transforms.ToTensor(), 
                                        transforms.Normalize([0.5], [0.5])
                                        ])
                                    )
    if args.dataset == 'cub200':
        data_pth = 'data/CUB_200_2011/'
        train_images_path, train_images_label = read_split_data(
            data_pth+'train',isTrain=True)
        trainset = MyDataSet(images_path=train_images_path,
                                  images_class=train_images_label,
                                  transform=transform_train)

    train_sampler = RandomSampler(trainset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.batch_size,
                              num_workers=4,
                              pin_memory=True)
    return train_loader
