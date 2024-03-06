import cv2
import numpy as np
from torch.utils import data
import torchvision.transforms as tv
from PIL import Image
import torch.utils.data as data
import torch
import os
import random
from PIL import Image
import source_aug

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def textread(path):
    # if not os.path.exists(path):
    #     print path, 'does not exist.'
    #     return False
    f = open(path)
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n', '')
    return lines
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path,num_views):
    if num_views==12:
       im = np.array(Image.open(path).convert('RGB'))
       im=255-im
       return Image.fromarray(im)
    elif num_views==1:
       return Image.open(path).convert('RGB')

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def make_dataset(root, label, dataset_name, mode):
    images = []
    if dataset_name == 'MI3DOR-1':
        labeltxt = open(label)
        for line in labeltxt:
            data = line.strip().split()
            if (int(data[1]) > 20) or (int(data[1]) < 0):
                raise RuntimeError('载入标签错误，请检查：{}'.format(data))
            if mode == '3D':
              img = os.path.join(root, os.path.dirname(data[0]))
              fileList = os.listdir(img)
              for filename in fileList:
                  img_path = os.path.join(img, filename)
                  if is_image_file(img_path):
                      path = img_path
                  else:
                      raise RuntimeError('path路径非图片，目标路径为{}'.format(line))
                  gt = int(data[1])
                  item = (path, gt)
                  images.append(item)
            elif mode == '2D':
                img_path = os.path.join(root, data[0])
                if is_image_file(img_path):
                  path = img_path
                else:
                  raise RuntimeError('path路径非图片，目标路径为{}'.format(data))
                gt = int(data[1])
                item = (path, gt)
                images.append(item)
            else:
                raise RuntimeError('导入路径出错，请检查路径{}{}'.format(root, judge[0]))

    elif dataset_name=='MI3DOR-2':
        labeltxt = open(label)
        for line in labeltxt:
            data = line.strip().split(' ')
            if (int(data[1]) > 39) or (int(data[1]) < 0):
                raise RuntimeError('载入标签错误，请检查：{}'.format(data))
            img_path = root + data[0]
            if is_image_file(img_path):
                path = img_path
            else:
                raise RuntimeError('path路径非图片，目标路径为{}'.format(data))
            gt = int(data[1])
            item = (path, gt)
            images.append(item)

    return images
        

class multiVisDAImage_self(data.Dataset):
    def __init__(self, root, label, num_views=12, training=True,
                 data_transform=None, dataset_name='MI3DOR-1', mode='2D', Instance = False):
        imgs = make_dataset(root, label,dataset_name, mode)
        print('len(imgs): ', len(imgs))
        self.root = root
        self.label = label
        self.imgs = imgs
        self.num_views = num_views
        self.mean_color = [104.006, 116.668, 122.678]
        self.multi_scale = [256, 257]
        self.output_size = [227, 227]
        self.training = training
        self.data_transform = data_transform
        self.loader = rgb_loader
        self.RandAugmentMC = source_aug.RandAugmentMC(3,5)
        self.Instance = Instance
        if self.training:
            if self.num_views > 1:
                self.strid = 1
                self.select_view = 12
            if self.num_views == 1:
                self.strid = 1
                self.select_view = 1
        else:
            if self.num_views > 1:
                self.strid = 1
                self.select_view = 12
            if self.num_views == 1:
                self.strid = 1
                self.select_view = 1
        imgs_new = []
        if self.num_views == 1:
            for i in range(int(len(self.imgs) / self.num_views)):
                imgs_new.extend(self.imgs[i * self.num_views:(i + 1) * self.num_views])
        if self.num_views > 1:
            for i in range(int(len(self.imgs) / self.num_views)):
                imgs_new.extend(self.imgs[i * self.num_views:(i + 1) * self.num_views:self.strid])
        self.imgs = imgs_new

    def __getitem__(self, index):
        tensor = tv.ToTensor()
        imgset = []
        for i in range(int(self.select_view)):
            path, target = self.imgs[index * self.select_view + i]
            img = cv2.imread(path)

            if type(img) == None:
                print('Error: Image at {} not found.'.format(path))
            if self.training and np.random.random() < 0.5:
                img = cv2.flip(img, 1)
            new_size = np.random.randint(self.multi_scale[0], self.multi_scale[1], 1)[0]
            img = cv2.resize(img, (new_size, new_size))
            img = img.astype(np.float32)
            if self.training:
                diff = new_size - self.output_size[0]
                offset_x = np.random.randint(0, diff, 1)[0]
                offset_y = np.random.randint(0, diff, 1)[0]
            else:
                offset_x = img.shape[0] // 2 - self.output_size[0] // 2
                offset_y = img.shape[1] // 2 - self.output_size[1] // 2
            img = img[offset_x:(offset_x + self.output_size[0]),
                  offset_y:(offset_y + self.output_size[1])]
            if self.data_transform:
                img = Image.fromarray(img)
                img = self.data_transform(img)
                img = img.numpy()
            if self.Instance == True:
                for img_index in range(12):
                    if img_index == 0:
                        img1 = img - np.asarray(self.mean_color)
                        img1 = tensor(img1)
                        imgset.append(img1)
                        del img1
                    else:
                        img2 = Image.fromarray(np.uint8(img))
                        img2 = self.RandAugmentMC(img2)
                        img2 = np.array(img2, np.float32)
                        img2 = img2 - np.asarray(self.mean_color)
                        img2 = tensor(img2)
                        imgset.append(img2)
                        del img2
            else:
                img -= np.asarray(self.mean_color)
                img = tensor(img)
                imgset.append(img)
            del img

        return torch.stack(imgset), target

    def __len__(self):
        return int(len(self.imgs) / self.select_view)

def data_dir(data_root, mode='train'):
    if "MI3DOR-1" in data_root:
        source_root = os.path.join(data_root, 'train/2D/', "images/")
        source_label = os.path.join(data_root, 'train/2D/', "label.txt")
        target_root = os.path.join(data_root, 'train/3D/', "images/")
        target_label = os.path.join(data_root, 'train/3D/', "label.txt")
        source_test_root = os.path.join(data_root, "test/2D/")
        source_test_label = os.path.join(data_root, "test/2D/", "labels.txt")
        target_test_root = os.path.join(data_root, "test/3D/")
        target_test_label = os.path.join(data_root, "test/3D/", "labels.txt")
        n_class = 21
        dataset_name = 'MI3DOR-1'
    elif "MI3DOR-2" in data_root:
        source_root = os.path.join(data_root, 'train/2D/')
        source_label = os.path.join(data_root, "train/MI3DOR_2_Image_train_list.txt")
        target_root = os.path.join(data_root, 'train/3D/')
        target_label = os.path.join(data_root, "train/MI3DOR_2_View_train_list.txt")
        source_test_root = os.path.join(data_root, 'test/2D/')
        source_test_label = os.path.join(data_root, "test/MI3DOR_2_Image_test_list.txt")
        target_test_root = os.path.join(data_root, 'test/3D/')
        target_test_label = os.path.join(data_root, "test/MI3DOR_2_View_test_list.txt")
        n_class = 40
        dataset_name = 'MI3DOR-2'
    else:
        raise RuntimeError("Mismatch data_root")

    if mode == 'train':
        return source_root, source_label, target_root, target_label, n_class, dataset_name
    elif mode == 'test':
        return source_test_root, source_test_label, target_test_root, target_test_label, n_class, dataset_name
    else:
        raise ValueError('mode must be train or test')

def data_loader(data_root, mode, source_view, target_view, *args):
    if mode == 'train':
        assert len(args) == 4
        batch_size_s, batch_size_t, batch_size_s_val, batch_size_t_val = args
        source_root, source_label, target_root, target_label, n_class, dataset_name = data_dir(data_root, 'train')

        s_loader_data = multiVisDAImage_self(source_root, source_label, num_views=source_view, 
                                                     dataset_name=dataset_name, mode='2D', Instance = True)
        t_loader_data = multiVisDAImage_self(target_root, target_label, num_views=target_view,
                                                     dataset_name=dataset_name, mode='3D')
        s_loader_val = multiVisDAImage_self(source_root, source_label, num_views=source_view,
                                                     dataset_name=dataset_name, mode='2D')
        t_loader_val = multiVisDAImage_self(target_root, target_label, num_views=target_view,
                                                    training=False, dataset_name=dataset_name, mode='3D')
        s_loader = torch.utils.data.DataLoader(s_loader_data, 
                                                 batch_size=batch_size_s, shuffle=True, drop_last=True, num_workers=8)
        t_loader = torch.utils.data.DataLoader(t_loader_data,
                                                 batch_size=batch_size_t, shuffle=True, drop_last=True, num_workers=8)
        s_val_loader = torch.utils.data.DataLoader(s_loader_val,
                                                    batch_size=batch_size_s_val, shuffle=False, num_workers=8)    #原4
        t_val_loader = torch.utils.data.DataLoader(t_loader_val,
                                                    batch_size=batch_size_t_val, shuffle=False, num_workers=8)    #原4
        return s_loader, t_loader, s_val_loader, t_val_loader, n_class
    elif mode == 'test':
        assert len(args) == 2
        batch_size_s, batch_size_t = args
        source_test_root, source_test_label, target_test_root, target_test_label, n_class, dataset_name = data_dir(data_root, 'test')
        s_loader_test = multiVisDAImage_self(source_test_root, source_test_label, num_views=source_view,
                                                     training=False, dataset_name=dataset_name, mode='2D')
        t_loader_test = multiVisDAImage_self(target_test_root, target_test_label, num_views=target_view,
                                                     training=False, dataset_name=dataset_name, mode='3D')
        s_loader = torch.utils.data.DataLoader(s_loader_test,
                                               batch_size=batch_size_s, shuffle=False, num_workers=8)
        t_loader = torch.utils.data.DataLoader(t_loader_test,
                                               batch_size=batch_size_t, shuffle=False, num_workers=8)
        s_loader_len, t_loader_len = len(s_loader_test), len(t_loader_test)
        print('source_label:{}, \ntarget_label:{}'.format(source_test_label, target_test_label))
        print('源域样本量：', s_loader_len)
        print('目标域样本量：', t_loader_len)
        return s_loader, t_loader, n_class
    elif mode == 'test_train':
        assert len(args) == 2
        batch_size_s, batch_size_t = args
        source_test_root, source_test_label, target_test_root, target_test_label, n_class, dataset_name = data_dir(
            data_root, 'train')
        s_loader_test = multiVisDAImage_self(source_test_root, source_test_label, num_views=source_view,
                                             training=False, dataset_name=dataset_name, mode='2D')
        t_loader_test = multiVisDAImage_self(target_test_root, target_test_label, num_views=target_view,
                                             training=False, dataset_name=dataset_name, mode='3D')
        s_loader = torch.utils.data.DataLoader(s_loader_test,
                                               batch_size=batch_size_s, shuffle=False, num_workers=8)
        t_loader = torch.utils.data.DataLoader(t_loader_test,
                                               batch_size=batch_size_t, shuffle=False, num_workers=8)
        s_loader_len, t_loader_len = len(s_loader_test), len(t_loader_test)
        print('source_label:{}, \ntarget_label:{}'.format(source_test_label, target_test_label))
        print('源域样本量：', s_loader_len)
        print('目标域样本量：', t_loader_len)
        return s_loader, t_loader, n_class
    else:
        raise ValueError('mode must be train or test')