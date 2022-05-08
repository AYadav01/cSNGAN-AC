import torch.utils.data as data
import data.util as util
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
from PIL import Image

class LRHRDataset(data.Dataset):
    '''
    Read LR and HR image pairs.
    If only HR image is provided, generate LR image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.paths_HR = None
        self.LR_env = None  # environment for lmdb
        self.HR_env = None

        self.scale = self.opt['scale']
        self.HR_size = self.opt['HR_size']

        # if self.opt['phase'] == 'train':
        #     if  self.opt['use_rot']:
        #         self.transform = transforms.Compose([
        #                                              # transforms.ToPILImage(),
        #                                              # BILINEAR makes images smoother
        #                                              transforms.RandomRotation(20,resample=Image.BILINEAR),
        #                                              transforms.RandomCrop(self.HR_size),
        #                                              transforms.RandomResizedCrop(self.HR_size,
        #                                                                          scale=(0.8, 1.0),
        #                                                                          ratio=(1.0, 1.0)),
        #
        #
        #                                             ])
        #     else:
        #         self.transform = transforms.Compose([
        #                                              # transforms.ToPILImage(),
        #                                              transforms.RandomCrop(self.HR_size),
        #                                             ])
        # else: # inference mode
        #     self.transform = transforms.Compose([transforms.ToPILImage(),
        #                                          ])

        self.ToTensor = util.ImgToTensor()

        # read image list from lmdb or image files
        self.HR_env, self.paths_HR = util.get_image_paths(opt['data_type'], opt['dataroot_HR'])
        self.LR_env, self.paths_LR = util.get_image_paths(opt['data_type'], opt['dataroot_LR'])

        assert self.paths_HR, 'Error: HR path is empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                'HR and LR datasets have different number of images - {}, {}.'.format(\
                len(self.paths_LR), len(self.paths_HR))

    def _transform(self, input, target):

        if self.opt['use_rot']:# random rotation
            angle = transforms.RandomRotation.get_params([-20, 20])
            input = tf.rotate(input, angle, resample=Image.BILINEAR)
            target = tf.rotate(target, angle, resample=Image.BILINEAR)

        params = transforms.RandomCrop.get_params(input, (self.HR_size, self.HR_size))
        input = tf.crop(input, *params)
        target = tf.crop(target, *params)

        if self.opt['use_zoom']:# random zoom in
            params = transforms.RandomResizedCrop.get_params(input, scale=(0.7, 1.0), ratio=(1.0, 1.0))
            input = tf.resized_crop(input, *params, self.HR_size)
            target = tf.resized_crop(target, *params, self.HR_size)

        return input, target

    # def __getitem__(self, index):
    #
    #     HR_path, LR_path = None, None
    #
    #     # get HR image
    #     HR_path = self.paths_HR[index]
    #     img_HR = util.read_img(self.HR_env, HR_path)
    #     # convert pillow mode to F so we can use resize function, pillow does not support int16
    #     # shame on you pillow! Very inconvenient
    #
    #     img_HR = self.transform(transforms.ToPILImage()(img_HR).convert('F'))
    #     # img_HR = self.transform(img_HR).convert('F')
    #
    #     # get LR image on the fly
    #
    #     if self.paths_LR:
    #     # already have LR images, perform the same transform
    #         LR_path = self.paths_LR[index]
    #         img_LR = util.read_img(self.LR_env, LR_path)
    #         img_LR = self.transform(transforms.ToPILImage()(img_LR).convert('F'))
    #     else:
    #         width = self.HR_size if self.opt['phase'] == 'train' else 512
    #         img_LR = transforms.Resize(width // self.scale)(img_HR)
    #
    #     # concert everything to tensor
    #     img_LR = self.ToTensor(img_LR)
    #     img_HR = self.ToTensor(img_HR)
    #
    #     if LR_path is None:
    #         LR_path = HR_path
    #     return {'LR': img_LR, 'HR': img_HR, 'LR_path': LR_path, 'HR_path': HR_path}

    def __getitem__(self, index):
        LR_path, HR_path = None, None
        # get HR image
        HR_path = self.paths_HR[index]
        img_HR = util.read_img(self.HR_env, HR_path)
        # convert pillow mode to F so we can use resize function, pillow does not support int16
        img_HR = transforms.ToPILImage()(img_HR).convert('F')

        # get LR image on the fly
        if self.paths_LR:
        # already have LR images, convert to pil images
            LR_path = self.paths_LR[index]
            img_LR = util.read_img(self.LR_env, LR_path)
            img_LR = transforms.ToPILImage()(img_LR).convert('F')
            if self.opt['phase'] == 'train':
                img_LR, img_HR = self._transform(img_LR, img_HR)
        else:
            # mimic the degradation by downsample
            width = self.HR_size if self.opt['phase'] == 'train' else 512
            img_LR = transforms.Resize(width // self.scale)(img_HR)

        # convert everything to tensor
        img_LR, img_HR = self.ToTensor(img_LR), self.ToTensor(img_HR)

        if LR_path is None:
            LR_path = HR_path
        return {'LR': img_LR, 'HR': img_HR, 'LR_path': LR_path, 'HR_path': HR_path}

    def __len__(self):
        return len(self.paths_HR)
