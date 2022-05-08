import numpy as np
import h5py
import torch.utils.data as data
import utils.util as util
import random
import os
import re
import torch


class h5Dataset(data.Dataset):
    def __init__(self, opt):
        super(h5Dataset, self).__init__()
        # useful when extracting 3D mask cube from the input masks
        self.FILL_RATIO_THRESHOLD = 0.8
        self.opt = opt
        self.in_folder = opt['dataroot_LR']
        self.tar_folder = opt['dataroot_HR']
        self.mask_folder = opt['maskroot_HR']
        # 3d voxel size
        if self.opt['phase'] == 'train' or self.opt['need_voxels']:
            self.ps = (opt['LR_slice_size'], opt['LR_size'], opt['LR_size'])
        self.uids = opt['uids'] # list of uids
        if opt['subset'] is not None:
           self.uids = self.uids[:opt['subset']]
        self.scale = opt['scale']
        self.ToTensor = util.ImgToTensor()

    def _cvt_int(self, inputString):
        find_digit = re.search(r'\d', inputString)
        reformat_num = int(float(inputString[find_digit.start():]))
        reformat_str = inputString[:find_digit.start()] + str(reformat_num)
        return reformat_str

    def __getitem__(self, index):
        uid = self.uids[index]
        if self.opt['data_merged']:
            uid_to_open_for_mask_target = uid.split('_')[0]
        else:
            uid_to_open_for_mask_target = uid
        # body mask - check if the random voxel contain 80% of the body mask
        vol_mask = None
        if self.mask_folder:
            with h5py.File(os.path.join(self.mask_folder, uid_to_open_for_mask_target+'.h5'), 'r') as file:
                IMG_THICKNESS, IMG_WIDTH, IMG_HEIGHT = file['data'].shape
                if self.opt['phase'] == 'train' or (self.opt['need_voxels'] and not self.opt['need_voxels']['tile_x_y']):
                    t, w, h = self.ps
                    fill_ratio = 0.
                    while fill_ratio < self.FILL_RATIO_THRESHOLD:
                        rnd_t_HR = random.randint(0, IMG_THICKNESS - int(t * self.scale))
                        rnd_h = random.randint(0, IMG_HEIGHT - h)
                        rnd_w = random.randint(0, IMG_WIDTH - w)
                        extracted_cube = file['data'][rnd_t_HR:rnd_t_HR+int(t*self.scale), rnd_h:rnd_h+h, rnd_w:rnd_w+w].sum()
                        total_required = (self.scale * t * w * h)
                        fill_ratio = extracted_cube/total_required
                    vol_mask = None
                # get vol_mask during validation
                if self.opt['phase'] == 'val':
                    if self.mask_folder and self.opt['need_voxels'] and not self.opt['need_voxels']['tile_x_y']:
                        vol_mask = file['data'][rnd_t_HR:rnd_t_HR+int(t*self.scale), rnd_h:rnd_h+h, rnd_w:rnd_w+w]
                    else:
                        vol_mask = file['data'][()]
        # LR
        with h5py.File(os.path.join(self.in_folder, uid+'.h5'), 'r') as file:
            if self.opt['phase'] == 'train':
                vol_in = file['data'][round(rnd_t_HR/self.scale):round(rnd_t_HR/self.scale)+t,
                                      rnd_h:rnd_h+h, rnd_w:rnd_w+w]
            else:
                if self.mask_folder and self.opt['need_voxels'] and not self.opt['need_voxels']['tile_x_y']:
                    vol_in = file['data'][round(rnd_t_HR/self.scale):round(rnd_t_HR/self.scale)+t,
                                          rnd_h:rnd_h+h, rnd_w:rnd_w+w]
                else:
                    vol_in = file['data'][()]
        
        # HR
        vol_tar = None
        if self.tar_folder:
            with h5py.File(os.path.join(self.tar_folder, uid_to_open_for_mask_target+'.h5'), 'r') as file:
                if self.opt['phase'] == 'train':
                    vol_tar = file['data'][rnd_t_HR:rnd_t_HR+int(t*self.scale), rnd_h:rnd_h+h, rnd_w:rnd_w+w]
                else:
                    if self.mask_folder and self.opt['need_voxels'] and not self.opt['need_voxels']['tile_x_y']:
                        vol_tar = file['data'][rnd_t_HR:rnd_t_HR+int(t*self.scale), rnd_h:rnd_h+h, rnd_w:rnd_w+w]
                    else:
                        vol_tar = file['data'][()]

        vol_in = np.expand_dims(vol_in, axis=0)
        vol_tar = np.expand_dims(vol_tar, axis=0)
        vol_mask = np.expand_dims(vol_mask, axis=0)
        vol_in, vol_tar = self.ToTensor(vol_in), self.ToTensor(vol_tar)
        vol_mask = self.ToTensor(vol_mask, raw_data_range = 1)

        # read LR spacings
        spacings = [] 
        if self.opt['phase'] == 'val':
            config_path = os.path.join(self.in_folder, uid + '.json')
            meta_data = util.read_config(config_path)
            spacings = meta_data['Spacing']
        
        # set up labels if needed
        if self.opt['need_label']:
            class_label = {'d10':0, 'd25':1, 'd100': 2, 'k1':0, 'k2':1, 'k3':2, 'st1':0, 'st0.6':1, 'st2':2}
            if not self.opt['data_merged']:
                kernel_lbl, dose_lbl, st_lbl = self.opt['dataroot_LR'].split('/')[-1].split('_')
                kernel_lbl_to_assign, dose_lbl_to_assign, st_lbl_to_assign = self._cvt_int(kernel_lbl),\
                                                                                        self._cvt_int(dose_lbl), self._cvt_int(st_lbl)
                kernel_lbl = np.expand_dims(class_label[kernel_lbl_to_assign], axis=0)
                dose_lbl = np.expand_dims(class_label[dose_lbl_to_assign], axis=0)
            else:
                kernel_lbl = np.expand_dims(class_label[uid.split('_')[1:][0]], axis=0)
                dose_lbl = np.expand_dims(class_label[uid.split('_')[1:][1]], axis=0)
            out_dict = {'LR': vol_in, 'HR': vol_tar, 'mask': vol_mask, 'spacings': spacings, 'uid': uid,
                        'kernel': kernel_lbl, 'dose': dose_lbl}
        else:
            out_dict = {'LR': vol_in, 'HR': vol_tar, 'mask': vol_mask, 'spacings': spacings, 'uid': uid}
        return out_dict

    def __len__(self):
        return len(self.uids)
