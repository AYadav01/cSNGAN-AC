import os
import glob
import torch
import torch.nn as nn
import math
import models.networks as networks
import copy


class BaseModel():
    def __init__(self, opt):
        # input dictionary
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu') # gpu id is not defined
        self.is_train = opt['is_train'] # True for training
        self.schedulers = [] 
        self.optimizers = [] 
        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)

    """
    Creates class attributes for 'LR' and 'HR' data
    """
    def feed_train_data(self, data, need_HR=True):
        if self.opt['is_train']:
            if self.opt['network_G']['need_embed'] or self.opt['network_D']['need_embed'] or self.opt['network_D']['aux_lbl_loss']:
                if set(['kernel', 'dose']).issubset(data.keys()):
                    self.train_kernel_label = data['kernel'].to(self.device)
                    self.train_dose_label = data['dose'].to(self.device)
                else:
                    raise ValueError('Labels not found in `feed_train_data`')
        # get LR data
        self.var_L = data['LR'].to(self.device, non_blocking=True)  # LR
        # get HR data
        if need_HR:
            self.real_H = data['HR'].to(self.device, non_blocking=True)  # HR

    """
    prepare LR and HR output to feed to model
    """
    def feed_test_data(self, data, need_HR=True):
        if self.opt['is_train']:
            if self.opt['network_G']['need_embed'] or self.opt['network_D']['need_embed'] or self.opt['network_D']['aux_lbl_loss']:
                if set(['kernel', 'dose']).issubset(data.keys()):
                    self.test_kernel_label = data['kernel'].to(self.device)
                    self.test_dose_label = data['dose'].to(self.device)
                else:
                    raise ValueError('Labels not found in `feed_test_data`')
        else:
            # during testing, only generator might need labels
            if self.opt['network_G']['need_embed']:
                if set(['kernel', 'dose']).issubset(data.keys()):
                    self.test_kernel_label = data['kernel'].to(self.device)
                    self.test_dose_label = data['dose'].to(self.device)
                else:
                    raise ValueError('Labels not found in `feed_test_data`')
        # Get the HR image
        if need_HR:
            self.real_H = data['HR'].to(self.device, non_blocking=True)  # HR
        opt_val = self.opt['datasets']['val']
        pt = opt_val['slice_size']
        self.ot = opt_val['overlap_slice_size']
        self.nt = 1 + math.ceil((data['LR'].size(2) - pt) / (pt - self.ot))
        # get 'slice_size' in z-direction
        if not opt_val['need_voxels']:
            # reshape the whole volume into blocks of voxel of size pt x 512 x 512
            self.var_L = torch.empty(self.nt, 1, pt, data['LR'].size(3), data['LR'].size(4))\
                .to(self.device, non_blocking=True)
            # fill the volume
            for i in range(0, self.nt - 1):
                self.var_L[i, :, :, :, :] = data['LR'][0, 0, i*(pt-self.ot):i*(pt-self.ot)+pt, :, :]
            # the last one
            self.var_L[-1, :, :, :, :] = data['LR'][0, 0, -pt:, :, :]
        else:
            self.var_L = data['LR'].to(self.device, non_blocking=True)

    """
    Calculate accuracy for auxially classes
    """
    def compute_acc(self, preds, labels):
        correct = 0
        preds_ = preds.data.max(1)[1]
        correct = preds_.eq(labels.data).cpu().sum()
        acc = float(correct) / float(len(labels.data)) * 100.0
        return acc
    
    """
    Copies fp32 model and convert to fp16
    """
    def half(self):
        print("Model being converted to fp16")
        print("-"*40)
        if self.opt['precision'] == 'fp16':
            self.netG_eval = copy.deepcopy(self.netG).half()
        else:
            self.netG_eval = self.netG
            
    def prepare_quant(self, loader):
        # PyTorch Static Quantization 
        # https://leimao.github.io/blog/PyTorch-Static-Quantization/
        fused_model = copy.deepcopy(self.netG)
        fused_model.eval()
        # fused conv3d + relu
        fused_model = torch.quantization.fuse_modules(fused_model.net, [["3", "4"],["5","6"]], inplace=True)
        for i in range(8):
            torch.quantization.fuse_modules(fused_model[1].res[i].res, [["0", "1"]], inplace=True)
        quantized_model = networks.define_Quant(model_fp32=fused_model)
        # config
        quantization_config = torch.quantization.get_default_qconfig("fbgemm")
        quantized_model.qconfig = quantization_config
        torch.quantization.prepare(quantized_model, inplace=True)
        # calibration
        data = next(iter(loader))  
        # get a small chunk for calibration
        _ = quantized_model(data['LR'][:,:,:32 ,:,:])
        quantized_model = torch.quantization.convert(quantized_model, inplace=True)
        self.netG_eval = quantized_model

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    """
    For each schedulers, we step up the learning rate
    """
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    """
    Returns the current learning rate of scheduler for generator only
    """
    def get_current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    """
    Get string representation of a network and its parameters
    """
    def get_network_description(self, network):
        # check if 'network' is an instance of 'nn.DataParallel'
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    """
    Save the model (both generator & discriminator) 'state_dict' given the iteration step
    """
    def save_network(self, network, network_label, iter_step):
        save_filename = '{}_{}.pth'.format(iter_step, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        # put 'state_dict' parameters on cpu before saving, so the inference could even be done on cpu
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        """
        # remove the previous files
        # old_files = glob.glob(os.path.join(self.opt['path']['models'], "*"+network_label+"*"))
        # for f in old_files:
        #     os.remove(os.path.join(self.opt['path']['models'], f))
        # save the new file
        """
        # save the models 
        torch.save(state_dict, save_path)

    """
    Loads the model with weights, given 'load_path' (generator or discriminator weight path), and
    the network architecture itself
    """
    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path), strict=strict)

    """
    Saves training state (epoch, current_step, schedulers, optimizers) during training, which will be 
    used for resuming
    """
    def save_training_state(self, epoch, iter_step):
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        old_files = os.listdir(self.opt['path']['training_state'])
        for f in old_files:
            os.remove(os.path.join(self.opt['path']['training_state'], f))
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)
