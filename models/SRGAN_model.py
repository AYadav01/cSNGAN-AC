import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import models.lr_scheduler as lr_scheduler
import models.networks as networks
from .base_model import BaseModel
from models.modules.loss import GANLoss, GradientPenaltyLoss
"""
Amp allows users to easily experiment with different pure and mixed precision modes. 
Commonly-used default modes are chosen by selecting an “optimization level” or opt_level; each opt_level 
establishes a set of properties that govern Amp’s implementation of pure or mixed precision training.
- opt_level: 01 = Mixed Precision (recommended for typical use)
"""
from apex import amp
import apex
logger = logging.getLogger('base')


"""
Instantiates both generator and discriminator, put models in training, define losses, optimizers and schedulers
"""
class SRGANModel(BaseModel):
    def __init__(self, opt):
        super(SRGANModel, self).__init__(opt)
        train_opt = opt['train']
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)  # D
            self.netG.train()
            self.netD.train()

        # define losses, optimizer and scheduler for training mode
        if self.is_train:
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion'] # l1
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight'] # get pixel weight
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # auxially classification loss
            if self.opt['network_D']['aux_lbl_loss']:
                logger.info('Including auxillary loss')
                self.aux_loss = nn.NLLLoss().to(self.device)

            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight'] # get feature weight
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None

            if self.cri_fea:
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            self.cri_gan = GANLoss(train_opt['gan_type'], real_label_val=1.0, fake_label_val=0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            if "wgan" in train_opt['gan_type']:
                if train_opt['gan_type'] == "wgan-gp":
                    self.cri_gp = GradientPenaltyLoss(center=1.).to(self.device)
                elif train_opt['gan_type'] == "wgan-gp0":
                    self.cri_gp = GradientPenaltyLoss(center=0.).to(self.device)
                else:
                    raise NotImplementedError("{:s} not found".format(train_opt['gan_type']))
                self.l_gp_w = 10. # weight for gradient penality (used in 'optimize_parameters' function)

            # optimizers - weight decay for generator
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], train_opt['beta2_G']))

            self.optimizers.append(self.optimizer_G)
            # weight decay for discriminator
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'], \
                weight_decay=wd_D, betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_D)
            # configure schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'], # [20e3, 40e3, 60e3]
                                                         restarts=train_opt['restarts'], # null
                                                         weights=train_opt['restart_weights'], # null
                                                         gamma=train_opt['lr_gamma'], # 0.5
                                                         clear_state=train_opt['clear_state'])) # None
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(optimizer, train_opt['T_period'],
                                                               eta_min=train_opt['eta_min'],
                                                               restarts=train_opt['restarts'],
                                                               weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('Choose MultiStepLR or CosineAnnealingLR')

            self.log_dict = OrderedDict()
        self.print_network() # print the instantiated model
        self.load()  # load G and D if needed

    """
    Initializes model for mixed precision training, depending on 'opt_level'. Below is sample implementation
    -----------------------------------------------------------------------
    # Declare model and optimizer as usual, with default (FP32) precision
    model = torch.nn.Linear(D_in, D_out).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Allow Amp to perform casts as required by the opt_level
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    ...
    # loss.backward() becomes:
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    """
    def initialize_amp(self):
        [self.netG, self.netD], [self.optimizer_G, self.optimizer_D] = \
        amp.initialize([self.netG, self.netD], [self.optimizer_G, self.optimizer_D],
                       opt_level=self.opt['opt_level'], num_losses = 2)
        if self.opt['gpu_ids']: 
            """
            Implements data parallelism at the module level. This container parallelizes 
            the application of the given module by splitting the input across the specified
            """
            assert torch.cuda.is_available()
            self.netG = nn.DataParallel(self.netG)
            self.netD = nn.DataParallel(self.netD)

    def test(self, data):
        self.netG_eval.eval()
        if self.opt['precision'] == 'fp16':
            var_L_eval = self.var_L.half()
        else:
            var_L_eval = self.var_L
        opt_val = self.opt['datasets']['val']
        num_HR = int (data['LR'].size(2) * self.opt['scale'])
        HR_ot = int(self.opt['scale'] * self.ot)
        if not opt_val['need_voxels']:
            pt, H, W = self.var_L.size(2), self.var_L.size(3), self.var_L.size(4)
            pt = int(pt * self.opt['scale'])
            self.fake_H = torch.empty(1, 1,  num_HR, H, W, device=self.device)
            if self.opt['precision'] == 'fp16':
                fake_H_in_chunks = torch.empty(self.nt, 1,  pt, H, W, dtype=torch.half, device=self.device)
            else:
                fake_H_in_chunks = torch.empty(self.nt, 1,  pt, H, W, device=self.device)

            stitch_mask = torch.zeros_like(self.fake_H, device=self.device)
            with torch.no_grad():
                if opt_val['full_volume']:
                    if self.opt['network_G']['need_embed']:
                        self.fake_H = self.netG_eval(var_L_eval, self.test_kernel_label, self.test_dose_label)
                    else:
                        self.fake_H = self.netG_eval(var_L_eval)
                else:
                    for i in range(0, self.nt):
                        if self.opt['network_G']['need_embed']:
                            fake_H_in_chunks[[i],...] = self.netG_eval(var_L_eval[[i],...], self.test_kernel_label, self.test_dose_label)
                        else:
                            fake_H_in_chunks[[i],...] = self.netG_eval(var_L_eval[[i],...])
                    # stitch volume in z-direction
                    for i in range(0, self.nt - 1):
                        ts, te = i * (pt - HR_ot), i * (pt - HR_ot) + pt
                        self.fake_H[0, 0, ts:te, :, :] = (self.fake_H[0, 0, ts:te, :, :] * stitch_mask[0, 0, ts:te, :, :] +
                        fake_H_in_chunks[i,...].float() * (2 - stitch_mask[0, 0, ts:te, :, :])) / 2
                        stitch_mask[0, 0, ts:te, :, :] = 1.
                    # stitch last volume
                    self.fake_H[0, 0, -pt:, :, :] = \
                        (self.fake_H[0, 0, -pt:, :, :] * stitch_mask[0, 0, -pt:, :, :] +
                        fake_H_in_chunks[-1,...].float() * (2 - stitch_mask[0, 0, -pt:, :, :])) / 2
        else:
            if opt_val['need_voxels'] and not opt_val['need_voxels']['tile_x_y']:
                if self.opt['network_G']['need_embed']:
                    self.fake_H = self.netG_eval(var_L_eval, self.test_kernel_label, self.test_dose_label).float()
                else:
                    self.fake_H = self.netG_eval(var_L_eval).float()
            elif opt_val['need_voxels'] and opt_val['need_voxels']['tile_x_y']:
                pt, H, W = opt_val['slice_size'], data['LR'].size(3), data['LR'].size(4)
                pt = int(pt * self.opt['scale'])
                self.fake_H = torch.empty(1, 1,  num_HR, H, W, device=self.device)
                # get predictions
                with torch.no_grad():
                    for row in range(0, data['LR'].size(3), opt_val['need_voxels']['tile_size']):
                        for column in range(0, data['LR'].size(3), opt_val['need_voxels']['tile_size']):
                            LR_chunked = var_L_eval[:, :, :, row:row+opt_val['need_voxels']['tile_size'], column:column+opt_val['need_voxels']['tile_size']]
                            GT_chunked = data['HR'][:, :, :, row:row+opt_val['need_voxels']['tile_size'], column:column+opt_val['need_voxels']['tile_size']]
                            # store chunked prediction results
                            if self.opt['precision'] == 'fp16':
                                # [12, 1, 32, 64, 64]
                                tmp_chunk_along_z = torch.empty(self.nt, 1, pt, opt_val['need_voxels']['tile_size'], opt_val['need_voxels']['tile_size'],
                                                    dtype=torch.half, device=self.device)
                            else:
                                tmp_chunk_along_z = torch.empty(self.nt, 1, pt, opt_val['need_voxels']['tile_size'], opt_val['need_voxels']['tile_size'],
                                                    device=self.device)
                            # iterate over number of blocks to get predictions
                            for i in range(0, self.nt - 1):
                                if self.opt['network_G']['need_embed']:
                                    tmp_chunk_along_z[i, :, :, :, :] = self.netG_eval(LR_chunked[:, :, i*(pt-self.ot):i*(pt-self.ot)+pt, :, :], 
                                                                self.test_kernel_label, self.test_dose_label)
                                else:
                                    tmp_chunk_along_z[i, :, :, :, :] = self.netG_eval(LR_chunked[:, :, i*(pt-self.ot):i*(pt-self.ot)+pt, :, :])
                
                            # add the last chunk
                            if self.opt['network_G']['need_embed']:
                                tmp_chunk_along_z[-1, :, :, :, :] = self.netG_eval(LR_chunked[:, :, -pt:, :, :], self.test_kernel_label, self.test_dose_label)
                            else:
                                tmp_chunk_along_z[-1, :, :, :, :] = self.netG_eval(LR_chunked[:, :, -pt:, :, :])

                            reconstructed_z = torch.empty(1, 1, num_HR, opt_val['need_voxels']['tile_size'],
                                                        opt_val['need_voxels']['tile_size'], device=self.device)
                            stitch_mask = torch.zeros_like(reconstructed_z, device=self.device)
                            for i in range(0, self.nt - 1):
                                ts, te = i * (pt - HR_ot), i * (pt - HR_ot) + pt
                                reconstructed_z[0, 0, ts:te, :, :] = (reconstructed_z[0, 0, ts:te, :, :] * stitch_mask[0, 0, ts:te, :, :] + 
                                                                        tmp_chunk_along_z[i,...].float() * (2 - stitch_mask[0, 0, ts:te, :, :])) / 2
                                stitch_mask[0, 0, ts:te, :, :] = 1.
                            # stich last volume
                            reconstructed_z[0, 0, -pt:, :, :] = \
                                (reconstructed_z[0, 0, -pt:, :, :] * stitch_mask[0, 0, -pt:, :, :] +
                                tmp_chunk_along_z[-1,...].float() * (2 - stitch_mask[0, 0, -pt:, :, :])) / 2
                            # accumulate volume together
                            self.fake_H[0, 0, :, row:row+opt_val['need_voxels']['tile_size'], column:column+opt_val['need_voxels']['tile_size']] = reconstructed_z
            else:
                raise ValueError('Unknown tiling case found in cSNGAN model!')
        self.netG.train()

    """
    Feed in LR and HR data to models, calculate loss and optimize gradients
    """
    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad() # zero-gradient
        if self.opt['network_G']['need_embed']:
            self.fake_H = self.netG(self.var_L, self.train_kernel_label, self.train_dose_label)
        else:
            self.fake_H = self.netG(self.var_L)
        l_g_total = 0 # loss for generator
        # ------------------------ #
        # Calcualte Generator Loss #
        # ------------------------ #
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.fake_H, self.real_H)
                l_g_total += self.l_pix_w * l_g_pix 

            if self.cri_fea:
                real_fea = self.netF(self.real_H).detach()
                fake_fea = self.netF(self.fake_H)
                l_g_fea = self.cri_fea(fake_fea, real_fea)
                l_g_total += self.l_fea_w * l_g_fea

            # send fake 'HR', output of gan model, as input to discriminator and get fake prediction 
            if self.opt['network_D']['need_embed']:
                if self.opt['network_D']['aux_lbl_loss']:
                    pred_g_fake, pred_g_kernel_fake, pred_g_dose_fake = self.netD(self.fake_H, self.train_kernel_label, self.train_dose_label)
                else:
                    pred_g_fake = self.netD(self.fake_H, self.train_kernel_label, self.train_dose_label)
            else:
                if self.opt['network_D']['aux_lbl_loss']:
                    pred_g_fake, pred_g_kernel_fake, pred_g_dose_fake = self.netD(self.fake_H)
                else:
                    pred_g_fake = self.netD(self.fake_H)

            if self.opt['train']['gan_type'] == 'hinge':
                l_g_gan = -pred_g_fake.mean()
            else:
                l_g_gan = self.cri_gan(pred_g_fake, True)

            l_g_total += self.l_gan_w * l_g_gan
            # calculate auxially loss for generator if specified
            if self.opt['network_D']['aux_lbl_loss']:
                if self.opt['network_D']['aux_lbl_loss']['apply_to_gen']:
                    g_aux_kernel_loss = self.aux_loss(pred_g_kernel_fake, self.train_kernel_label.view(-1, 1).squeeze(1))
                    g_aux_dose_loss = self.aux_loss(pred_g_dose_fake, self.train_dose_label.view(-1, 1).squeeze(1))
                    l_g_total += (g_aux_kernel_loss + g_aux_dose_loss)

            # backpropogate loss, step-up optimizer
            with amp.scale_loss(l_g_total , self.optimizer_G, loss_id=0) as errG_scaled:
                errG_scaled.backward()
            self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        l_d_total = 0
        # for 'wgan-gp0', we do need gradient on real data
        if self.opt['train']['gan_type'] == 'wgan-gp0':
            self.real_H.requires_grad_()

        # get prediction from discriminator based on real 'HR' data
        if self.opt['network_D']['need_embed']:
            if self.opt['network_D']['aux_lbl_loss']:
                pred_d_real, aux_kernel_real, aux_dose_real = self.netD(self.real_H, self.train_kernel_label, self.train_dose_label)
                pred_d_fake, aux_kernel_fake, aux_dose_fake = self.netD(self.fake_H.detach(), self.train_kernel_label.detach(), self.train_dose_label.detach())  # detach to avoid back propogation to G
            else:
                pred_d_real = self.netD(self.real_H, self.train_kernel_label, self.train_dose_label)
                pred_d_fake = self.netD(self.fake_H.detach(), self.train_kernel_label.detach(), self.train_dose_label.detach())  # detach to avoid back propogation to G
        else:
            if self.opt['network_D']['aux_lbl_loss']:
                pred_d_real, aux_kernel_real, aux_dose_real = self.netD(self.real_H)
                pred_d_fake, aux_kernel_fake, aux_dose_fake = self.netD(self.fake_H.detach())
            else:
                pred_d_real = self.netD(self.real_H)
                pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid back propogation to G

        # ---------------------------- #
        # Calcualte Discriminator Loss #
        # ---------------------------- #
        l_d_real = self.cri_gan(pred_d_real, True)
        l_d_fake = self.cri_gan(pred_d_fake, False)
        l_d_total = l_d_real + l_d_fake
        # calculate auxially loss for discriminator generator
        if self.opt['network_D']['aux_lbl_loss']:
            l_d_aux_kernel_real = self.aux_loss(aux_kernel_real, self.train_kernel_label.view(-1, 1).squeeze(1))
            l_d_aux_dose_real = self.aux_loss(aux_dose_real, self.train_dose_label.view(-1, 1).squeeze(1))
            l_d_aux_kernel_fake = self.aux_loss(aux_kernel_fake, self.train_kernel_label.view(-1, 1).squeeze(1))
            l_d_aux_dose_fake = self.aux_loss(aux_dose_fake, self.train_dose_label.view(-1, 1).squeeze(1))
            l_d_total += (l_d_aux_kernel_real + l_d_aux_kernel_fake + l_d_aux_dose_real + l_d_aux_dose_fake)
            # compute accuracy
            kernel_accuracy = self.compute_acc(torch.exp(aux_kernel_real), self.train_kernel_label.view(-1, 1).squeeze(1))
            dose_accuracy = self.compute_acc(torch.exp(aux_dose_real), self.train_dose_label.view(-1, 1).squeeze(1))

        # if 'wgan' in 'gan_tpye', we calculate gradient penality
        if 'wgan' in self.opt['train']['gan_type']:
            if self.opt['train']['gan_type'] == 'wgan-gp0':
                l_d_gp = self.cri_gp(self.real_H, pred_d_real)
            elif self.opt['train']['gan_type'] == 'wgan-gp':
                batch_size = self.real_H.size(0)
                eps = torch.rand(batch_size, device=self.device).view(batch_size, 1, 1, 1, 1)
                x_interp = (1 - eps) * self.real_H + eps * self.fake_H.detach()
                x_interp.requires_grad_()
                if self.opt['network_D']['need_embed']:
                    if self.opt['network_D']['aux_lbl_loss']:
                        pred_d_x_interp, prd_x_kernel_interp, pred_x_dose_interp = self.netD(x_interp, self.train_kernel_label, self.train_dose_label)
                    else:
                        pred_d_x_interp = self.netD(x_interp, self.train_kernel_label, self.train_dose_label)
                else:
                    if self.opt['network_D']['aux_lbl_loss']:
                        pred_d_x_interp, prd_x_kernel_interp, pred_x_dose_interp = self.netD(x_interp)
                    else:
                        pred_d_x_interp = self.netD(x_interp)
                l_d_gp = self.cri_gp(x_interp, pred_d_x_interp)
            else:
                raise NotImplementedError('Gan type [{:s}] not recognized'.format(self.opt['train']['gan_type']))
            l_d_total += self.l_gp_w * l_d_gp # weight for gp (self.l_gp_w) = 10

        # backpropogate loss, step-up optimizer
        with amp.scale_loss(l_d_total , self.optimizer_D, loss_id=1) as errD_scaled:
            errD_scaled.backward()
        self.optimizer_D.step()

        # Set logs - Log Generator  
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()
            self.log_dict['l_g_total'] = l_g_total.item()
        # Log Discriminator
        self.log_dict['l_d_total'] = l_d_total.item()
        if 'wgan' in self.opt['train']['gan_type']:
            self.log_dict['l_d_gp'] = l_d_gp.item()
            self.log_dict['w_dist'] = - ( l_d_real.item() + l_d_fake.item() )
        
        # D outputs (mean of output from real HR and mean of output from fake HR)
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())
        if self.opt['network_D']['aux_lbl_loss']:
            self.log_dict['Kernel_acc'] = kernel_accuracy
            self.log_dict['Dose_acc'] = dose_accuracy
    
    """
    Returns the log dictionary that contains the gan (generator & discriminator) losses
    """
    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, data, maskOn=True, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0, 0].float() # [channel, 512, 512]
        out_dict['SR'] = self.fake_H.detach()[0, 0].float() # [channel, 512, 512]
        if maskOn:
            # the way we contructed mask it is 1 x 1 x depth x height x width
            mask = data['mask'].to(self.device).float()[0, 0, :]
            out_dict['SR'] *= mask
        if need_HR:
            out_dict['HR'] = self.real_H.detach().float()[0, 0, :]
            if maskOn:
                out_dict['HR'] *= mask
        return out_dict

    """
    Prints out both the generator and discriminator network structure
    """
    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

            if self.cri_fea:
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                    self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

    """
    Load the generator and discriminator weights if provided
    """
    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None: # load, model weights, if path is not None
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        load_path_D = self.opt['path']['pretrain_model_D']
        # load if opt['is_train'] is 'Train' and discriminator weight path is not None
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD)

    """
    Save model weights, both generator and discriminator, at a particular iteration
    """
    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
