import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model'] # srgan for 'train_gan_A01.json'
    # instantiate only G
    if model == 'sr':
        from .SR_model import SRModel as M
    # 'srgan' instantiates both G and D
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M
    else:
        raise NotImplementedError('Model [{:s}] not implemented.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
