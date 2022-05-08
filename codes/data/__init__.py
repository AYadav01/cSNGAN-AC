'''create dataset and dataloader'''
import logging
import torch.utils.data


"""
Create dataloader given the dataset object and other parameter dictionary
"""
def create_dataloader(dataset, dataset_opt):
    phase = dataset_opt['phase']
    # return dataloader depending on phase, either 'train' or 'test'
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['n_workers'],
            drop_last=True,
            pin_memory=True)
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1)


"""
Create Dataset object for dataloader
"""
def create_dataset(dataset_opt):
    # import dataset class depending on 'data_type'
    if dataset_opt['data_type'] == 'h5':
        from data.h5Dataset_with_label import h5Dataset as D
    elif dataset_opt['data_type'] == 'dicom':
        from data.dcmDataset import dcmDataset as D
    # load uids in this phase    
    with open(dataset_opt['uids_path'], 'r') as f:
        lines = f.readlines() # ['1d7425c7412df778d26c027dc5f46a38\n', ....]
        dataset_opt['uids'] = [l.rstrip() for l in lines] 
    dataset = D(dataset_opt) # create dataobject
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
