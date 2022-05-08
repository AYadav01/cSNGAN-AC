import pandas as pd
import os
import numpy as np


def _populate_value(incoming_lines, file_substr, data_dict):
    for line in incoming_lines:
        metric = line.split(' {} '.format(file_substr))[-1].split(' for ')[0].upper()
        view = line.split(' in ')[-1].split(' view' )[0].upper()
        value = float(line.split(' view :')[-1])
        d_place = float("{0:.4f}".format(value))
        data_dict[metric][view].append(d_place)
    return data_dict

def _save_csv(data_dict, num_sub, merge_folds, path_to_save):
    for each_model in data_dict:
        model_data = data_dict[each_model]
        for each_metric in model_data:
            metric_data = model_data[each_metric]
            for each_view in metric_data:
                view_data = np.array(metric_data[each_view])
                if merge_folds:
                    csv_dict = {'Subject':np.arange(1, 5+1, 1)} # 5 folds
                    # csv_dict['{}_out'.format(each_model)] = np.mean(view_data, axis=0)
                    csv_dict['{}_out'.format(each_model)] = np.std(view_data, axis=1)
                else:
                    csv_dict = {'Subject':np.arange(1, num_sub+1, 1)}
                    csv_dict['{}_out'.format(each_model)] = view_data
                    print('csv ditc:', csv_dict)
                file_name = "SD_across_Sub_{}_{}_{}.csv".format(each_model, each_metric, each_view)
                file_path = os.path.join(path_to_save, file_name)
                df = pd.DataFrame.from_dict(csv_dict) 
                df.to_csv (file_path, index=False, header=True)
    print('ALL Files Created!')

def get_avg_values(log_files, file_substr, start_from, num_sub, num_lines_to_skip, merge_folds):
    # scan all files
    master_dict = {}
    for root, dirs, files in os.walk(log_files, topdown=True):
        if len(files) == 1 or len(files) == 2:
            for file in files:
                if file.endswith('.log') and file_substr in file:
                    file_path = os.path.join(root, file)
                    fold_name = root.split('/')[-1]
                    master_dict[fold_name] = {}
                    # read file for a specific fold/case
                    with open(file_path, 'r') as file:
                        lines = file.readlines()
                        master_dict[fold_name] = {  'PSNR':{'XY': [], 'XZ':[], 'YZ':[]},
                                                    'SSIM':{'XY': [], 'XZ':[], 'YZ':[]},
                                                    'PDIST':{'XY': [], 'XZ':[], 'YZ':[]}
                                                 }
                        start_lines_at, end_lines_at = 0, num_sub 
                        for index, counter in enumerate(range(0, len(lines), num_sub)):
                            if index == 0:
                                lines_of_interest = lines[start_lines_at:end_lines_at]
                            else:
                                lines_of_interest = lines[end_lines_at+num_lines_to_skip: end_lines_at+num_lines_to_skip+num_sub]
                                end_lines_at = end_lines_at + num_sub + num_lines_to_skip
                            # read values into a dictionary            
                            if len(lines_of_interest) != 0:
                                master_dict[fold_name] = _populate_value(lines_of_interest, file_substr, master_dict[fold_name])

    os.makedirs('../csv_files', exist_ok=True)
    # merge csv files before saving
    if merge_folds:
        per_fold_data = {}
        for each_fold in master_dict:
            model_name = each_fold.split('k2d100-')[-1].split('-L1')[0]
            if model_name not in per_fold_data:
                per_fold_data[model_name] = { 'PSNR':{'XY': [], 'XZ':[], 'YZ':[]},
                                              'SSIM':{'XY': [], 'XZ':[], 'YZ':[]},
                                              'PDIST':{'XY': [], 'XZ':[], 'YZ':[]}
                                            }
            # combine data for each fold
            per_fold_data[model_name]['PSNR']['XY'].append(master_dict[each_fold]['PSNR']['XY']) 
            per_fold_data[model_name]['PSNR']['XZ'].append(master_dict[each_fold]['PSNR']['XZ']) 
            per_fold_data[model_name]['PSNR']['YZ'].append(master_dict[each_fold]['PSNR']['YZ'])
            per_fold_data[model_name]['SSIM']['XY'].append(master_dict[each_fold]['SSIM']['XY']) 
            per_fold_data[model_name]['SSIM']['XZ'].append(master_dict[each_fold]['SSIM']['XZ']) 
            per_fold_data[model_name]['SSIM']['YZ'].append(master_dict[each_fold]['SSIM']['YZ'])
            per_fold_data[model_name]['PDIST']['XY'].append(master_dict[each_fold]['PDIST']['XY']) 
            per_fold_data[model_name]['PDIST']['XZ'].append(master_dict[each_fold]['PDIST']['XZ']) 
            per_fold_data[model_name]['PDIST']['YZ'].append(master_dict[each_fold]['PDIST']['YZ'])      
        _save_csv(per_fold_data, num_sub, merge_folds, '../csv_files')
    else:
        _save_csv(master_dict, num_sub, merge_folds, '../csv_files')


if __name__ == "__main__":
    path_to_logs = 'path_to_log_files'
    get_avg_values(path_to_logs, file_substr='test', start_from=0, num_sub=32, num_lines_to_skip=1, merge_folds=False)
    