import os
import random
import nrrd
import matplotlib.pyplot as plt
from skimage.transform import rotate

def main(path_to_predictions, path_to_LR, path_to_HR, num_cases, path_to_save):
    predictions = os.listdir(path_to_predictions)
    LR, HR = [], []
    for arg_lr, arg_hr in zip(os.listdir(path_to_LR), os.listdir(path_to_HR)):
        if arg_lr.endswith('.nrrd'):
            LR.append(arg_lr)
        if arg_hr.endswith('.nrrd'):
            HR.append(arg_hr)
    
    # folder to save samples
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    # select required number of cases
    random_case_idx = random.sample(range(0, len(predictions)-1), num_cases)
    for case in random_case_idx:
        print("Running case {}".format(predictions[case]))
        # read data
        prediction, pd_header = nrrd.read(os.path.join(path_to_predictions, predictions[case]))
        lr_GT, lr_header = nrrd.read(os.path.join(path_to_LR, LR[case]))
        hr_GT, hr_header = nrrd.read(os.path.join(path_to_HR, HR[case]))
        # get center slice
        ctr_slice = prediction.shape[2]//2
        # draw figure
        f, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(20, 10), dpi=200)
        ax0.imshow(rotate(lr_GT[:, :, ctr_slice], -90), cmap='gray')
        ax0.set_title('K1-D10')
        ax0.axis('off')
        ax1.imshow(rotate(prediction[:, :, ctr_slice], -90), cmap='gray')
        ax1.set_title('GAN-K2-D100')
        ax1.axis('off')
        ax2.imshow(rotate(hr_GT[:, :, ctr_slice], -90), cmap='gray')
        ax2.set_title('K2-D100')
        ax2.axis('off')
        # save images
        if path_to_save:
            file_name = predictions[case].split(".nrrd")[0] + ".png"
            file_save_path = os.path.join(path_to_save, file_name)
            f.savefig(file_save_path)
            f.clf()
    print("All files saved!")


if __name__ == "__main__":
    path_to_predictions = "path_to_model_normalized"
    path_to_LR = "path_to_unnormalized"
    path_to_HR = "path_to_GT_normalized"
    save_path = "path_to_save_slices"
    num_cases = 20
    main(path_to_predictions, path_to_LR, path_to_HR, num_cases, save_path)