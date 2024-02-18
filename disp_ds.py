from glob import glob
import os.path as osp
import frame_utils
from PIL import Image

def load_cos_dataset(dataset_name, root):
    dataset = {}
    dataset['train'] ={}
    if dataset_name == 'sintel':
        image1_list = sorted( glob(osp.join(root, 'training/*_left/*/frame_*.png')) )
        disp_list = sorted( glob(osp.join(root, 'training/disparities/*/frame_*.png')) )
        image_list = []
        disparity_list = []
        disp_reader=frame_utils.readDispSintelStereo
        for img1_path, disp_path in zip(image1_list[:10], disp_list[:10]):
            assert img1_path.split('/')[-2:] == disp_path.split('/')[-2:]
            # img1 = Image.open(img1_path)
            # disp = disp_reader(disp_path)
            # image_list += [img1]
            # disparity_list += [disp]

            image_list += [img1_path]
            disparity_list += [disp_path]
        
        
        # dataset['img'] = image_list
        # dataset['disp'] = disparity_list
        dataset['train']['img'] = image_list
        dataset['train']['disp'] = disparity_list
        

    return dataset['train']