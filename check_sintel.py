from glob import glob
import os.path as osp
import frame_utils
from PIL import Image
import os
import numpy as np

root_path = '/media/neuralmaster/9d5af100-a900-4e89-bab1-43c8b5025daf/neuromaster/Documents/haozheng/StereoDS/MPI-Sintel-stereo-training-20150305/training/disparities/alley_1'


disp_path = os.path.join(root_path,'frame_0001.png')
disp_reader=frame_utils.readDispSintelStereo
disp = disp_reader(disp_path)
print(disp.shape)
print(np.max(disp))
print(np.min(disp))
