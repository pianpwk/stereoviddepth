import numpy as np
from shutil import copyfile
import imageio
from warp import assign_img_for_pc
from kitti_util import Calibration
import os

superv = open('../city_training/superv.txt','w')
unsuperv = open('../city_training/unsuperv.txt','w')

iter_count = 0
for r,_,fs in os.walk('/home/wc635/KITTI_dream/raw/object/city_training/image_2/'):
    for f in fs:
        print(iter_count)
        iter_count += 1
        img_L = r+'/'+f
        img_R = 'image_3'.join(r.split('image_2'))+'/'+f
        disp = 'velodyne'.join(r.split('image_2'))+'/'+f[:-3]+'bin'
        cal = 'calib'.join(r.split('image_2'))+'/'+f[:-3]+'txt'

        copyfile(img_L,'../city_training/image_2/'+f)
        copyfile(img_R,'../city_training/image_3/'+f)
        cloud = assign_img_for_pc(np.fromfile(disp,dtype=np.float32).reshape([-1,4])[:,:3],imageio.imread(img_L),Calibration(cal))
        disp = np.where(cloud>0.0,0.54*721/cloud,0.0)
        disp = np.array(disp*256,dtype=np.uint16)
        imageio.imsave('../city_training/disp/'+f,disp) 

        superv.write('city_training/image_2/'+f+' '+'city_training/image_3/'+f+' '+'city_training/disp/'+f)
        unsuperv.write('city_training/image_2/'+f+' '+'city_training/image_3/'+f)
        unsuperv.write("\n")
 
