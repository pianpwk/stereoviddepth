from shutil import copyfile
import os
import numpy as np
import kitti_util
import kitti_object
from warp import *
import imageio

c = 0
for r,_,fs in os.walk('/share/nikola/export/yw763/Dataset/kitti_od/training/image_2/'):
    for f in fs:
        print(c)
        c += 1
        copyfile(r+'/'+f,'kitti_od/image_2/'+f)
        copyfile('image_3'.join(r.split('image_2'))+f,'kitti_od/image_3/'+f)
        vel = np.fromfile('velodyne'.join(r.split('image_2'))+f[:-4]+'.bin',dtype=np.float32).reshape([-1,4])[:,:3]
        cal = kitti_util.Calibration('calib'.join(r.split('image_2'))+f[:-4]+'.txt')
        lidar = assign_img_for_pc(vel,imageio.imread(r+'/'+f),cal)
        lidar = np.where(lidar>0.0,0.54*721/lidar,0.0)
        imageio.imsave('kitti_od/disp/'+f,np.array(lidar*256,dtype=np.uint16))

