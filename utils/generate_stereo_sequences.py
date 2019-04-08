import os
import argparse

parser = argparse.ArgumentParser(description='just get k')
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--kitti_dir', default='/home/pp456/KITTI/raw/')
args = parser.parse_args()

val_folders = ['2011_09_26_drive_0056','2011_09_26_drive_0104','2011_09_29_drive_0026','2011_09_30_drive_0027','2011_10_03_drive_0042']

def check_if_val(r):
    for f in val_folders:
        if f in r:
            return True
    return False

train_file = open('train_stereo_sequences_'+str(args.k)+'.txt','w')
val_file = open('val_stereo_sequences_'+str(args.k)+'.txt','w')

for r,_,fs in os.walk(args.kitti_dir):
    if 'image_02/data' in r:
        if check_if_val(r):
            split_file = val_file
        else:
            split_file = train_file
        for i in range(len(fs)-args.k):
            sequence = fs[i:i+args.k]
            for f in sequence:
                image_02 = f
                image_03 = "image_03".join(f.split("image_02"))
                split_file.write(image_02+" "+image_03+"\n")
            split_file.write("\n")