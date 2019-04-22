import os
import argparse

parser = argparse.ArgumentParser(description='just get k')
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--kitti_dir', default='/home/pp456/KITTI/raw/')
parser.add_argument('--max_seq', default=None)
args = parser.parse_args()

val_folders = ['2011_09_26_drive_0056','2011_09_26_drive_0104','2011_09_29_drive_0026','2011_09_30_drive_0027','2011_10_03_drive_0042']

def check_if_val(r):
    for f in val_folders:
        if f in r:
            return True
    return False

train_file = open('data/train_stereo_sequences_'+str(args.k)+'.txt','w')
val_file = open('data/val_stereo_sequences_'+str(args.k)+'.txt','w')

total_seq = 0
for r,_,fs in os.walk(args.kitti_dir):
    if 'image_02/data' in r:
        if check_if_val(r):
            split_file = val_file
        else:
            split_file = train_file
        fs = sorted(fs)
        if not args.max_seq is None and total_seq >= int(args.max_seq):
            break
        for i in range(len(fs)-args.k):
            sequence = fs[i:i+args.k]
            to_write = []
            skip = False
            for f in sequence:
                image_02 = r+"/"+f
                image_03 = "image_03".join(image_02.split("image_02"))                
                to_write.append(image_02+" "+image_03+"\n")
                if not os.path.isfile(image_03):
                    skip = True

            if not skip:
                for tw in to_write:
                    split_file.write(tw)
                split_file.write("\n")
        total_seq += 1
        print(total_seq)
