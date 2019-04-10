import os
import argparse

parser = argparse.ArgumentParser(description='get directory')
parser.add_argument('--in_dir', default='/home/pp456/KITTI/training/')
parser.add_argument('--out_dir', default='data/')
args = parser.parse_args()

lines = []

for r,_,fs in os.walk(args.in_dir):
    if r[-7:] == 'image_2':
        for f in fs:
            L_path = r+'/'+f
            R_path = 'image_3'.join(L_path.split('image_2'))
            disp_path = 'disp_occ_0'.join(L_path.split('image_2'))
            
            if '_10.png' in L_path:
                lines.append(L_path+" "+R_path+" "+disp_path)

trainfile = open(os.path.join(args.out_dir,'train_supervised.txt'),'w')
valfile = open(os.path.join(args.out_dir,'val_supervised.txt'),'w')

for line in lines[:int(len(lines)*0.8)]:
    trainfile.write(line)
    trainfile.write("\n")

for line in lines[int(len(lines)*0.8):]:
    valfile.write(line)
    valfile.write("\n")
