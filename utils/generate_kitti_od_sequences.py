import numpy as np

k = 3

mapping_f = open("mapping/train_mapping.txt","r")
train_f = open("data/train_kitti_od_sequences_"+str(k)+".txt","w")
val_f = open("data/val_kitti_od_sequences_"+str(k)+".txt","w")

subfolder_ids = {}

for idx,line in enumerate(mapping_f):
    line = line[:-1].split(" ")
    if line[1] in subfolder_ids:
        subfolder_ids[line[1]].append((idx,line))
    else:
        subfolder_ids[line[1]] = [(idx,line)]

for subfolder in subfolder_ids:
    if len(subfolder_ids[subfolder]) > 2:
        flip = np.random.binomial(1,0.7)
        train = (flip==1)
        fs = subfolder_ids[subfolder]
        for i in range(len(fs)-2):
            seq = fs[i:i+3]
            if int(seq[0][1][2])+1 == int(seq[1][1][2]) and int(seq[1][1][2])+1 == int(seq[2][1][2]):
                if train:
                    for scene in seq:
                        train_f.write(str(scene[0]).zfill(6)+".txt")
                        train_f.write("\n")
                    train_f.write("\n")
                else:
                    for scene in seq:
                        val_f.write(str(scene[0]).zfill(6)+".txt")
                        val_f.write("\n")
                    val_f.write("\n")




