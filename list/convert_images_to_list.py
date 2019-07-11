import numpy as np
import os
from absl import flags
import click

@click.command()
@click.option('--ucf_dir', type=str, default="/home/chenjia/ssd/video-recognition/UCF-101", help='UCF-101 dataset')

def convert(ucf_dir):
    assert os.path.exists(ucf_dir)
    ratio = 1/5.      # ratio of test data

    class_name = sorted(os.listdir(ucf_dir))
    print("num of class:", len(class_name))

    train_files, test_files = [], []
    for c in range(len(class_name)):
        dirname = os.path.join(ucf_dir, class_name[c])
        filenames = np.array(sorted(os.listdir(dirname)))
        file_num = filenames.shape[0]
        print("class num:", c, ", class:", class_name[c], ", number:", file_num, end=",")

        # split training set and test set
        rand = np.random.random(file_num)
        train_list = rand > ratio
        test_list  = rand <= ratio
        
        train_file = filenames[train_list]
        test_file  = filenames[test_list]
        print(" train num:", len(train_file), ", test num:", len(test_file))

        train_files.append([os.path.join(ucf_dir, dirname, f) for f in train_file]) 
        test_files.append([os.path.join(ucf_dir, dirname, f) for f in test_file])

    with open("train.list", "w") as f:
        for i in range(len(train_files)):
            for j in range(len(train_files[i])):
                f.write(train_files[i][j] + " " + str(i) + "\n")


    with open("test.list", "w") as f:
        for i in range(len(test_files)):
            for j in range(len(test_files[i])):
                f.write(test_files[i][j] + " " + str(i) + "\n")

if __name__ == '__main__':
    convert()
