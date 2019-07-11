import os
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time
from params import PARAMS


def get_frames_data(filename):
    # 根据filename下存储的图片, 按照顺序提取16帧图片作为一个视频的截取, 进行训练
    num_frames_per_clip = PARAMS["num_frames_per_clip"]
    ret_arr = []
    s_index = 0
    for parent, dirnames, filenames in os.walk(filename):
        # filenames 中存储最底层的文件名 ['001.jpg', '002.jpg', ...]
        if(len(filenames) < num_frames_per_clip):
            return [], s_index
        filenames = sorted(filenames)
        s_index = random.randint(0, len(filenames) - num_frames_per_clip)
        for i in range(s_index, s_index + num_frames_per_clip):
            image_name = str(filename) + '/' + str(filenames[i])
            img = Image.open(image_name)
            img_data = np.array(img)         # numpy, shape=(240, 320, 3)
            ret_arr.append(img_data)
    return ret_arr, s_index


def read_clip_and_label(filename, shuffle=True):
    num_frames_per_clip = PARAMS["num_frames_per_clip"]
    crop_size = PARAMS["crop_size"]
    batch_size = PARAMS["batch_size"]

    # 
    lines = list(open(filename,'r'))
    read_dirnames = []
    data = []
    label = []
    batch_index = 0
    np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])

    # shuffle    
    video_indices = list(range(len(lines)))
    random.seed(time.time())
    random.shuffle(video_indices)
    
    for index in video_indices:
        if(batch_index >= batch_size):
            break
        line = lines[index].strip('\n').split()
        dirname = line[0]
        tmp_label = line[1]

        # 读取图片
        tmp_data, _ = get_frames_data(dirname)
        if len(tmp_data) == 0:
            continue

        # 依次对读取的图片进行截取, 截取中心的 (112,112) 区域
        img_datas = []
        for j in range(len(tmp_data)):                             # len(tmp_data)=16
            img = Image.fromarray(tmp_data[j].astype(np.uint8))
            if img.width > img.height:
                # 原来 (width, height)=(320, 240), 现在为 (150, 112) 
                scale = float(crop_size)/float(img.height)
                img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
            else:
                scale = float(crop_size)/float(img.width)
                img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
            # img.shape = (112, 150, 3), crop_x=0, crop_y=16
            crop_x = int((img.shape[0] - crop_size)/2)
            crop_y = int((img.shape[1] - crop_size)/2)
            # 提取中心位置的图像部分, 转换后 shape=(112,112,3)
            img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean[j]
            img_datas.append(img)  # 循环结束后, img_datas包含一个视频片段中16个时间步的图片, length=16, 每个元素shape=(112,112,3)
        # 循环结束后data中包含一个批量10个视频,转为numpy后 shape=(10,16,112,112,3)
        data.append(img_datas)
        label.append(int(tmp_label))
        read_dirnames.append(dirname)
        batch_index += 1

    # pad (duplicate) data/label if less than batch_size
    valid_len = len(data)
    pad_len = batch_size - valid_len
    if pad_len:
        for i in range(pad_len):
            data.append(img_datas)
            label.append(int(tmp_label))

    np_arr_data = np.array(data).astype(np.float32) 
    np_arr_label = np.array(label).astype(np.int64)

    return np_arr_data, np_arr_label, read_dirnames, valid_len


# if __name__ == '__main__':
#     for _ in range(20):
#         train_images, train_labels, read_dirnames, valid_len = read_clip_and_label(filename='list/train.list')
#         print(train_labels)
#     