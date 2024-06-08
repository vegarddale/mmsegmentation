# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import math
import os
import os.path as osp
import tempfile
import zipfile
import random
import mmcv
import numpy as np
from mmengine.utils import ProgressBar, mkdir_or_exist
from collections import Counter
import cv2

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert potsdam dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='potsdam folder path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=3000)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=2000)
    args = parser.parse_args()
    return args


# oversample images containing the most underrepresented classes
# channels must equal 3 and we need to crop both the image and the labels
# and we need to make sure the image name is unique
def oversample(label_dir, seg_map_path, out_dir, data_type):
    
    # Getting the parent directory
    parent_directory = osp.dirname(label_dir)
    img_dir = None
    dirs = os.listdir(parent_directory)
    for dir in dirs:
        if dir != 'labels' and not dir.startswith('VD') and not dir.startswith('.'):
            img_dir = dir
    
    
    
    seg_map = mmcv.imread(osp.join(label_dir, seg_map_path), flag='grayscale')
    image =  mmcv.imread(osp.join(f"{parent_directory}/{img_dir}", seg_map_path))
    
    underrepresented_classes = [2,4,5,6,7]
    if not np.isin(seg_map, underrepresented_classes).any():
        print("not in underrepresented classes")
        return
    print(seg_map_path)
    present_classes = np.unique(seg_map[np.isin(seg_map, underrepresented_classes)])
    
    for cls in present_classes:
        # Find indices where image contains underrepresented classes
        class_indices = np.where(seg_map == cls)
        tmp_image = np.zeros((seg_map.shape[0], seg_map.shape[1]), dtype=np.uint8)
        tmp_image[class_indices] = 255

        # locate different instances of objects
        contours, _ = cv2.findContours(tmp_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            top_left_x, top_left_y, w, h = cv2.boundingRect(contour)
            
            for dx, dy in [(-500, -500), (500, -500), (-500, 500), (500, 500)]:
                crop_x1 = max(top_left_x + dx, 0)
                crop_y1 = max(top_left_y + dy, 0)
                if crop_x1 > seg_map.shape[1] - 3000:
                    crop_x1 = seg_map.shape[1] - 3000
                if crop_y1 > seg_map.shape[0] - 3000:
                    crop_y1 = seg_map.shape[0] - 3000
                crop_x2 = min(crop_x1 + 3000, seg_map.shape[1])
                crop_y2 = min(crop_y1 + 3000, seg_map.shape[0])
                crop_seg_map = seg_map[crop_y1:crop_y2, crop_x1:crop_x2]
                crop_image = image[crop_y1:crop_y2, crop_x1:crop_x2]#TODO
                assert crop_seg_map.shape == (3000, 3000), "cropped segmentation map does not match output size" 
                mmcv.imwrite(
                crop_seg_map.astype(np.uint8),
                osp.join(
                    osp.join(out_dir, 'ann_dir_2', data_type),
                    f'{dx}_{dy}_{seg_map_path}'))
                mmcv.imwrite(
                crop_image.astype(np.uint8),
                osp.join(
                    osp.join(out_dir, 'img_dir_2', data_type),
                    f'{dx}_{dy}_{seg_map_path}'))



def clip_big_image(image_path, clip_save_dir, args, to_label=False):

    image = None
    h = 0
    w = 0
    c = 1
    if to_label:
        image = mmcv.imread(image_path, flag='grayscale')
        h, w = image.shape
    else:
        image = mmcv.imread(image_path)
        h, w, c = image.shape
    
    stride_size = args.stride_size
    clip_size = args.clip_size
    
    num_rows = math.ceil((h - clip_size) / stride_size) if math.ceil(
        (h - clip_size) /
        stride_size) * stride_size + clip_size >= h else math.ceil(
            (h - clip_size) / stride_size) + 1
    num_cols = math.ceil((w - clip_size) / stride_size) if math.ceil(
        (w - clip_size) /
        stride_size) * stride_size + clip_size >= w else math.ceil(
            (w - clip_size) / stride_size) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * clip_size
    ymin = y * clip_size

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + clip_size > w, w - xmin - clip_size,
                           np.zeros_like(xmin))
    ymin_offset = np.where(ymin + clip_size > h, h - ymin - clip_size,
                           np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + clip_size, w),
        np.minimum(ymin + clip_size, h)
    ],
                     axis=1)

    # if to_label:
    #     color_map = np.array([
    #                         [0,0,0],
    #                         [255, 0, 0],
    #                         [255, 255, 102],
    #                         [153, 102, 51],
    #                         [204, 204, 204],
    #                         [102, 102, 102],
    #                         [255, 255, 255],
    #                         [204, 153, 102],
    #                         [51, 153, 102]])

    #     flatten_v = np.matmul(
    #         image.reshape(-1, c),
    #         np.array([2, 3, 4]).reshape(3, 1))
    #     out = np.zeros_like(flatten_v)
    #     for idx, class_color in enumerate(color_map):
    #         value_idx = np.matmul(class_color,
    #                               np.array([2, 3, 4]).reshape(3, 1))
    #         out[flatten_v == value_idx] = idx
    #     image = out.reshape(h, w)

    for box in boxes:
        start_x, start_y, end_x, end_y = box
        clipped_image = image[start_y:end_y,
                              start_x:end_x] if to_label else image[
                                  start_y:end_y, start_x:end_x, :]
        idx = osp.basename(image_path).split('.')[0]
        mmcv.imwrite(
            clipped_image.astype(np.uint8),
            osp.join(
                clip_save_dir,
                f'{idx}_{start_x}_{start_y}_{end_x}_{end_y}.png'))
    


def get_all_images(data_dir):

    all_images = []
    data_dir = "./data/shorelines/all_images"
    for img_root_dir in os.listdir(data_dir):
        path = osp.join(data_dir, img_root_dir)
        for img_dir in os.listdir(path):
            if(img_dir=="labels"):
                continue
            if osp.isdir(osp.join(path, img_dir)):
                for img in os.listdir(osp.join(path, img_dir)):
                    if img.endswith("png"):
                        all_images.append(img)
    return all_images


'''
data.zip
- VD_*
- - - labels
- - - imgfolder

'''


def main():
    args = parse_args()
    
    # splits = {
    #     'train': [
    #         '2_10', '2_11', '2_12', '3_10', '3_11', '3_12', '4_10', '4_11',
    #         '4_12', '5_10', '5_11', '5_12', '6_10', '6_11', '6_12', '6_7',
    #         '6_8', '6_9', '7_10', '7_11', '7_12', '7_7', '7_8', '7_9'
    #     ],
    #     'val': [
    #         '5_15', '6_15', '6_13', '3_13', '4_14', '6_14', '5_14', '2_13',
    #         '4_15', '2_14', '5_13', '4_13', '3_14', '7_13'
    #     ]
    # }

    # dataset_path = args.dataset_path
    dataset_path = "./data/shorelines/all_images"
    all_images = get_all_images(dataset_path)
    print(len(all_images))
    random.shuffle(all_images)
    print(len(all_images))
    train_val_split = 0.7
    
    splits = {
        'train': all_images[:int(len(all_images)*train_val_split)],
        'val': all_images[int(len(all_images)*train_val_split):]
    }
    
    
    train_list = splits['train']
    element_counts = Counter(train_list)
    duplicates = [element for element, count in element_counts.items() if count > 1]
    print(duplicates)
    assert len(duplicates) == 0, "Found duplicate images, stopping..."

    print("nof train images: ", len(splits['train']))
    print("nof val images: ", len(splits['val']))
    
    if args.out_dir is None:
        out_dir = osp.join('data', 'shorelines')
    else:
        out_dir = args.out_dir
    
    print('Making directories...')
    mkdir_or_exist(osp.join(out_dir, 'img_dir_2', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir_2', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir_2', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir_2', 'val'))
    
    for img_root_dir in os.listdir(dataset_path):
        path = osp.join(dataset_path, img_root_dir)
        for img_dir in os.listdir(path):
            if osp.isdir(osp.join(path, img_dir)):
                img_dir_path = osp.join(path, img_dir)
                for img in os.listdir(img_dir_path):
                    if f'{img}' in splits['train']:
                        data_type = 'train'
                    elif f'{img}' in splits['val']:
                        data_type = 'val'
                    else:
                        print(f'{img}')
                        continue
                    if 'label' in img_dir:
                        dst_dir = osp.join(out_dir, 'ann_dir_2', data_type)
                        clip_big_image(osp.join(img_dir_path, img), dst_dir, args, to_label=True)
                        oversample(img_dir_path, img, out_dir, data_type)
                    else:
                        dst_dir = osp.join(out_dir, 'img_dir_2', data_type)
                        clip_big_image(osp.join(img_dir_path, img), dst_dir, args, to_label=False)
        

    print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    main()
