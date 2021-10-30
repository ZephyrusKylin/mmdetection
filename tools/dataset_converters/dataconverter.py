import argparse
import os
import json

import mmcv
from PIL import Image
import numpy as np
import xml.etree.cElementTree as ET

import random
import shutil

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert DOTA dataset or rolabel dataset to coco format or custom format or YOLOv5 format')
    parser.add_argument('xml_dir', type=str, help='The root path of images')
    parser.add_argument('out_dir', type=str, help='The root path of images')
    # parser.add_argument(
    #     'classes', type=str, help='The text file name of storage class list')
    # parser.add_argument(
    #     'out',
    #     type=str,
    #     help='The output annotation json file name, The save dir is in the '
    #     'same directory as img_path')
    args = parser.parse_args()
    return args

def load_xml(tree):
    
    root1 = tree.getroot()

    object_out = dict()
    file_name = root1.find('filename').text
    imagesize = root1.find('size')  # find(match)查找第一个匹配的子元素， match可以时tag或是xpaht路径
    imageWidth = int(imagesize.find('width').text)
    imageHeight = int(imagesize.find('height').text)

    object_out.setdefault('filename', file_name)
    object_out.setdefault('height', imageHeight)
    object_out.setdefault('width', imageWidth)



    ann = {'bboxes': [],
            'rbboxes': [],
            'labels': [],
            'bboxes_ignore': [],
            'rbboxes_ignore': [],
            'labels_ignore': []}
    for obj in tree.findall('object'):
        
        
        Type = obj.find('type').text
        name = obj.find('name').text

        ignore = False
        if name == 'other-ship' or name == 'ship':
            continue
        elif name == 'invalid':
            ignore = True
            name = 'plane'
        else:
            name = 'plane'

        difficult = obj.find('difficult').text

        if difficult == 1:
            ignore = True

        if Type == 'bndbox':
            bbox = obj.find('bndbox')
            bb = [int(bbox.find('xmin').text),
                                    int(bbox.find('ymin').text),
                                    int(bbox.find('xmax').text),
                                    int(bbox.find('ymax').text)]
            if ignore:

                ann['bboxes_ignore'].append(bb)
                ann['labels_ignore'].append(name)
            else:


                ann['bboxes'].append(bb)
                ann['labels'].append(name)
        elif Type == 'robndbox':
            # if obj_struct['name'] != 'Helicopter':   # 不统计直升机
            bbox = obj.find('robndbox')
            rbb = [float(bbox.find('cx').text),
                                    float(bbox.find('cy').text),
                                    float(bbox.find('w').text),
                                    float(bbox.find('h').text),
                                    float(bbox.find('angle').text)]
            if ignore:

                ann['rbboxes_ignore'].append(rbb)
                # ann['labels_ignore'].append(name)## equal to bbox
            else:

                ann['rbboxes'].append(rbb)
                # ann['labels'].append(name)
        
    object_out.setdefault('ann', ann)
    # print(object_out)
    return object_out

def merge_custom(custom_path_1, custom_path_2, out_path):
    custom_1 = open(custom_path_1, 'r')
    custom_2 = open(custom_path_2, 'r')
    anno1 = json.load(custom_1)
    anno2 = json.load(custom_2)
    out_list = anno1 + anno2
    out_file = open(out_path, 'w')
    json.dump(out_list, out_file)
    out_file.close()
    custom_1.close()
    custom_2.close()


def rolabelXML2Coco(xml_dir, rotate_label=False, distinguishability=0.5):
    if rotate_label:
        pass
    else:
        for xml_file in os.listdir(xml_dir):
            if xml_file.endswith('.xml'):
                tree = ET.ElementTree(file=xml_file)
                obj = load_xml(tree)

    return 

"""
def rolabelXML2Custom(xml_dir, label_ids, rotate_label=False, distinguishability=0.5):
    out_list = list()
    for xml_file in xml_dir:
        load_xml()
        tree = ET.parse(xml_dir)
        root = tree.getroot()
        file_name = root.find('filename').text
        name_clean = file_name.rstrip('.tif')
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            label = label_ids[name]
            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]
            if difficult:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        annotation = {
            'filename': file_name,
            'width': w,
            'height': h,
            'ann': {
                'bboxes': bboxes.astype(np.float32),
                'labels': labels.astype(np.int64),
                'bboxes_ignore': bboxes_ignore.astype(np.float32),
                'labels_ignore': labels_ignore.astype(np.int64)
            }
        }
        out_list.append(annotation)
    return out_list
"""
def custom2coco(custom_file, coco_path):
    return

def rolabelXML2DOTA(xml_dir, out_dir, rotate_label=False, distinguishability=0.5):
    k = 0
    for xml_file in os.listdir(xml_dir):
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        annotations = load_xml(tree)

        fn = annotations['filename']
        fn_out = fn.split('.')[0] + '.txt'

        f = open(os.path.join(out_dir, fn_out), 'w')
        f.write('imagesource:imagesource\n')
        f.write('gsd:null\n')

        if not rotate_label:
            ann = annotations['ann']
            bboxes = ann['bboxes']
            labels = ann['labels']
            # print(len(bboxes), len(labels))
            assert len(bboxes) == len(labels)
            

            for i, bb in enumerate(bboxes):
                x1, y1, x2, y2 = bb
                dis = distinguishability/0.5
                x1, y1, x2, y2 = int(x1/dis), int(y1/dis), int(x2/dis), int(y2/dis)
                category = labels[i]
                difficult = 0
                outstr ='{} {} {} {} {} {} {} {} {} {}\n'.format(x1, y1, x2, y1, x2, y2, x1, y2, category, difficult)
                f.write(outstr)
            
            bboxes_ignore = ann['bboxes_ignore']
            labels_ignore = ann['labels_ignore']
            for j, bb in enumerate(bboxes_ignore):
                x1, y1, x2, y2 = bb
                dis = distinguishability/0.5
                x1, y1, x2, y2 = int(x1/dis), int(y1/dis), int(x2/dis), int(y2/dis)
                category = labels_ignore[j]
                difficult = 1
                outstr ='{} {} {} {} {} {} {} {} {} {}\n'.format(x1, y1, x2, y1, x2, y2, x1, y2, category, difficult)
                f.write(outstr)
        f.close()
        # break
        print(k)
        k += 1

def random_train_val(ori_dir, out_dir):
    random.seed(10)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    os.makedirs(os.path.join(out_dir, 'train'))
    os.makedirs(os.path.join(out_dir, 'val'))

    for i in os.listdir(ori_dir):
        T = random.randint(1,10)
        if T > 8:
            shutil.copy(os.path.join(ori_dir, i), os.path.join(out_dir, 'val'))
        else:
            shutil.copy(os.path.join(ori_dir, i), os.path.join(out_dir, 'train'))
def main():
    
    args = parse_args()
    # classes = args.classes
    # label_ids = {name: i for i, name in enumerate(classes)}
    # rolabelXML2DOTA(args.xml_dir, args.out_dir, distinguishability=1.5)
    random_train_val(args.xml_dir, args.out_dir)
if __name__ == '__main__':
    main()
