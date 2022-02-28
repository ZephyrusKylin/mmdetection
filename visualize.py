from __future__ import division


from PIL import Image
import os
from pod.apis.inference import Predictor
from pod.utils.dist_helper import finalize, setup_distributed

from .subcommand import Subcommand
from pod.utils.registry_factory import SUBCOMMAND_REGISTRY
import numpy as np
import cv2
import torch

@SUBCOMMAND_REGISTRY.register('visualize')
class Visualize(Subcommand):
    def add_subparser(self, name, parser):
        sub_parser = parser.add_parser(name, help='sub-command for visualize')
        sub_parser.add_argument('--config', type=str, required=True, help='path to yaml config')
        sub_parser.add_argument('--work_dir', type=str, default='./', help='path to work directory')
        sub_parser.set_defaults(run=_main)

        return sub_parser

def _main(args):
    setup_distributed()
    predictor = Predictor(args.config, args.work_dir)
    
    filepath = '/home/SENSETIME/mayan1/Projects/POD/experiments/visualization/Image/shuihua/train#长宁0712#水花飞溅11.PNG'
    output = predictor.predict([filepath])
    features=[f.cpu().numpy() for f in output['features']] 

    model=predictor.detector
    print(model)
    
    # get the softmax weight
    params_dict = model.state_dict()
    param_roi = params_dict['roi_head.conv3x3.weight']
    # param_roi.shape = [N, C_in, H, W] = [256,256,3,3]
    param_roi = torch.mean(param_roi, dim=3)
    param_roi = torch.mean(param_roi, dim=2)
    param_roi = torch.mean(param_roi, dim=0)
    param_roi = param_roi.cpu().numpy()

    param_bbox = params_dict['bbox_head.fc6.weight']
    # param_bbox.shape = [C_out, C_in] = [512,9408]
    cout, cin = param_bbox.shape
    param_bbox = param_bbox.view(cout,-1,7,7)
    param_bbox = torch.mean(param_bbox, dim=3)
    param_bbox = torch.mean(param_bbox, dim=2)
    param_bbox = torch.mean(param_bbox, dim=0)
    param_bbox = param_bbox.cpu().numpy()
    
    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        size_upsample = (256, 256)
        bz, nc, h, w = feature_conv.shape
        cam = weight_softmax.dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # normalize
        cam_img = 255 - np.uint8(255 * cam_img)
        output_cam = cv2.resize(cam_img, size_upsample)
        return output_cam

    img = cv2.imread(filepath)
    height, width, _ = img.shape
    
    filename_ext = os.path.basename(filepath)
    filename,ext = os.path.splitext(filename_ext)
    
    file_cam_dir = os.getcwd() + '/cam_results/' + filename
    if not(os.path.exists(file_cam_dir)):
        os.makedirs(file_cam_dir)

    IMAGE_COLUMN = 2
    IMAGE_ROW = len(features)
    to_image = Image.new('RGB', (IMAGE_COLUMN * width, IMAGE_ROW * height))


    for i, f in enumerate(features):
        CAM = returnCAM(f, param_roi, 0)
        heatmap = cv2.applyColorMap(cv2.resize(CAM,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite(file_cam_dir + '/CAM_roi_layer%d.jpg'%(i), result)
        result32=result.astype(np.uint8)
        from_image = Image.fromarray(cv2.cvtColor(result32, cv2.COLOR_BGR2RGB))
        to_image.paste(from_image, (0 * width, i * height))

    for i, f in enumerate(features):
        CAM = returnCAM(f, param_bbox, 0)
        heatmap = cv2.applyColorMap(cv2.resize(CAM,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite(file_cam_dir + '/CAM_bbox_layer%d.jpg'%(i), result)
        result32=result.astype(np.uint8)
        from_image = Image.fromarray(cv2.cvtColor(result32, cv2.COLOR_BGR2RGB))
        to_image.paste(from_image, (1 * width, i * height))

    to_image.save(file_cam_dir + '/stitch.jpg')
    finalize()