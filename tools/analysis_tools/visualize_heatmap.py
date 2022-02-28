import argparse
import os
import numpy as np
import torch
import cv2
from PIL import Image
from mmcv import Config, DictAction
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize feature heatmap after FPN.')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--cam type', default='bbox_weight', help='cam tpye to choose.')
    parser.add_argument(
        '--feature',
        default='bbox',
        help='feature to visualize.'
        'Support bbox and roi(two stage detectors with roi pooling).')
    parser.add_argument(
        '--stage',
        default='one',
        help='detector types.'
        'Support one or two stage detectors.')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='Path to save visualized result.')

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    assert args.stage == 'one' == args.feature == 'bbox'
    # print(model._modules)
    # print(model._modules.get('neck').out_convs[0])
    features = []
    def forward_hook(module, input, output):
        features.append(output.cpu().numpy())
    for layer in model._modules.get('neck').out_convs:
        layer.register_forward_hook(forward_hook)
    result = inference_detector(model, args.img)
    # print(result)

    conv_T = ['reg', 'cls']
    M = 1
    
    params_dict = model.state_dict()
    # print(params_dict.keys())
    param_bboxes = []
    for i, _ in enumerate(features):
        param_bbox = params_dict['bbox_head.multi_level_{}_convs.{}.0.conv.weight'.format(conv_T[M], i)]
        param_bbox = torch.mean(param_bbox, dim=3)
        param_bbox = torch.mean(param_bbox, dim=2)
        param_bbox = torch.mean(param_bbox, dim=0)
        param_bbox = param_bbox.cpu().numpy()
        param_bboxes.append(param_bbox)



    #     param_roi = params_dict['roi_head.conv3x3.weight']
    #     # param_roi.shape = [N, C_in, H, W] = [256,256,3,3]
    #     param_roi = torch.mean(param_roi, dim=3)
    #     param_roi = torch.mean(param_roi, dim=2)
    #     param_roi = torch.mean(param_roi, dim=0)
    #     param_roi = param_roi.cpu().numpy()
    # elif args.feature == 'bbox':
    #     param_bbox = params_dict['bbox_head.fc6.weight']
    #     # param_bbox.shape = [C_out, C_in] = [512,9408]
    #     cout, cin = param_bbox.shape
    #     param_bbox = param_bbox.view(cout,-1,7,7)
    #     param_bbox = torch.mean(param_bbox, dim=3)
    #     param_bbox = torch.mean(param_bbox, dim=2)
    #     param_bbox = torch.mean(param_bbox, dim=0)
    #     param_bbox = param_bbox.cpu().numpy()

    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        size_upsample = (1024, 1024)
        bz, nc, h, w = feature_conv.shape
        cam = weight_softmax.dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # normalize
        cam_img = 255 - np.uint8(255 * cam_img)
        output_cam = cv2.resize(cam_img, size_upsample)
        return output_cam
    img = cv2.imread(args.img)
    height, width, _ = img.shape

    filename_ext = os.path.basename(args.img)
    filename,ext = os.path.splitext(filename_ext)
    
    file_cam_dir = os.getcwd() + '/cam_results/' + filename
    if not(os.path.exists(file_cam_dir)):
        os.makedirs(file_cam_dir)

    IMAGE_COLUMN = 1
    IMAGE_ROW = len(features)
    # to_image = Image.new('RGB', (IMAGE_COLUMN * width, IMAGE_ROW * height))
    show_result_pyplot(model, img, result, score_thr=0.4)
    for i, f in enumerate(features):
        CAM = returnCAM(f, param_bboxes[i], 0)
        heatmap = cv2.applyColorMap(cv2.resize(CAM,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite(file_cam_dir + '/CAM_bbox_%s_layer%d.jpg'%(conv_T[M], i), result)
        # result32=result.astype(np.uint8)
        # from_image = Image.fromarray(cv2.cvtColor(result32, cv2.COLOR_BGR2RGB))
        # to_image.paste(from_image, (1 * width, i * height))

    # to_image.save(file_cam_dir + '/stitch.jpg')
if __name__ == '__main__':
    main()
