import torch
import argparse
from models.experimental import attempt_load
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from pathlib import Path

def clip_coords(boxes, shape):

    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])  
        boxes[:, 1].clamp_(0, shape[0])  
        boxes[:, 2].clamp_(0, shape[1])  
        boxes[:, 3].clamp_(0, shape[0])  
    else:  
        boxes[:, 0] = np.clip(boxes[:, 0], 0, shape[1])  
        boxes[:, 1] = np.clip(boxes[:, 1], 0, shape[0])  
        boxes[:, 2] = np.clip(boxes[:, 2], 0, shape[1])  
        boxes[:, 3] = np.clip(boxes[:, 3], 0, shape[0]) 

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):

    if ratio_pad is None: 
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[..., [0, 2]] -= pad[0]  
    coords[..., [1, 3]] -= pad[1]  
    coords[..., :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def load_models(weights, device):
    models = []
    for weight in weights:
        model = torch.load(weight, map_location=device)['model'].float()  # Load model
        model.to(device).eval()
        models.append(model)
    return models

def ensemble_predict(models, img, conf_thres=0.25, iou_thres=0.45, agnostic_nms=False):
    ensemble_output = []
    for model in models:
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=agnostic_nms)
        ensemble_output.append(pred)
    return ensemble_output

def merge_ensemble_predictions(predictions):
    merged_pred = []
    for preds in zip(*predictions):
        merged_pred.append(torch.cat(preds, dim=0))
    return merged_pred

def main(opt):
    device = select_device(opt.device)
    models = load_models(opt.weights, device)

    dataset = LoadImages(opt.source, img_size=opt.img_size)
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        predictions = ensemble_predict(models, img, opt.conf_thres, opt.iou_thres, opt.agnostic_nms)
        predictions = merge_ensemble_predictions(predictions)

        for i, det in enumerate(predictions):  # detections per image
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            print(f'Prediction for {path}: {det}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    main(opt)
