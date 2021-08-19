import sys
import cv2
import torch
from torchvision import transforms
import numpy as np

import math
from boundingbox import BoundingBox


class Rescale(object):
    '''
    Rescale image & bounding box.
    --> Args :: Output_Size (tuple or int) : If int, square crop.
    '''

    def __init__(self, output_size):
        if isinstance(output_size, (int, tuple)):
            self.output_size = output_size
        else:
            print(' --- Enter Output Size either as an int or tuple --- ')
            sys.exit()

    def __call__(self, sample, opts):
        image, bb = sample['image'], sample['bb']
        h, w = image.shape[0], image.shape[1]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = int(self.output_size*(h/w)), int(self.output_size)
            else:
                new_h, new_w = int(self.output_size), int(self.output_size*(w/h))

        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_h, new_w), interpolation=cv2.INTER_CUBIC)
        bbox = BoundingBox(bb[0], bb[1], bb[2], bb[3])
        bbox.scale(opts['search_region'])

        return {'image' : img, 'bb' : bbox.get_bb_list()}



class NormalizeToTensor(object):
    '''
    Returns torch tensor with normalized images.
    '''

    def __call__(self, sample):
        prev_img, curr_img = sample['previmg'], sample['currimg']
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(
                                                 mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                                             ])

        prev_img = self.transform(prev_img)
        curr_img = self.transform(curr_img)

        if 'currbb' in sample:
            currbb = np.array(sample['currbb'])
            return { 'previmg' : prev_img,
                     'currimg' : curr_img,
                     'currbb'  : torch.Tensor(currbb) }
        else:
            return { 'previmg' : prev_img,
                     'currimg' : curr_img }


def bgr2rgb(image):
    '''
    check if the image is color/gray and convert to RGB accordingly.
    '''

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def shift_crop_training_sample(sample, bb_params):
    '''
    returns current image crop with shfited box (wrt current image)
    --> given an image with boudning box, this method randomly shifts the box and generated a training sample.
    '''

    output_sample, opts = {}, {}
    currimg = sample['image']
    currbb = sample['bb']
    bbox_curr_gt = BoundingBox(currbb[0], currbb[1], currbb[2], currbb[3])
    bbox_curr_shift = BoundingBox(0, 0, 0, 0)
    bbox_curr_shift = bbox_curr_gt.shift(currimg,
                                         bb_params['lambda_scale_frac'],
                                         bb_params['lambda_shift_frac'],
                                         bb_params['min_scale'],
                                         bb_params['max_scale'], True,
                                         bbox_curr_shift)

    (rand_search_region, rand_search_location,
            edge_spacing_x, edge_spacing_y) = cropPadImage(bbox_curr_shift, currimg)

    bbox_gt_recentered = BoundingBox(0, 0, 0, 0)
    bbox_gt_recentered = bbox_curr_gt.recenter(rand_search_location,
                                               edge_spacing_x,
                                               edge_spacing_y,
                                               bbox_gt_recentered)

    output_sample['image'] = rand_search_region
    output_sample['bb'] = bbox_gt_recentered.get_bb_list()


    # for visualization purpose
    opts['edge_spacing_x'] = edge_spacing_x
    opts['edge_spacing_y'] = edge_spacing_y
    opts['search_location'] = rand_search_location
    opts['search_region'] = rand_search_region

    return output_sample, opts



def crop_sample(sample):
    '''
    --> Given a sample image with its bounding box, this will return the image crop at the bouding box loaction with twice the width and height for context.
    '''

    output_sample, opts = {}, {}
    image, bb = sample['image'], sample['bb']
    orig_bbox = BoundingBox(bb[0], bb[1], bb[2], bb[3])
    (output_image, pad_image_location,
            edge_spacing_x, edge_spacing_y) = cropPadImage(orig_bbox, image)

    new_bbox = BoundingBox(0, 0, 0, 0)
    new_bbox = new_bbox.recenter(pad_image_location,
                                 edge_spacing_x,
                                 edge_spacing_y,
                                 new_bbox)

    output_sample['image'] = output_image
    output_sample['bb'] = new_bbox.get_bb_list()

    # for visualization
    opts['edge_spacing_x'] = edge_spacing_x
    opts['edge_spacing_y'] = edge_spacing_y
    opts['search_location'] = pad_image_location
    opts['search_region'] = output_image

    return output_sample, opts


def cropPadImage(bbox_tight, image):
    
    pad_image_location = computeCropPadImageLocation(bbox_tight, image)
    roi_left = min(pad_image_location.x1, (image.shape[1] - 1))
    roi_bottom = min(pad_image_location.y1, (image.shape[0] - 1))

    roi_width = min(image.shape[1], 
            max(1.0,
                math.ceil(pad_image_location.x2 - pad_image_location.x1)))

    roi_height = min(image.shape[0], 
            max(1.0,
                math.ceil(pad_image_location.y2 - pad_image_location.y1)))

    err = 1e-9              # to take care of floating point errors

    cropped_image = image[int(roi_bottom + err):int(roi_bottom + roi_height), int(roi_left + err):int(roi_left + roi_width)]

    output_width = max(math.ceil(bbox_tight.compute_output_width()), roi_width)

    output_height = max(math.ceil(bbox_tight.compute_output_height()), roi_height)

    if image.ndim > 2:
        output_image = np.zeros((int(output_height),
                                 int(output_width),
                                 image.shape[2]), dtype=image.dtype)
    else:
        output_image = np.zeros((int(output_height),
                                 int(output_width)), dtype=image.dtype)

    edge_spacing_x = min(bbox_tight.edge_spacing_x(), (image.shape[1] - 1))
    edge_spacing_y = min(bbox_tight.edge_spacing_y(), (image.shape[0] - 1))

    # rounding should be done to match the width and height

    output_image[int(edge_spacing_y):int(edge_spacing_y) + cropped_image.shape[0],
                 int(edge_spacing_x):int(edge_spacing_x) + cropped_image.shape[1]] = cropped_image

    return output_image, pad_image_location, edge_spacing_x, edge_spacing_y



def computeCropPadImageLocation(bbox_tight, image):
    # center of the bounding box

    bbox_center_x = bbox_tight.get_center_x()
    bbox_center_y = bbox_tight.get_center_y()

    image_height = image.shape[0]
    image_width = image.shape[1]

    # padded output width and height
    output_width = bbox_tight.compute_output_width()
    output_height = bbox_tight.compute_output_height()

    roi_left = max(0.0, bbox_center_x - (output_width / 2.0))
    roi_bottom = max(0.0, bbox_center_y - (output_height / 2.0))

    # padded roi width
    left_half = min(output_width / 2.0, bbox_center_x)
    right_half = min(output_width / 2.0, image_width - bbox_center_x)
    roi_width = max(1.0, left_half + right_half)

    # padded roi height
    top_half = min(output_height / 2.0, bbox_center_y)
    bottom_half = min(output_height / 2.0, image_height - bbox_center_y)
    roi_height = max(1.0, top_half + bottom_half)

    # padded image location in the original image
    objPadImageLocation = BoundingBox(roi_left,
                                      roi_bottom,
                                      roi_left + roi_width,
                                      roi_bottom + roi_height)

    return objPadImageLocation
