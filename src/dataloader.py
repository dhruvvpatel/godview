import cv2
import numpy as np
from torch.utils.data import Dataset

import os
from helper import *



class ALOVDataset(Dataset):
    ''' handler for the ALOV tracking dataset '''

    def __init__(self, root_dir, target_dir, transform=None, input_size=227):
        super(ALOVDataset, self).__init__()

        self.exclude = ['01-Light_video00016',
                        '01-Light_video00022',
                        '01-Light_video00023',
                        '02-SurfaceCover_video00012',
                        '03-Specularity_video00003',
                        '03-Specularity_video00012',
                        '10-LowContrast_video00013',
                        '.DS_Store']

        self.root_dir = root_dir
        self.target_dir = target_dir
        self.input_size = input_size
        self.transform = transform
        self.x, self.y = self._parse_data(root_dir, target_dir)
        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample, _  = self.get_sample(idx)
        if (self.transform):
            sample = self.transform(sample)
        return sample

    def _parse_data(self, root_dir, target_dir):
        ''' 
        Note : Parse ALOV dataset and return tuples of (template, search region)
               from annotated frames
        '''

        x = []
        y = []

        t_dir = os.listdir(target_dir)
        num_annot = 0
        print('Parsing ALOV dataset ...')

        for _file in t_dir:
            if not _file.startswith('.'):
                vids = os.listdir(root_dir + _file)
                for vid in vids:
                    if vid in self.exclude:
                        continue

                    vid_src = self.root_dir + _file + "/" + vid
                    vid_ann = self.target_dir + _file + "/" + vid + ".ann"
                    frames = os.listdir(vid_src)
                    frames.sort()
                    # frames as the entire path 
                    frames = [vid_src + "/" + frame for frame in frames]
                    frames = np.array(frames)

                    f = open(vid_ann, "r")
                    annotations = f.readlines()
                    f.close()
                    frame_idxs = [int(ann.split(' ')[0])-1 for ann in annotations]
                    num_annot += len(annotations)

                    for i in range(len(frame_idxs)-1):
                        curr_idx = frame_idxs[i]
                        next_idx = frame_idxs[i+1]

                        x.append([frames[curr_idx], frames[next_idx]])
                        y.append([annotations[i], annotations[i+1]])


        x, y = np.array(x), np.array(y)
        self.len = len(y)
        print('ALOV dataset parsing done.')
        print(f'Total annotations in ALOV dataset : {num_annot}')

        return x, y


    def get_sample(self, idx):
        '''
        Get sample without doing any transformation for visualization.

        Sample consists of resized previous and current frame with target which
        is passed to the network. Bounding box values are normalized between 0 and 1
        with respect to the target frame and then scaled by factor of 10.
        '''
        opts_curr = {}
        curr_sample = {}
        curr_img = self.get_orig_sample(idx, 1)['image']
        currbb = self.get_orig_sample(idx, 1)['bb']
        prevbb = self.get_orig_sample(idx, 0)['bb']

        bbox_curr_shift = BoundingBox(prevbb[0],
                                      prevbb[1],
                                      prevbb[2],
                                      prevbb[3])

        rand_search_reg, rand_search_loc, edge_spacing_x, edge_spacing_y = cropPadImage(bbox_curr_shift, curr_img)

        bbox_curr_gt = BoundingBox(currbb[0],
                                   currbb[1],
                                   currbb[2],
                                   currbb[3])
        bbox_gt_recenter = BoundingBox(0, 0, 0, 0)
        bbox_gt_recenter = bbox_curr_gt.recenter(rand_search_loc, edge_spacing_x, edge_spacing_y, bbox_gt_recenter)

        curr_sample['image'] = rand_search_reg
        curr_sample['bb'] = bbox_gt_recenter.get_bb_list()

        # options for visualization
        opts_curr['edge_spacing_x'] = edge_spacing_x
        opts_curr['edge_spacing_y'] = edge_spacing_y
        opts_curr['search_location'] = rand_search_loc
        opts_curr['search_region'] = rand_search_reg

        # build prev sample
        prev_sample = self.get_orig_sample(idx, 0)
        preV_sample, opts_prev = crop_sample(prev_sample)

        # scale
        scale = Rescale((self.input_size, self.input_size))
        scaled_curr_obj = scale(curr_sample, opts_curr)
        scaled_prev_obj = scale(prev_sample, opts_prev)

        training_sample = {'previmg' : scaled_prev_obj['image'],
                           'currimg' : scaled_curr_obj['image'],
                           'currbb'  : scaled_curr_obj['bb']}

        return training_sample, opts_curr


    def get_orig_sample(self, idx, i=1):
        '''
        Returns original image with bounding box at a specific index.
        Range of valid index : [0, self.len - 1]

        i : 0 for prev_img and 1 for curr_img
        '''

        curr = cv2.imread(self.x[idx][i])
        curr = bgr2rgb(curr)
        currbb = self.get_bb(self.y[idx][i])
        sample = {'image' : curr, 'bb' : currbb}

        return sample

    def get_bb(self, ann):
        '''
        ALOV annotation parser and returns bounding box in the format:
        [left, upper, width, height]
        '''

        ann = ann.strip().split(' ')
        left = min(float(ann[1]), float(ann[3]), float(ann[5]), float(ann[7]))
        top  = min(float(ann[2]), float(ann[4]), float(ann[6]), float(ann[8]))
        right = max(float(ann[1]), float(ann[3]), float(ann[5]), float(ann[7]))
        bottom = max(float(ann[2]), float(ann[4]), float(ann[6]), float(ann[8]))

        return [left, top, right, bottom]

    
    def show(self, idx, is_curr=1):
        '''
        For visualizing an image at a given index with it's corresponding gt_bb
        
        Arguments :
            idx = index
            is_curr = for prev frame : 0 | for current frame : 1
        '''

        sample = self.get_orig_sample(idx, is_curr)
        image = sample['image']
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bb = sample['bb']
        bb = [int(round(val)) for val in bb]
        image = cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
        cv2.imshow('alov dataset sample : ' + str(idx), image)
        cv2.waitKey(0)


    def show_sample(self, idx):
        '''
        For visualizing the sample that is passed to GOTURN.
        Shows previous frame and current frame with bounding box.
        '''

        x, _ = self.get_sample(idx)
        prev_img = x['previmg']
        curr_img = x['currimg']
        bb = x['currbb']
        bbox = BoundingBox(bb[0], bb[1], bb[2], bb[3])
        bbox.unscale(curr_image)
        bb = bbox.get_bb_list()
        bb = [int(round(val)) for val in bb]

        prev_img = cv2.cvtColor(prev_img, cv2.COLOR_RGB2BGR)
        curr_img = cv2.cvtColor(curr_img, cv2.COLOR_RGB2BGR)
        curr_img = cv2.rectangle(curr_img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)

        concat_img = np.hstack((prev_img, curr_img))
        cv2.imshow('ALOV dataset sample : ' + str(idx), concat_image)
        cv2.waitKey(0)




