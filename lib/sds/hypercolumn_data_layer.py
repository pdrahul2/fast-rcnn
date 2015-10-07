import cv2
import caffe
import numpy as np
import numpy.random as npr
import argparse, pprint
import pickle
from prepare_blobs import get_blobs
import sbd
import os
from utils.cython_bbox import bbox_overlaps
import sds_config as cfg

def get_box_overlap(box_1, box_2):
  box1 = box_1.copy().astype(np.float32)
  box2 = box_2.copy().astype(np.float32)
  xmin = np.maximum(box1[:,0].reshape((-1,1)),box2[:,0].reshape((1,-1)))
  ymin = np.maximum(box1[:,1].reshape((-1,1)),box2[:,1].reshape((1,-1)))
  xmax = np.minimum(box1[:,2].reshape((-1,1)),box2[:,2].reshape((1,-1)))
  ymax = np.minimum(box1[:,3].reshape((-1,1)),box2[:,3].reshape((1,-1)))
  iw = np.maximum(xmax-xmin+1.,0.)
  ih = np.maximum(ymax-ymin+1.,0.)
  inters = iw*ih
  area1 = (box1[:,3]-box1[:,1]+1.)*(box1[:,2]-box1[:,0]+1.)  
  area2 = (box2[:,3]-box2[:,1]+1.)*(box2[:,2]-box2[:,0]+1.)  
  uni = area1.reshape((-1,1))+area2.reshape((1,-1))-inters
  iu = inters/uni
  return iu

class HypercolumnDataLayer(caffe.Layer):
  def _parse_args(self, str_arg):
    parser = argparse.ArgumentParser(description='Hypercolumn Data Layer Parameters')
    parser.add_argument('--imdb_name', default='nyud2_images_2015_train', type=str)
    parser.add_argument('--ov_thresh', default=0.7, type=float)
    parser.add_argument('--train_samples_per_img', default=5, type=int)
    parser.add_argument('--num_classes', default=19, type=int)
    args = parser.parse_args(str_arg.split())
    print('Using config:')
    pprint.pprint(args)
    return args




  def setup(self, bottom, top):
    self._params = self._parse_args(self.param_str_)
    imdb          = lib.datasets.factory.get_imdb(self._params.imdb_name)
    gt_roidb      = imdb.gt_roidb();
    roidb         = imdb.roidb;
    self._imdb    = imdb
    self._roidb   = roidb;
    self.gt_roidb = gt_roidb;
    imdb._attach_instance_segmentation;

    self.gt['classes'] = [x-1 for x in self.gt['classes']]


    #how many categories are there?
    self.num_classes = self._params.num_classes + 1
    
    #initialize
    self.data_percateg = []
    for i in range(self.num_classes):
      self.data_percateg.append({'boxids':[],'imids':[],'instids':[], 'im_end_index':[-1]})

    # compute all overlaps and pick boxes that have greater than threshold overlap
    for i in range(len(imdb.num_images)): 
      roidb_i = roidb[i]
      gt_i = gt_roidb[i]
      ov = bbox_overlaps(roidb_i['boxes'].astype(np.float), 
        gt_i['boxes'].astype(np.float))
      
      # this maintains the last index for each image for each category
      for classlabel in range(self.num_classes):
        self.data_percateg[classlabel]['im_end_index'].append(self.data_percateg[classlabel]['im_end_index'][-1])
      
      #for every gt
      for j in range(len(gt_i['gt_classes'])): 
        idx = ov[:,j] >= self._params.ov_thresh
        if not np.any(idx):
          continue
        
        #save the boxes
        classlabel = gt_i['classes'][j]
        self.data_percateg[classlabel]['boxids'].extend(np.where(idx)[0].tolist())
        self.data_percateg[classlabel]['imids'].extend([i]*np.sum(idx))
        self.data_percateg[classlabel]['instids'].extend([j]*np.sum(idx))
        self.data_percateg[classlabel]['im_end_index'][-1] += np.sum(idx)

    #convert everything to a np array because python is an ass
    for j in range(self.num_classes):
      self.data_percateg[j]['boxids']=np.array(self.data_percateg[j]['boxids'])
      self.data_percateg[j]['imids']=np.array(self.data_percateg[j]['imids'])
      self.data_percateg[j]['instids']=np.array(self.data_percateg[j]['instids'])


    #also save a dictionary of where each blob goes to
    self.blob_names = ['image','normalizedboxes','sppboxes','categids','labels', 'instance_wts']
    blobs=dict()
    self.myblobs=blobs


  def reshape(self, bottom, top):
    #sample a category
    categid = np.random.choice(self.num_classes)
    #sample an image for this category
    imid = self.data_percateg[categid]['imids'][np.random.choice(len(self.data_percateg[categid]['imids']))]

    roidb_i = self._roidb[imid]
    gt_i = self._gt_roidb[imid]
    
    img = cv2.imread(imdb.image_path_at(imid)[0])

    #get all possibilities for this category
    start = self.data_percateg[categid]['im_end_index'][imid]+1
    stop = self.data_percateg[categid]['im_end_index'][imid+1]
    
    #pick a box
    idx = np.random.choice(np.arange(start,stop+1), self._params.train_samples_per_img)
    boxid = self.data_percateg[categid]['boxids'][idx]
    boxes = roidb_i[imid][boxid,:]-1    

    instid = self.data_percateg[categid]['instids'][idx]

    #load the gt
    inst = gt_i[imid]['inst_segm']
    masks = np.zeros((idx.size, 1, inst.shape[0], inst.shape[1]))
    for k in range(idx.size):
      masks[k,0,:,:] = (inst == gt_i['instance_id'][instid[k]]).astype(np.float32)
    categids = categid*np.ones(idx.size)

    #get the blobs
    im_new, spp_boxes, normalized_boxes, categids, masksblob, instance_wts = get_blobs(img, boxes.astype(np.float32), categids, masks)

    #save blobs in private dict
    self.myblobs['image']=im_new.astype(np.float32)
    self.myblobs['normalizedboxes']=normalized_boxes.astype(np.float32)
    self.myblobs['sppboxes']=spp_boxes.astype(np.float32)
    self.myblobs['categids']=categids.astype(np.float32)
    self.myblobs['labels']=masksblob.astype(np.float32)
    self.myblobs['instance_wts']=instance_wts.astype(np.float32)

    #and reshape
    for i in range(len(top)):
      top[i].reshape(*(self.myblobs[self.blob_names[i]].shape))

  def forward(self, bottom, top):
    for i in range(len(top)):
      top[i].data[...] = self.myblobs[self.blob_names[i]]

  def backward(self, top, propagate_down, bottom):
    pass

    
 

