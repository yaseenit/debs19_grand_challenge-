#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:12:30 2019

@author: yassin
"""


import numpy as np
import pcl
import time
import os

#input: dataframe has x, y, z values
#output: list of written files which contain the extracted objects
def get_objects_clusters(df_scene):

  objects_list = []
    
  #start = time.time()
  scene = df_scene.drop(['laser_id', 'time'], axis=1) # drop first two colums of the dataframe, since they are not used in our proposed solution

  scene = scene[scene['x'].between(-25., 25., inclusive=True)] # z   keep only point which their x in (-25,25)
  scene = scene[scene['z'].between(-25., 25., inclusive=True)] # y   keep only point which their z in (-25,25)
  scene = scene.loc[scene['y'] > -2.5]  #    keep only point which their y in (-25,25)
  #scene.drop_duplicates(subset=None, keep='first', inplace=True) # use np.unique which might be fast
    
  points = scene.values.astype(np.float32) # convert dataframe to numpy array
  points = np.unique(points,axis=0) # drop duplicated points
  cloud = pcl.PointCloud()
  cloud.from_array(points)
    
    
  #  seg = cloud.make_segmenter()
  #  seg.set_optimize_coefficients (True)
  #  seg.set_model_type (pcl.SACMODEL_PLANE)
  #  seg.set_method_type (pcl.SAC_RANSAC)
  #  seg.set_MaxIterations (1)
  #  seg.set_distance_threshold (0.05)
  #  
  i = 0
  #output_files = []
  tree = cloud.make_kdtree()
  
  ec = cloud.make_EuclideanClusterExtraction()
  ec.set_ClusterTolerance (0.3) # 550 cm
  ec.set_MinClusterSize (55) # minimum size of a cluster 






  ec.set_MaxClusterSize (5000) # maximum size of a cluster 
  ec.set_SearchMethod (tree)
  cluster_indices = ec.Extract()
    
  #dir_path = os.getcwd()
  for j, indices in enumerate(cluster_indices):
            
      points = np.zeros((len(indices), 3), dtype=np.float32)
  
      for i, indice in enumerate(indices):

          points[i][0] = cloud[indice][0]
          points[i][1] = cloud[indice][1]
          points[i][2] = cloud[indice][2]
          
      
      objects_list.append(points)
  return objects_list
      
        
  
# testing  
if __name__== "__main__":
  import pandas as pd  
  xyz = pd.read_csv('/home/yassin/debs/debs2019_dataset2/in.csv',header=None,names = ["time","id","x", "y", "z"]) 
  xyz = xyz[0:72000] # one scene
  get_objects_clusters(xyz)
