import requests
import json
import pandas as pd
import os
import time
import pcl
import subprocess
import pprint
import pickle
import numpy as np
import multiobjects_scene_to_pointclouds as extracter
from scipy.spatial import distance
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from my_pcl import PyPCL
cppPCL = PyPCL();






with open('classifier.pkl', 'rb') as f:
 classifier = pickle.load(f)
        
#scaler = joblib.load("scaler.save") 



def host_url(host, path):
    print("http://" + host + path)
    return "http://" + host + path


def get_scene(host):
    return requests.get(host_url(host, '/scene/'))


def post_answer(host, payload):
    headers = {'Content-type': 'application/json'}
    response = requests.post(host_url(host, '/scene/'), json = payload, headers=headers)

    print('Response status is: ', response.status_code)
    if (response.status_code == 201):
        return {'status': 'success', 'message': 'updated'}
    if (response.status_code == 404):
        return {'message': 'Something went wrong. No scene exist. Check if the path is correct'}


if __name__ == "__main__":
    print('ENV is ', os.getenv('BENCHMARK_SYSTEM_URL'))

    host = os.getenv('BENCHMARK_SYSTEM_URL')
    if host is None or '':
        print('Error reading Server address!')

    print('Getting scenes for predictions...')

    # Here is an automated script for getting all scenes
    # and submitting prediction for each of them
    # you may change to fit your needs
    scene_counter = 0
    while(True):

         #Making GET request
         #Each request will fetch new scene
        response = get_scene(host)

        if response.status_code == 404:
            print(response.json())
            #time.sleep(10000)
            break
        scene_counter+=1
        data = response.json()
        # example of reconstruction json payload from GET request into DataFrame
        reconstructed_scene = pd.read_json(data['scene'], orient='records')
        
        #######################################################
        reconstructed_scene.rename(columns={'X':'x','Y':'y','Z':'z'}, inplace=True)
       ####################
        final_results = {}
        #reconstructed_scene.columns =  ["time","laser_id","x", "y", "z"]
      
        for pc_object in extracter.get_objects_clusters(reconstructed_scene):
          features = cppPCL.compute(pc_object)
          resultant = classifier.predict(np.array(features).reshape(1, -1))[0].strip("'") 
          if resultant in final_results:
            final_results[resultant] +=1
          else:
            final_results[resultant] = 1
        
        #pprint.pprint(final_results)      
            
    


        post_answer(host, final_results)
    print('Submission for all scenes done successfully!')
