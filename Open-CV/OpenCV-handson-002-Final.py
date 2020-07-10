# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:09:18 2020

@author: Murali
"""

import cv2
import os
import pandas as pd


def main():
    path = "D:/hackathon_files/2019Dec/Open-CV/train_HNzkrPW/image_data/"
  
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
    
    df = pd.DataFrame([])
    list_dict = []
    
    for image_path in os.listdir(path):
        input_path = os.path.join(path, image_path )

        # take image as input
        img = cv2.imread(input_path)
        
        # convert image into gray scale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # detect faces
        faces = face_cascade.detectMultiScale(
                gray_img,
                scaleFactor=1.1,
                minNeighbors=1,
                minSize=(5, 5)#,
                #flags = cv2.CV_HAAR_SCALE_IMAGE
                )
        
        list_dict.append({'Name':image_path, 'HeadCount':len(faces)})
        
    df = df.append(list_dict)
        
    df.to_csv('sample_result.csv', index = False)
        
    #print(df)
    
    
if __name__ == '__main__':
    
    main()
    