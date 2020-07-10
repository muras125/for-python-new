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
    
    sample_submission_fChOj3V_df = pd.read_csv("sample_submission_fChOj3V.csv")
    
    print (path)
    for Index, row in sample_submission_fChOj3V_df.iterrows():
        
        #print('before: '+getattr(row, "Name") + ": "+str(getattr(row, "HeadCount")))
        
        input_path = os.path.join(path, getattr(row, "Name"))
        
        # take image as input
        img = cv2.imread(input_path)
        
        # convert image into gray scale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # detect faces
        faces = face_cascade.detectMultiScale(
                gray_img,
                scaleFactor=1.2,
                minNeighbors=1,
                minSize=(5, 5)#,
                #flags = cv2.CV_HAAR_SCALE_IMAGE
                )
        
        setattr(row, "HeadCount", str(len(faces))) 
        #sample_submission_fChOj3V_df.set_value(row.Index, 'HeadCount', len(faces))
        #sample_submission_fChOj3V_df.at[Index, 'HeadCount'] = len(faces)
        #print(str(len(faces)))
        print('after: '+getattr(row, "Name") + ": "+str(getattr(row, "HeadCount")))
        
    #print(sample_submission_fChOj3V_df)
    
    sample_submission_fChOj3V_df.to_csv('mm_sample_result.csv', index = False)
    
    #df = df.append(list_dict)
        
    #df.to_csv('sample_result.csv', index = False)
        
    #print(df)
    
    
if __name__ == '__main__':
    
    main()
    