# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 07:45:13 2019

@author: Murali

OpenCV practice
"""

#from scipy import ndimage, misc
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd


def main():
    path = "D:/hackathon_files/2019Dec/Open-CV/train_HNzkrPW/image_data/"
   
    #img = cv2.imread('\train_HNzkrPW\image_data\')
    #plt.imshow(img)
    
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
        
        #image_to_read = ndimage.imread(input_path)
        
        #print('input_path = '+input_path)

        #use_cv2(image_to_read)
        
        #print("Faces found : ",image_path,"|",new_face_detect(input_path))
        
        #print(image_path , len(faces))
        
        list_dict.append({'filename':image_path, 'heads':len(faces)})
        
        #data = dict(image_path+","+len(faces))
        
    df = df.append(list_dict)
        
    df.to_csv('sample_result.csv')
        
    print(df)
    
        
def use_cv2():
    temp_image_path = "D:/hackathon_files/2019Dec/Open-CV/train_HNzkrPW/image_data/18206.jpg"
    image_to_read = plt.imread(temp_image_path,0)
    plt.imshow(image_to_read)
    
    """
    # Convert the image into RGB    
    img_rgb = cv2.cvtColor(image_to_read, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)

    # Convert the image into gray scale
    img_gray = cv2.cvtColor(image_to_read, cv2.COLOR_BGR2GRAY)
    plt.imshow(img_gray, cmap = 'gray')


    # Plot the three channels of the image
    fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 20))
    for i in range(0, 3):
        ax = axs[i]
        ax.imshow(img_rgb[:, :, i], cmap = 'gray')
    plt.show()


    # Transform the image into HSV and HLS models
    img_hsv = cv2.cvtColor(image_to_read, cv2.COLOR_BGR2HSV)
    img_hls = cv2.cvtColor(image_to_read, cv2.COLOR_BGR2HLS)
    # Plot the converted images
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 20))
    ax1.imshow(img_hsv)
    ax2.imshow(img_hls)
    plt.show()
    

    #img = cv2.imread('messi5.jpg',0)
    edges = cv2.Canny(image_to_read,100,200)

    plt.subplot(121),plt.imshow(image_to_read,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()

    """
   
    
  
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    #eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') 
    
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(image_to_read, cv2.COLOR_BGR2GRAY) 
    
    # Detects faces of different sizes in the input image 
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    faces = face_cascade.detectMultiScale(gray, 5, 5) 
  
    for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
        cv2.rectangle(image_to_read,(x,y),(x+w,y+h),(255,255,0),2)  
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = image_to_read[y:y+h, x:x+w] 
  
        # Detects eyes of different sizes in the input image 
        eyes = eye_cascade.detectMultiScale(roi_gray)  
  
        #To draw a rectangle in eyes 
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 
  
    # Display an image in a window 
    cv2.imshow('image_to_read',image_to_read) 
    
def face_detect( img):
        """ Detect the face location of the image img, using Haar cascaded face detector of OpenCV.
        
        return : x,y w, h of the bouning box.
        """
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(img, 1.3, 50)
        x = -1
        y = -1
        w = -1
        h = -1
        print('faces : ', faces)
        if len(faces) == 1: # we take only when we have 1 face, else, we return nothing.
            x,y,w,h = faces[0]
        else:
            for (x_,y_,w_,h_) in faces:
                x = x_
                y = y_
                w = w_
                h = h_
                print (x,y,w,h)
##                break # we take only the first face,
            print ("More than one face!!!!!!!!!")
            
        
        return x,y,w,h 

def new_face_detect(img_path):
    # Load opencv library
    import cv2
    
    # load face_cascade from haarcascade face xml file
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
    
    # take image as input
    img = cv2.imread(img_path)
    
    # convert image into gray scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # detect faces
    #faces = face_cascade.detectMultiScale(
    #        gray_img, 
    #        scaleFactor=1.1, 
    #        minNeighbors=1
    #    )
    
    faces = face_cascade.detectMultiScale(
            gray_img,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(5, 5)#,
            #flags = cv2.CV_HAAR_SCALE_IMAGE
            )
    

    #print ("Found {0} faces!".format(len(faces)))
    return(len(faces))

    # create list of faces and make rectangle around faces
    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    # resize image to fit
    resized_img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    
    # show image
    cv2.imshow("gray", resized_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    temp_image_path = "D:/hackathon_files/2019Dec/Open-CV/train_HNzkrPW/image_data/10003.jpg"
    image_to_read = plt.imread(temp_image_path,0)
    #print (new_face_detect( temp_image_path))
    main()
    