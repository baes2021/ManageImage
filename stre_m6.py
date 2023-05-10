import streamlit as st
import base64
from zipfile import ZipFile
from tensorflow.keras.preprocessing.image import load_img
from os import listdir
import os
import glob
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import shutil
import os.path
import random
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import fnmatch
import numpy as np
import matplotlib.image as mpimg
import h5py
import keras
from keras.layers import Flatten, Dense, Input,concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout
from keras.models import Model
from keras.models import Sequential
import tensorflow as tf
import time
import keras.utils as image
from keras.applications.imagenet_utils import preprocess_input
import argparse
import imutils
##### side bar 
with st.sidebar:
#### header #####
    st.sidebar.title('Dataset')
    st.sidebar.subheader('Input data')

    st.write("Give either the path of your dataset or upload a zip file. The accepted format of \
             photos are jpg, jpeg \
             and png.")

##### dropdown menue ####
    option_input = st.selectbox(
     '',
    ('<select>','upload a zip file', 'give the path'), format_func=lambda x: 'Select an option' if x == '<select>' else x)
##### data extraction ####
    if option_input== 'give the path':
        st.title('Giving the path of dataset')
        # define parent_dir (the directory here input data)
        parent_dir  = st.text_input("give the path to your dataset:",'../test/')
        isExist = os.path.exists(parent_dir)
        if not isExist:
            new_title = '<p style="font-family:sans-serif; color:Red; font-size: 22px;">Warning: Input cannot be found. Please check your path.</p>'
            st.markdown(new_title, unsafe_allow_html=True)
        else:
            files=listdir(parent_dir)
            if files:
                pass
            else:
                new_title = '<p style="font-family:sans-serif; color:Red; font-size: 22px;">Warning: There is no dataset in the given path. Please check your path.</p>'
                st.markdown(new_title, unsafe_allow_html=True)
    elif option_input== 'upload a zip file':
        st.title('Uploading a zip file')
        input=st.sidebar.file_uploader('Upload your file:',type='zip')
        ## zip file should be in the current diectory
        input_dir='./'
        if (input is not None): 
            # get the name of zip file
            input_dir=input.name.split('.')[0]
            # check if there is a directory with the same name in the current directory
            isExist=os.path.exists('./'+input_dir)
            # check its size 
            isEmpty=os.stat(input.name).st_size
            if isEmpty ==0:
                # if file size is zero, meaning there is no data in zip file
                new_title = '<p style="font-family:sans-serif; color:Red; font-size: 22px;">Warning: Input is empty.</p>'
                st.markdown(new_title, unsafe_allow_html=True)
            if isExist and isEmpty !=0:
                # if a directory with the same name of zip file exists in the current directory, it will be removed
                shutil.rmtree(input_dir)
                # extract the zip file
            with ZipFile(input, 'r') as f:
                f.extractall('./')           
                ### removing _MACOSX (a file which is created during extracting) ####
                if '__MACOSX' in listdir('./'):
                    shutil.rmtree('./__MACOSX', ignore_errors=False, onerror=None)
            
        parent_dir = './'+input_dir+'/'
        #ext=listdir(parent_dir)[0].split('.')[-1]





    
# define the background of sidebar
def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
side_bg = './image_half_trans.png'
sidebar_bg(side_bg)

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# background image of the main page
def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover;

         }}
         </style>
         """, unsafe_allow_html=True)
set_bg_hack('./back.png')

# define three containers
header=st.container()
visualisation=st.container()
input_query=st.container()


with header:
    st.title('ManageImage')
    st.markdown('<div style="text-align: justify; font-size:120%;">Are you on holiday, taking many photos by your phone to capture memories and your phone is exploding of dulicate or similar photos? Does\
              your phone has limited memory space for storage and you want to manage your photos?\
             MangeImage helps you to organize your photos by removing duplicate photos and seprating\
             similar photos or blur photos to a new folder. In ManageImage you can save photos in a seprate folder based on the level of similarity /blur which is chosen by you. </div>', unsafe_allow_html=True)

with visualisation:
    st.header('Visualisation')
    option_vis = st.selectbox(
     'Do you want to have a look at some photos (here photos are selected randomly):',
    ('<select>','no','yes'), format_func=lambda x: 'Select an option' if x == '<select>' else x)
    if option_vis == 'yes' and (option_input == '<select>' or input is None):
        new_title = '<p style="font-family:sans-serif; color:Red; font-size: 22px;">Warning: Input your data first!</p>'
        st.markdown(new_title, unsafe_allow_html=True)


    if option_vis =='yes' and option_input != '<select>' and input is not None:

        @st.experimental_memo(suppress_st_warning=True)
        def visualisation_photos():
            img_paths=listdir(parent_dir)
            # get all images
            img_paths = [os.path.join(parent_dir, img_paths[i]) for i in range(len(img_paths)) if (img_paths[i].endswith('jpg') or img_paths[i].endswith('jpeg') or img_paths[i].endswith('png'))]
            random_img = random.sample(img_paths, 9)
            # display random images
            n=1
            fig1=plt.figure(figsize=(20,20))
            for i in range(9):
                plt.subplot(3, 3, n)
                plt.axis('off')
                im=random_img[i]
                img = cv2.imread(im) 
                plt.imshow(img)
                plt.tight_layout()
                n+=1
            st.pyplot(fig1)
            return random_img

        random_img=visualisation_photos()


with input_query:
    st.header('Manage your photos')
    option_query = st.selectbox(
     'Do you want to delete the duplicate photos or just put similar photos to a seprate folder or perhapse do the both of them, or detect blur images?',
    ('<select>','Delete duplicate photos', 'Seprate similar photos', 'Delete duplicate photos and seprate similar photos', 'Detect blur photos',
    'Delete duplicate photos and seprate similar and blur photos'), format_func=lambda x: 'Select an option' if x == '<select>' else x)
    if option_query != '<select>' and (option_input == '<select>' or input is None):
        new_title = '<p style="font-family:sans-serif; color:Red; font-size: 22px;">Warning: Input your data first!</p>'
        st.markdown(new_title, unsafe_allow_html=True)

    option_out = st.selectbox(
     'How do you like to have your output?',
    ('<select>','save in a specific path','download as a zip file'), format_func=lambda x: 'Select an option' if x == '<select>' else x)

    if option_out != '<select>' and option_out=='save in a specific path':
        output_dir  = st.text_input("give the path you want to save the output",'./output_image/')
        isExist = os.path.exists('./'+output_dir)



## Use vgg16 last lyer to get the features of images
    vgg16 = tf.keras.applications.VGG16(weights='imagenet', include_top=True, pooling='max', input_shape=(224, 224, 3))
    basemodel = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)

    # create a new directory for output (parent_out_dir) and move data to it
    parent_out_dir='./output_image/'
    isExist = os.path.exists(parent_out_dir)
    # if parent_dout_dir exists in the current directory, first remove it and then copy the input data there
    if isExist and option_input != '<select>':
        shutil.rmtree(parent_out_dir)
        shutil.copytree(parent_dir, parent_out_dir)
    # if parent_dout_dir does not exist in the current directory, then copy the input data there
    elif not isExist and option_input != '<select>':   
        shutil.copytree(parent_dir, parent_out_dir)

# if delete duplicate photis is selected    
    if option_query != '<select>' and option_input != '<select>' and input is not None  and (option_query== 'Delete duplicate photos'):
        # function to get features from vgg16
        def extract_features(img_path):
            img = load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = basemodel.predict(x)
            return features.flatten()

        # List of images in parent_out_dir
        directoryh=parent_out_dir
        img_paths = listdir(parent_out_dir)
        # join the path and image names
        img_paths = [os.path.join(directoryh, img_paths[i]) for i in range(len(img_paths)) if (img_paths[i].endswith('jpg') or img_paths[i].endswith('jpeg') or img_paths[i].endswith('png'))]
        # Extract features from each image
        features_list = [extract_features(path) for path in img_paths]
        
        @st.experimental_memo
        def photo_delete():
            num_del=0
            name_list=list()
            # Compute pairwise differences between feature vectors
            for i in range(len(features_list)):
                for j in range(len(features_list)):
                    if  i !=j:
                        A=np.stack((features_list[i], features_list[j]))
                        A_sparse = sparse.csr_matrix(A)
                        similarities_sparse = cosine_similarity(A_sparse,dense_output=False)
                        sim= np.mean(similarities_sparse)
                        # remove images which have mean of similarities_sparse more than 0.98 
                        if sim >=0.985:
                            img1 = cv2.imread(img_paths[i])
                            img2 = cv2.imread(img_paths[j])
                            if img_paths[i] in name_list or img_paths[j] in name_list:
                                pass
                            else:
                                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                                fm1= cv2.Laplacian(gray1, cv2.CV_64F).var()
                                fm2= cv2.Laplacian(gray2, cv2.CV_64F).var()
                                if fm1 < fm2:
                                    pass
                                else:                           
                                    if img_paths[i] not in name_list:
                                        name_list.append(img_paths[i])
                                    if img_paths[j] not in name_list:
                                        name_list.append(img_paths[j])
                                        os.remove(img_paths[j])
                                        num_del+=1
            if num_del !=0:
                st.write('List and visualisation of the duplicate images (maximum of 25 photos can be displayed):')
                st.write(name_list)
            else:
                st.write('There are no duplicate images')                
            return num_del, name_list
        

        num_del, name_list=photo_delete()
        n=1
        # plot duplicate images (only 25 images will be displayed)
        # get the path of images in name_list (name_list has the name of images which have been removed)
        if len(name_list) !=0:
            image_dir=[]
            for i in name_list:
                filename = os.path.basename(i)
                # since the duplicte images has already removed from the parent_out_dir, to show the duplicate images wwe have to switch to 
                # parent directory
                image_dir.append(os.path.join(parent_dir, filename))
            images= [cv2.imread(image_path) for image_path in image_dir]
            # For visualisation, only 25 images are shown. So if name_list has more than 25 images, it shows only the first 25 images of name_lst
            if len(images) > 25:
                images=images[0:25]
            # plot the duplicate images
            fig1=plt.figure(figsize=(20,20))
            for i in images:
                plt.subplot(5, 5, n)
                plt.imshow(i) 
                plt.tight_layout()
                plt.axis('off')
                n += 1
            st.pyplot(fig1)




 # if seprate similar images is selected       
    if option_query != '<select>' and option_input!= '<select>' and input is not None and (option_query == 'Seprate similar photos'):
        similarity_value=st.slider('Here you can specify the level of similarity between photos. Zero means there are not any similarities between photos and 100 means photos are hundred percent similar (recommended values > 85)', min_value=0, max_value=100)
        # 5 second time to select the similarity level from slider bar
        time.sleep(5) 
        similarity_value=float(similarity_value/100)
        #Zero similarity does not make sense
        if similarity_value == 0:
            new_title = '<p style="font-family:sans-serif; color:Red; font-size: 22px;">Warning: Zero does not make sense. Select a value higher than 0.</p>'
            st.markdown(new_title, unsafe_allow_html=True)

    # make a directora to put similar images
        directory = "Similar_images/"  
    # Path of this new directory in parent_out_dir
        path = os.path.join(parent_out_dir, directory) 
        isExist = os.path.exists(path)
    # if directory does not exist create it
        if not isExist:
            os.mkdir(path)
        # If directory exists get features images
        if similarity_value !=0:
            def extract_features(img_path):
                img = load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                features = basemodel.predict(x)
                return features.flatten()

        # List of image names and paths
            directoryh=parent_out_dir
            img_paths = listdir(parent_out_dir)
            img_paths = [os.path.join(directoryh, img_paths[i]) for i in range(len(img_paths)) if (img_paths[i].endswith('jpg') or img_paths[i].endswith('jpeg') or img_paths[i].endswith('png')) ]
        # Features of images
            features_list = [extract_features(path) for path in img_paths]
            # seprate similar images
            def similar_photos():
                name_list=list()
                num_sim=0
                #Extract features from each image
                for i in range(len(features_list)):
                    for j in range(len(features_list)):
                        if  i !=j:
                            A=np.stack((features_list[i], features_list[j]))
                            A_sparse = sparse.csr_matrix(A)
                            similarities_sparse = cosine_similarity(A_sparse,dense_output=False)
                            sim= np.mean(similarities_sparse)

                            # seprate images with mean of similaries_sparse between similarity_value (given by user) and 0.9999
                            if sim <=0.99999 and sim >=similarity_value:
                                if img_paths[i] in name_list or img_paths[j] in name_list:
                                    pass
                                else:
                                    img1 = cv2.imread(img_paths[i])
                                    img2 = cv2.imread(img_paths[j])
                                    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                                    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                                    fm1= cv2.Laplacian(gray1, cv2.CV_64F).var()
                                    fm2= cv2.Laplacian(gray2, cv2.CV_64F).var()
                                    if fm1 < fm2:
                                        pass
                                    else:
                                        src_path = img_paths[j]
                                        filename = os.path.basename(img_paths[j])
                                        simi= parent_out_dir +directory
                                        dst_path = os.path.join(simi, filename)
                                        if img_paths[i] not in name_list:
                                            name_list.append(img_paths[i])
                                        if img_paths[j] not in name_list:
                                            name_list.append(img_paths[j])
                                            shutil.move(src_path, dst_path, copy_function='copy2')

                                            num_sim +=1

                return num_sim, name_list
        
            num_sim, name_list=similar_photos()
            if len(name_list) !=0:
                st.write('List and visualisation of the similar images (maximum of 25 photos can be displayed):')
                st.write(name_list)
            else:
                st.write('There are no similar images')               
            n=1

            # Plot similar images (only 25 images will be displayed)
            # get the path of images in name_list (name_list has the name of images which have been removed)
            # since one of the similar files has already removed from the parent_out_dir, to show the duplicate image wwe have to switch to 
            # parent directory
            if len(name_list) !=0:
                image_dir=[]
                for i in name_list:
                    filename = os.path.basename(i)
                    image_dir.append(os.path.join(parent_dir, filename))
                images= [cv2.imread(image_path) for image_path in image_dir]
                # Since only 25 images are shown here, if images are more than 25, only the first 25 similar images are shown
                if len(images) > 25:
                    images=images[0:25]
                # plot similar images
                fig1=plt.figure(figsize=(20,20))
                for i in images:
                    plt.subplot(5, 5, n)
                    plt.imshow(i) 
                    plt.tight_layout()
                    plt.axis('off')
                    n += 1
                st.pyplot(fig1)

 # if Detect blur images is selected       
    if option_query != '<select>' and option_input!= '<select>' and input is not None and (option_query == 'Detect blur photos'):
        Blur_value=st.slider('Here you can specify the threshold for blur photos (recommended value= 100) )', min_value=90, max_value=150)
        # 5 second time to select the contrast level from slider bar
        time.sleep(5) 
        Blur_value=float(Blur_value)


    # make a directora to put low contrast images
        directory = "Blur_images/"  
    # Path of this new directory in parent_out_dir
        path = os.path.join(parent_out_dir, directory) 
        isExist = os.path.exists(path)
    # if directory does not exist create it
        if not isExist:
            os.mkdir(path)
        # If directory exists get features images

        def extract_features(img_path):
            img = load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = basemodel.predict(x)
            return features.flatten()

        # List of image names and paths
        directoryh=parent_out_dir
        img_paths = listdir(parent_out_dir)
        img_paths = [os.path.join(directoryh, img_paths[i]) for i in range(len(img_paths)) if (img_paths[i].endswith('jpg') or img_paths[i].endswith('jpeg') or img_paths[i].endswith('png')) ]

        # seprate similar images
        def blur_photos():
            name_list=list()
            num_blur=0
            for i in range(len(img_paths)):
                img = cv2.imread(img_paths[i])
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                fm= cv2.Laplacian(gray, cv2.CV_64F).var()
                if fm < Blur_value:
                    src_path = img_paths[i]
                    filename = os.path.basename(img_paths[i])
                    cont= parent_out_dir +directory
                    dst_path = os.path.join(cont, filename)
                    if img_paths[i] not in name_list:
                        name_list.append(img_paths[i])
                    num_blur +=1
                    shutil.move(src_path, dst_path, copy_function='copy2')

            return num_blur, name_list
        
        num_blur, name_list=blur_photos()
        if len(name_list) !=0:
            st.write('List and visualisation of the blur images (maximum of 25 photos can be displayed):')
            st.write(name_list)
        else:
            st.write('There are no blur images')               
        n=1

            # Plot low contrast images (only 25 images will be displayed)
            # get the path of images in name_list (name_list has the name of images which have been removed)
            # since one of the similar files has already removed from the parent_out_dir, to show the duplicate image wwe have to switch to 
            # parent directory
        if len(name_list) !=0:
            image_dir=[]
            for i in name_list:
                filename = os.path.basename(i)
                image_dir.append(os.path.join(parent_dir, filename))
            images= [cv2.imread(image_path) for image_path in image_dir]
                # Since only 25 images are shown here, if images are more than 25, only the first 25 similar images are shown
            if len(images) > 25:
                images=images[0:25]
                # plot similar images
            fig1=plt.figure(figsize=(20,20))
            for i in images:
                plt.subplot(5, 5, n)
                plt.imshow(i) 
                plt.tight_layout()
                plt.axis('off')
                n += 1
            st.pyplot(fig1)

# If delete duplicate and seprate similar images is selected
    if option_query != '<select>' and option_input!= '<select>' and  input is not None  and (option_query == 'Delete duplicate photos and seprate similar photos'):
        similarity_value=st.slider('Here you can specify the level of similarity between photos. Zero means there are not any similarities between photos and 100 means photos are hundred percent similar (recommended values > 85)', min_value=0, max_value=100)
        time.sleep(5) 
        # give the level of similarity
        similarity_value=float(similarity_value/100)

        # make similarity directory to save similar images
        directory = "Similar_images/"  
    # define the path of similarity directory
        path = os.path.join(parent_out_dir, directory) 
        isExist = os.path.exists(path)
    # Create the directory
        if not isExist:
            os.mkdir(path)
        # give warning regarding zero similarity level
        if similarity_value ==0:
            new_title = '<p style="font-family:sans-serif; color:Red; font-size: 22px;">Zero does not make sense.  Select a value higher than 0.</p>'
            st.markdown(new_title, unsafe_allow_html=True)
        # define the function of extracing image features
        if similarity_value !=0:
            def extract_features(img_path):
                img = load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                features = basemodel.predict(x)
                return features.flatten()

        # create list of images with their paths and list of image features
            directoryh=parent_out_dir
            img_paths = listdir(parent_out_dir)

            img_paths = [os.path.join(directoryh, img_paths[i]) for i in range(len(img_paths)) if (img_paths[i].endswith('jpg') or img_paths[i].endswith('jpeg') or img_paths[i].endswith('png')) ]
            features_list = [extract_features(path) for path in img_paths]
        # delecte duplicate images
        #@st.experimental_memo
        if similarity_value !=0:
            def photo_delete():
                num_del=0
                name_list=list()
                # since we have another loop for seprating similar images, here we have to create new feature_list and img_paths (since in the current loop, we cannot drop any element of these lists- loop is based on length of feature_list )
                features_list_back=features_list
                img_paths_back=img_paths
                for i in range(len(features_list)):
                    for j in range(i+1, len(features_list)):
                        if  i !=j:
                            # calculte similarity of two images
                            A=np.stack((features_list[i], features_list[j]))
                            A_sparse = sparse.csr_matrix(A)
                            similarities_sparse = cosine_similarity(A_sparse,dense_output=False)
                            sim= np.mean(similarities_sparse)
                            # delete images with higher similarities of 0.985
                            if sim >=0.985:
                                img1 = cv2.imread(img_paths[i])
                                img2 = cv2.imread(img_paths[j])
                                if img_paths[i] in name_list or img_paths[j] in name_list:
                                    pass
                                else:
                                    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                                    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                                    fm1= cv2.Laplacian(gray1, cv2.CV_64F).var()
                                    fm2= cv2.Laplacian(gray2, cv2.CV_64F).var()
                                    if fm1 < fm2:
                                        pass
                                    else:                           
                                        if img_paths[i] not in name_list:
                                            name_list.append(img_paths[i])
                                        if img_paths[j] not in name_list:
                                            name_list.append(img_paths[j])
                                            os.remove(img_paths[j])
                                            features_list_back=features_list_back[:j]+ features_list_back[j+1:]
                                            img_paths_back=img_paths_back[:j]+ img_paths_back[j+1:]
                                            num_del+=1
                # Write list of duplicate images
                if len(name_list) != 0: 
                    st.write('List and visualisation of the duplicate images (maximum of 25 photos can be displayed):')
                    st.write(name_list)
                else:
                     st.write('There are no duplicate images')                   
                return num_del, name_list, features_list_back, img_paths_back
            
    
            num_del, name_list, features_list_back, img_paths_back = photo_delete()
            n=1
        # plot duplicate images (only 25 images will be displayed)
        # get the path of images in name_list (name_list has the name of images which have been removed)
            if len(name_list) !=0:
                image_dir=[]
                for i in name_list:
                    filename = os.path.basename(i)
                    # since the duplicte images has already removed from the parent_out_dir, to show the duplicate images wwe have to switch to 
                    # parent directory
                    image_dir.append(os.path.join(parent_dir, filename))
                images= [cv2.imread(image_path) for image_path in image_dir]
                # For visualisation, only 25 images are shown. So if name_list has more than 25 images, it shows only the first 25 images of name_lst
                if len(images) > 25:
                    images=images[0:25]
                fig1=plt.figure(figsize=(20,20))
                for i in images:
                    plt.subplot(5, 5, n)
                    plt.imshow(i) 
                    plt.tight_layout()
                    plt.axis('off')
                    n += 1
                st.pyplot(fig1)
            # function for seprating similar images
            def similar_photos():
                name_list=list()
                num_sim=0
                # here we use feature_list_back and img_path_back (since in these two lists, the information of deleted images has removed) 
                for i in range(len(features_list_back)):
                    for j in range(i+1, len(features_list_back)):
                        if  i !=j:
                            A=np.stack((features_list_back[i], features_list_back[j]))
                            A_sparse = sparse.csr_matrix(A)
                            similarities_sparse = cosine_similarity(A_sparse,dense_output=False)
                            sim= np.mean(similarities_sparse)
                            # move images with similarity of more than similarity_level (which is given by user and less than 0.99999 to a seprate folder)
                            if sim <=0.99999 and sim >=similarity_value:
                                if img_paths[i] in name_list or img_paths[j] in name_list:
                                    pass
                                else:
                                    img1 = cv2.imread(img_paths[i])
                                    img2 = cv2.imread(img_paths[j])
                                    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                                    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                                    fm1= cv2.Laplacian(gray1, cv2.CV_64F).var()
                                    fm2= cv2.Laplacian(gray2, cv2.CV_64F).var()
                                    if fm1 < fm2:
                                        pass
                                    else:
                                        src_path = img_paths[j]
                                        filename = os.path.basename(img_paths[j])
                                        simi= parent_out_dir +directory
                                        dst_path = os.path.join(simi, filename)
                                        if img_paths[i] not in name_list:
                                            name_list.append(img_paths[i])
                                        if img_paths[j] not in name_list:
                                            name_list.append(img_paths[j])
                                            shutil.move(src_path, dst_path, copy_function='copy2')
                                            num_sim +=1
                return num_sim, name_list
        
            num_sim, name_list=similar_photos()
            # Write the list of similar images 
            if len(name_list) ==0:
                st.write('There are no similar photos')
            else:
                st.write('List and visualisation of the similar images (maximum of 25 photos can be displayed):')
                st.write(name_list)
                n=1

                # plot similar images
                # since one of the similar files has already removed from the parent_out_dir, to show the similar images wwe have to switch to 
                # parent directory
                image_dir=[]
                for i in name_list:
                    filename = os.path.basename(i)
                    image_dir.append(os.path.join(parent_dir, filename))
                images= [cv2.imread(image_path) for image_path in image_dir]
                # Show only the forst 25 images if similar images are larger than 25
                if len(images) > 25:
                    images=images[0:25]
                fig1=plt.figure(figsize=(20,20))
                for i in images:
                    plt.subplot(5, 5, n)
                    plt.imshow(i) 
                    plt.tight_layout()
                    plt.axis('off')
                    n += 1
                st.pyplot(fig1)


# If delete duplicate and seprate similar and blur images is selected
    if option_query != '<select>' and option_input!= '<select>' and  input is not None  and (option_query == 'Delete duplicate photos and seprate similar and blur photos'):
        similarity_value=st.slider('Here you can specify the level of similarity between photos. Zero means there are not any similarities between photos and 100 means photos are hundred percent similar (recommended values > 85)', min_value=0, max_value=100)
        time.sleep(3) 
        # give the level of similarity
        similarity_value=float(similarity_value/100)
        Blur_value=st.slider('Here you can specify the threshold for blur photos (recommended value= 100) )', min_value=90, max_value=150)
        # 5 second time to select the contrast level from slider bar
        time.sleep(3) 
        Blur_value=float(Blur_value)

        # make similarity directory to save similar images
        directory = "Similar_images/"  
    # define the path of similarity directory
        path = os.path.join(parent_out_dir, directory) 
        isExist = os.path.exists(path)
    # Create the directory
        if not isExist:
            os.mkdir(path)
        # give warning regarding zero similarity level
        if similarity_value ==0:
            new_title = '<p style="font-family:sans-serif; color:Red; font-size: 22px;">Zero does not make sense.  Select a value higher than 0.</p>'
            st.markdown(new_title, unsafe_allow_html=True)
        # define the function of extracing image features
        if similarity_value !=0:
            def extract_features(img_path):
                img = load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                features = basemodel.predict(x)
                return features.flatten()

        # create list of images with their paths and list of image features
            directoryh=parent_out_dir
            img_paths = listdir(parent_out_dir)

            img_paths = [os.path.join(directoryh, img_paths[i]) for i in range(len(img_paths)) if (img_paths[i].endswith('jpg') or img_paths[i].endswith('jpeg') or img_paths[i].endswith('png')) ]
            features_list = [extract_features(path) for path in img_paths]
        # delecte duplicate images
        #@st.experimental_memo
        if similarity_value !=0:
            def photo_delete():
                num_del=0
                name_list=list()
                # since we have another loop for seprating similar images, here we have to create new feature_list and img_paths (since in the current loop, we cannot drop any element of these lists- loop is based on length of feature_list )
                features_list_back=features_list
                img_paths_back=img_paths
                for i in range(len(features_list)):
                    for j in range(i+1, len(features_list)):
                        if  i !=j:
                            # calculte similarity of two images
                            A=np.stack((features_list[i], features_list[j]))
                            A_sparse = sparse.csr_matrix(A)
                            similarities_sparse = cosine_similarity(A_sparse,dense_output=False)
                            sim= np.mean(similarities_sparse)
                            # delete images with higher similarities of 0.985
                            if sim >=0.985:
                                img1 = cv2.imread(img_paths[i])
                                img2 = cv2.imread(img_paths[j])
                                if img_paths[i] in name_list or img_paths[j] in name_list:
                                    pass
                                else:
                                    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                                    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                                    fm1= cv2.Laplacian(gray1, cv2.CV_64F).var()
                                    fm2= cv2.Laplacian(gray2, cv2.CV_64F).var()
                                    if fm1 < fm2:
                                        pass
                                    else:                           
                                        if img_paths[i] not in name_list:
                                            name_list.append(img_paths[i])
                                        if img_paths[j] not in name_list:
                                            name_list.append(img_paths[j])
                                            os.remove(img_paths[j])
                                            features_list_back=features_list_back[:j]+ features_list_back[j+1:]
                                            img_paths_back=img_paths_back[:j]+ img_paths_back[j+1:]
                                            num_del+=1
                # Write list of duplicate images
                if len(name_list) != 0: 
                    st.write('List and visualisation of the duplicate images (maximum of 25 photos can be displayed):')
                    st.write(name_list)
                else:
                     st.write('There are no duplicate images')                   
                return num_del, name_list, features_list_back, img_paths_back
            
    
            num_del, name_list, features_list_back, img_paths_back = photo_delete()
            n=1
        # plot duplicate images (only 25 images will be displayed)
        # get the path of images in name_list (name_list has the name of images which have been removed)
            if len(name_list) !=0:
                image_dir=[]
                for i in name_list:
                    filename = os.path.basename(i)
                    # since the duplicte images has already removed from the parent_out_dir, to show the duplicate images wwe have to switch to 
                    # parent directory
                    image_dir.append(os.path.join(parent_dir, filename))
                images= [cv2.imread(image_path) for image_path in image_dir]
                # For visualisation, only 25 images are shown. So if name_list has more than 25 images, it shows only the first 25 images of name_lst
                if len(images) > 25:
                    images=images[0:25]
                fig1=plt.figure(figsize=(20,20))
                for i in images:
                    plt.subplot(5, 5, n)
                    plt.imshow(i) 
                    plt.tight_layout()
                    plt.axis('off')
                    n += 1
                st.pyplot(fig1)
            # function for seprating similar images
            def similar_photos():
                name_list=list()
                num_sim=0
                # here we use feature_list_back and img_path_back (since in these two lists, the information of deleted images has removed) 
                for i in range(len(features_list_back)):
                    for j in range(i+1, len(features_list_back)):
                        if  i !=j:
                            A=np.stack((features_list_back[i], features_list_back[j]))
                            A_sparse = sparse.csr_matrix(A)
                            similarities_sparse = cosine_similarity(A_sparse,dense_output=False)
                            sim= np.mean(similarities_sparse)
                            # move images with similarity of more than similarity_level (which is given by user and less than 0.99999 to a seprate folder)
                            if sim <=0.99999 and sim >=similarity_value:
                                if img_paths[i] in name_list or img_paths[j] in name_list:
                                    pass
                                else:
                                    img1 = cv2.imread(img_paths[i])
                                    img2 = cv2.imread(img_paths[j])
                                    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                                    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                                    fm1= cv2.Laplacian(gray1, cv2.CV_64F).var()
                                    fm2= cv2.Laplacian(gray2, cv2.CV_64F).var()
                                    if fm1 < fm2:
                                        pass
                                    else:
                                        src_path = img_paths[j]
                                        filename = os.path.basename(img_paths[j])
                                        simi= parent_out_dir +directory
                                        dst_path = os.path.join(simi, filename)
                                        if img_paths[i] not in name_list:
                                            name_list.append(img_paths[i])
                                        if img_paths[j] not in name_list:
                                            name_list.append(img_paths[j])
                                            shutil.move(src_path, dst_path, copy_function='copy2')
                                            num_sim +=1
                return num_sim, name_list
        
            num_sim, name_list=similar_photos()
            # Write the list of similar images 
            if len(name_list) ==0:
                st.write('There are no similar photos')
            else:
                st.write('List and visualisation of the similar images (maximum of 25 photos can be displayed):')
                st.write(name_list)
                n=1

                # plot similar images
                # since one of the similar files has already removed from the parent_out_dir, to show the similar images wwe have to switch to 
                # parent directory
                image_dir=[]
                for i in name_list:
                    filename = os.path.basename(i)
                    image_dir.append(os.path.join(parent_dir, filename))
                images= [cv2.imread(image_path) for image_path in image_dir]
                # Show only the forst 25 images if similar images are larger than 25
                if len(images) > 25:
                    images=images[0:25]
                fig1=plt.figure(figsize=(20,20))
                for i in images:
                    plt.subplot(5, 5, n)
                    plt.imshow(i) 
                    plt.tight_layout()
                    plt.axis('off')
                    n += 1
                st.pyplot(fig1)

               # make a directora to put low contrast images
                directory = "Blur_images/"  
            # Path of this new directory in parent_out_dir
                path = os.path.join(parent_out_dir, directory) 
                isExist = os.path.exists(path)
            # if directory does not exist create it
                if not isExist:
                    os.mkdir(path)
                # If directory exists get features images

                def extract_features(img_path):
                    img = load_img(img_path, target_size=(224, 224))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    features = basemodel.predict(x)
                    return features.flatten()

                # List of image names and paths
                directoryh=parent_out_dir
                img_paths = listdir(parent_out_dir)
                img_paths = [os.path.join(directoryh, img_paths[i]) for i in range(len(img_paths)) if (img_paths[i].endswith('jpg') or img_paths[i].endswith('jpeg') or img_paths[i].endswith('png')) ]

                # seprate similar images
                def blur_photos():
                    name_list=list()
                    num_blur=0
                    for i in range(len(img_paths)):
                        img = cv2.imread(img_paths[i])
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        fm= cv2.Laplacian(gray, cv2.CV_64F).var()
                        if fm < Blur_value:
                            src_path = img_paths[i]
                            filename = os.path.basename(img_paths[i])
                            cont= parent_out_dir +directory
                            dst_path = os.path.join(cont, filename)
                            if img_paths[i] not in name_list:
                                name_list.append(img_paths[i])
                            num_blur +=1
                            shutil.move(src_path, dst_path, copy_function='copy2')

                    return num_blur, name_list
                
                num_blur, name_list=blur_photos()
                if len(name_list) !=0:
                    st.write('List and visualisation of the blur images (maximum of 25 photos can be displayed):')
                    st.write(name_list)
                else:
                    st.write('There are no blur images')               
                n=1

                    # Plot low contrast images (only 25 images will be displayed)
                    # get the path of images in name_list (name_list has the name of images which have been removed)
                    # since one of the similar files has already removed from the parent_out_dir, to show the duplicate image wwe have to switch to 
                    # parent directory
                if len(name_list) !=0:
                    image_dir=[]
                    for i in name_list:
                        filename = os.path.basename(i)
                        image_dir.append(os.path.join(parent_dir, filename))
                    images= [cv2.imread(image_path) for image_path in image_dir]
                        # Since only 25 images are shown here, if images are more than 25, only the first 25 similar images are shown
                    if len(images) > 25:
                        images=images[0:25]
                        # plot similar images
                    fig1=plt.figure(figsize=(20,20))
                    for i in images:
                        plt.subplot(5, 5, n)
                        plt.imshow(i) 
                        plt.tight_layout()
                        plt.axis('off')
                        n += 1
                    st.pyplot(fig1)
    
# write statistical information
    st.subheader('Statistical information about the input and output:')
    ## caculate number of images in input file
    if option_input != '<select>' and  input is not None:
        num_im=0
        for i in listdir(parent_dir):
            if i.split('.')[-1] =='jpg' or i.split('.')[-1] =='jpeg' or i.split('.')[-1] =='png' :
                num_im+=1
    # write the number of files if no option regarding managing your photo (delete dulicate, 
    # seprate similar images or delete duplicate and seprete similar images) 
    # were selected 
    if option_input != '<select>' and option_query == '<select>' and  input is not None:
        d={'Number of input files':num_im}
        df = pd.DataFrame(data=d, index=[0])
        st.table(df.assign(hack='').set_index('hack'))
    # write the number of files and number of deleted duplicate images if delete dulicate photos
    # was selected    
    elif option_input != '<select>' and (option_query == 'Delete duplicate photos') and  input is not None:
        d={'Number of input files':num_im,\
                                'Number of deleted files':num_del}
        df = pd.DataFrame(data=d, index=[0])
        st.table(df.assign(hack='').set_index('hack'))
    # write the number of files and number of deleted duplicate images if delete dulicate photos
    # was selected    
    elif option_input != '<select>' and (option_query == 'Detect blur photos') and  input is not None:
        d={'Number of input files':num_im,\
                                'Number of blur photos':num_blur}
        df = pd.DataFrame(data=d, index=[0])
        st.table(df.assign(hack='').set_index('hack'))
    # write the number of files and number of similar images if Seprate similar photos
    # was selected 
    elif option_input != '<select>' and (option_query == 'Seprate similar photos') and  input is not None and similarity_value !=0:
        d={'Number of input files':num_im,\
                                'Number of similar images':num_sim}
        df = pd.DataFrame(data=d, index=[0])       
        st.table(df.assign(hack='').set_index('hack'))

    # write the number of files, number of deleted images  and number of similar images if delete duplicate photos and seprate similar photos
    # was selected
    elif option_input != '<select>' and (option_query == 'Delete duplicate photos and seprate similar photos') and  input is not None and similarity_value != 0:
          
        d={'Number of input files':num_im,\
                                'Number of deleted files':num_del,'Number of similar images':num_sim}
        df = pd.DataFrame(data=d, index=[0])

        st.table(df.assign(hack='').set_index('hack'))
    # write the number of files, number of deleted images  and number of similar and blur images if delete duplicate photos and seprate similar and blur photos
    # was selected
    elif option_input != '<select>' and (option_query == 'Delete duplicate photos and seprate similar and blur photos') and  input is not None and similarity_value != 0:
          
        d={'Number of input files':num_im,\
                                'Number of deleted files':num_del,'Number of similar images':num_sim,'Number of blur images':num_blur}
        df = pd.DataFrame(data=d, index=[0])

        st.table(df.assign(hack='').set_index('hack'))

    # Write output file
    if option_out != '<select>' and option_out=='download as a zip file':
        isExist=os.path.exists('./output.zip')
        #if output.zip exists in the current directory it is removed and a new one is created
        if  isExist:
            os.remove('output.zip')
        shutil.make_archive('output', 'zip', parent_out_dir)
        with open('./output.zip', 'rb') as f:
            st.download_button('Download Zip', f, file_name='output.zip')  
    elif option_out != '<select>' and option_out=='save in a specific path':
        # the outputs are writen to a directory calles 'out' in the given path
        isExist = os.path.exists(output_dir+'out')
        if isExist and (output_dir != parent_out_dir):
            new_title = '<p style="font-family:sans-serif; color:Red; font-size: 22px;">Warning: There is a directory with the name \
                ''out'' in the given output directory. The output data will be merged with the contents of the existing folder. </p>'
            st.markdown(new_title, unsafe_allow_html=True)            
            shutil.copytree(parent_out_dir, output_dir+'out', dirs_exist_ok=True)
        elif isExist and (output_dir == parent_out_dir+'out'):
            pass
        else:
            shutil.copytree(parent_out_dir, output_dir+'out')







