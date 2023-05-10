# ManageImage

This is an app for Managing your photos. It deletes all the duplicated photos and seprate similar or blur photos. The level of blurness and similarity are defined by users. 

The input of this app can be a zip file or a directory. In case of zip file, the zip file should be located in the current directory (where the source code is). The output can be either in the form of a zip file or a directory. 

The user has five options in managing his/her photos: 

1) Delete duplicate photos, 
2) Seprate similar photos, 
3) Delete duplicate photos and seprate similar photos, 
4) Detect blur photos or 
5) Delete duplicate photos and seprate similar and blur photos. 
 
In case of selecting seprate similar photos, a directory naned 'Smilar_images' and keep the best quality one in the origin directory (default output directory is ./output_image) and move the rest of similar photos to the Smilar_images directory (the critery for selecting the best photo is based on blurness of photis). In the case of detect blur photos, a diirectory called 'blur_images' is created in the output directory and all the blur images are moved to this directory.
