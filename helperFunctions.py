import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


def configure_for_performance(ds, shuffle=True):
    ds = ds.cache()
    if shuffle: ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_png(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [image_height, image_width])

def processPathTrain(*argsv):    
    img = tf.io.read_file(argsv[0])
    img = decode_img(img)
    return (img,*argsv[1:])

def processPathTest(file_path):    
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img

def getTfDataset(path = r"data\trainingData.csv", img_height = 220, img_width = 220, 
                       batchSize = 32, angle=True, speed=True, shuffle=True):
    """
    This method is used to convert the images in a folder to tensorflow dataset type (td.data.Dataset).
    This combines the images with the respective labels (angle, speed) by matching the image_id
    It is expected that image name is a image id and the same is availabel in the labels_file
    
    Arguments:
    - bucket: type=str, gsc bucket name, example 'bucket-self-driving-car-XXXXXX'
    - images_path: type=str, path of the folder where images are, example: 'input/v0/training_data'
    - labels_file_path: type=str, path of the csv file where image labels and for image id are present, exampel: 'input/v0/training_norm.csv'
    - image_height: default value 220,
    - image_width: default value 220
    - batch_size: default value 32
    
    Returns:
     tensor dataset which contains: image (array), angle , speed 
    """
    ##varibales we use in other helper functions
    global image_height, image_width, batch_size
    image_height, image_width, batch_size = img_height, img_width, batchSize
    
    ## Getting image names
    training_df = pd.read_csv(path)
    image_names = training_df["image_path"].to_list()
    angles = training_df["angle"].to_list()
    speeds = training_df["speed"].to_list()
    
    ## Converting image names along with the labels into tensorflow dataset    
    if angle and speed:
        list_ds = tf.data.Dataset.from_tensor_slices((image_names, angles, speeds))
    elif angle:
        list_ds = tf.data.Dataset.from_tensor_slices((image_names, angles))
    elif speed:
        list_ds = tf.data.Dataset.from_tensor_slices((image_names, speeds))   
    
    ## Now converting image path into image
    tf_ds = list_ds.map(processPathTrain,num_parallel_calls=tf.data.AUTOTUNE)
    ## Making the dataset performant 
    tf_ds = configure_for_performance(tf_ds,shuffle)
    
    return tf_ds
    
def getTestDataset(file_path=None, images_path = None, img_height = 220, img_width = 220, batchSize = 32):
    ##varibales we use in other helper functions
    global image_height, image_width, batch_size
    image_height, image_width, batch_size = img_height, img_width, batchSize

    
    ## Getting image names
    # If file given
    if file_path is not None:
        test_data_df = pd.read_csv(file_path)
        image_names = test_data_df["image_path"].to_list()
        image_ids = test_data_df["image_id"].to_list()
    elif images_path is not None:
        image_names = os.listdir(images_path)
        image_ids = [image_name.split(".")[0] for image_name in image_names]
        image_names = [os.path.join(images_path,image_name) for image_name in image_names]
    
    ## Converting image names into tensorflow dataset
    list_ds = tf.data.Dataset.from_tensor_slices(image_names)
    ## Now converting image path into image
    tf_ds = list_ds.map(processPathTest,num_parallel_calls=tf.data.AUTOTUNE)
    ## Making the dataset performant 
    tf_ds = configure_for_performance(tf_ds, shuffle=False)
    
    
    return tf_ds, image_ids


def generateTestData(root_folder:str, nimages_per_class= 1500):
    """
    This method takes the root folder of the data and generates a csv file with test data of images.
    If you have number of folders under the root folder having data, evry folder
    must contain "training_data" folder inside. This function goes through all images and samples 
    `2 * nimages_per_class` number of images.
    """
    
    all_images = []
    all_training_norm = pd.DataFrame()
    
    for folder in os.listdir(root_folder):
        data_folder = os.path.join(root_folder,folder)
        if not os.path.isdir(data_folder): continue #if a listed object is not a folder do nothing
        ## read training_norm.csv
        df = pd.read_csv(os.path.join(data_folder,"training_norm.csv"))
        all_training_norm = pd.concat([all_training_norm, df], ignore_index=True)
        ## Images
        images_path = os.path.join(data_folder,"training_data")
        images = os.listdir(images_path)    
        images = [(image.split(".")[0],os.path.join(images_path,image)) for image in images]
        all_images += images
    #Converting list to pandas data frame
    all_images = pd.DataFrame(all_images,columns=["image_id","image_path"])
    all_images["image_id"] = all_images["image_id"].astype("int64")
    #Join the labels and image paths
    total_data = pd.merge(all_training_norm,all_images,how="inner",on="image_id")

    ## Spli the data into classes and sample equal ammount of images
    df_zero_class = total_data[total_data["speed"]==0.0].sample(n=nimages_per_class,random_state=51)
    df_1_class = total_data[total_data["speed"]==1.0].sample(n=nimages_per_class,random_state=51)
    ##Merging them into single data frame again and sufflin them and writing them
    df_test = pd.concat([df_zero_class,df_1_class],ignore_index=True)
    df_test = df_test.sample(frac=1)
    df_test.to_csv(os.path.join(root_folder,"testData.csv"),index=False)
    print(f"Test data saved successfully to {os.path.join(root_folder,'testData.csv')}")
        
        
def generateTrainData(root_folder:str, validation_split=0.2, test_data = "data\testData.csv"):
    """
    This method takes the root folder of the data and generates a csv file with test data of images.
    If you have number of folders under the root folder having data, evry folder
    must contain "training_data" folder inside. This function goes through all images and samples 
    `2 * nimages_per_class` number of images.
    """
    
    all_images = []
    all_training_norm = pd.DataFrame()
    
    for folder in os.listdir(root_folder):
        data_folder = os.path.join(root_folder,folder)
        if not os.path.isdir(data_folder): continue #if a listed object is not a folder do nothing
        ## read training_norm.csv
        df = pd.read_csv(os.path.join(data_folder,"training_norm.csv"))
        all_training_norm = pd.concat([all_training_norm, df], ignore_index=True)
        ## Images
        images_path = os.path.join(data_folder,"training_data")
        images = os.listdir(images_path)    
        images = [(image.split(".")[0],os.path.join(images_path,image)) for image in images]
        all_images += images
    #Converting list to pandas data frame
    all_images = pd.DataFrame(all_images,columns=["image_id","image_path"])
    all_images["image_id"] = all_images["image_id"].astype("int64")
    #Join the labels and image paths
    total_data = pd.merge(all_training_norm,all_images,how="inner",on="image_id")
    total_data.index = total_data["image_id"]
    #Test data
    df_test = pd.read_csv(test_data)
    df_test.index = df_test["image_id"]
    training_data = total_data.loc[total_data.index.difference(df_test.index)]
    training_data = training_data[training_data["speed"]<=1.0] # only 0s and ones
    train, valid = train_test_split(training_data,test_size=validation_split,stratify=training_data["speed"])
    train.to_csv(os.path.join(root_folder,'trainingData.csv'),index=False)
    valid.to_csv(os.path.join(root_folder,'validationData.csv'),index=False)

    return os.path.join(root_folder,'trainingData.csv'),os.path.join(root_folder,'validationData.csv')
        
def getPredicitons(model_path, out_filename, test_data_file= None, test_images_path=None,
                   img_height = 220, img_width = 220, batchSize = 32, makeDataframe=False):
    """    
    This helper function is useful for getting predictions for the data.
    Two mandatory arguments model_path (which model to use), out_filename (output predictions will be
    written to this csv file name)
    
    You can get predictions from both, images in the folder or images paths in the csv file.
    If you have images in the folder set test_images_path = path\to\images
    If you have images paths in the csv file then set test_data_file = path\to\file
    
    returns: a data frame with image id, angle, speed if you set   makeDataframe to True
    else just predictions
    
    Examples
    ---------
    
    pred_df = getPredicitons(r"model_outputs\toy_resnet\saved_model",
                         r"model_outputs\toy_resnet\submission_predictions\submission.csv",
                         test_images_path = r"data\v0\test_data",
                         img_height = 220, 
                         img_width = 220, 
                         batchSize = 32,
                         makeDataframe=True)
    
    """
    #get the data
    if test_data_file is not None:
        test_ds, image_ids = hf.getTestDataset(file_path = test_data_file)
    elif test_images_path is not None:
        test_ds, image_ids = getTestDataset(images_path = test_images_path)
    #Load the model
    model = tf.keras.models.load_model(model_path)
    #get predictions
    predictions = model.predict(test_ds)
    if not makeDataframe:
        return predictions
    #make dataframe
    df = pd.DataFrame()
    df["image_id"] = image_ids
    df["angle"] = predictions[0]
    speed = np.where(predictions[1] < 0.5,0,1)
    df["speed"] = speed
    
    df.to_csv(out_filename,index=False)
    print(f"Written predictions to {out_filename}")
    
    return df                
