import tensorflow.keras as keras
import extract_bottleneck_features
import cv2      
import gradio as gr
import numpy as np      
from glob import glob
from keras.preprocessing import image           
InceptionV3_model = keras.models.load_model("saved_models/weights.best.InceptionV3.hdf5",)

dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
labels =  dog_names

def extract_InceptionV3(tensor):
	from keras.applications.inception_v3 import InceptionV3, preprocess_input
	return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def extract_Resnet50(tensor):
	from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
	return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


###########################################

from tensorflow.keras.applications.resnet50 import preprocess_input

######################################

import tensorflow as tf
from keras.preprocessing import image                  
from tqdm import tqdm

######################################

from tensorflow.keras.applications.resnet50 import ResNet50
# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

from keras.preprocessing import image                  
from tqdm import tqdm



from tensorflow.keras.applications.resnet50 import preprocess_input

def ResNet50_predict_labels(img):
    # returns prediction vector for image located at img_path
    img = np.expand_dims(img, axis=0)
    img = preprocess_input((img))
    return np.argmax(ResNet50_model.predict(img)) 


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    #img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    #x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(img_path, axis=0)



# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')



def face_detector(image):
    """
    returns "True" if face is detected in image stored at image

    """
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    if len(faces) > 0:
        return "Number of human faces found in this image: {}". format(len(faces))
    else:
        return "There are no human faces in this image"




def InceptionV3_prediction_breed(img_path):
    """
    Return: dog breed that is predicted by the model
    input: image
    """
    
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = InceptionV3_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)].split('.')[-1]



def dog_detector(img):
    """
    input: uploaded image by user
    return: "True" if a dog is detected in the image stored at img
    """
    
    prediction = ResNet50_predict_labels(img)
    return ((prediction <= 268) & (prediction >= 151)) 

def identify_dog_app(img):
    """This function predicts the breed of the human or dog"
    
    input: uploaded image by user
    Return: dog or human, and breed of the uploaded image
    """
    
    breed = InceptionV3_prediction_breed(img)
    if dog_detector(img):
        return("This looks like a dog and its breed is:"),"{}".format(breed)
    elif face_detector(img):
        return("This looks like a human but might be classified as a dog of the following breed:"),"{}".format(breed)
    else:
        return("I have no idea what this might be. Please upload another image!"), ("Not applicable")




image = gr.inputs.Image(shape=(224, 224),  label="Image")
label = gr.outputs.Label(num_top_classes=1)    

iface = gr.Interface(   
    fn=identify_dog_app,
    inputs=image,
    outputs=[gr.outputs.Label(label="Human or Dog?"), gr.outputs.Label(label="Breed:")],  
    title="Human or dog Identification - Breed Classification",
    #description ="Please find the jypyter notebook on ___",
    article = 
    '<b><span style="color: #ff9900;">Acknowledgement:</span></b><br/>'
    +'<p><span style="color: #ff9900;">I would like to express my special thanks of gratitude'
    +'to Misk &amp; Sdaia for giving me the opportunity to enrol in "Data Scientist" Udacity nanodegree,'
    +'&nbsp;as well as to my mentor Mr. Haroon who was of great help during my learning journey.</span></p>'
    +'<p><span style="color: #ff9900;">This is my capstone project and herewith I finish this ND.</span></p>',

    theme="dark-huggingface"

)

iface.launch(share=True)




