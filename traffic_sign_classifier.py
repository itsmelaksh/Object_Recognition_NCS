import cv2
import glob
import os
import numpy as np
import pandas as pd
import tensorflow as tf

import skimage.io
import mvnc.mvncapi as mvnc

from PIL import Image

IMAGES_DIR = './data/test'
LOAD_PNG_PATTERN = 'sign*.png'
LABELS = []

def load_data():
    # Convert pickled data to human readable images
    image_files = os.path.join(IMAGES_DIR, LOAD_PNG_PATTERN)

    # Sort file names in alphabetical order to line up with labels
    files = glob.glob(image_files)
    files.sort()

    # Load images and save in X matrix. Convert to numpy array.
    x = []
    for file in files:
        img = Image.open(file)
        x.append(np.asarray(img.copy()))
        img.close()
    x = np.array(x)

    # Return images
    return x


def load_labels():
    labels = pd.read_csv('labels.csv')['SignName']

    return labels

def flatten(image):
    return image.flatten()


def normalize_im(image):
    mini, maxi = np.min(image), np.max(image)
    return (image - mini) / (maxi - mini) * 2 - 1


def preprocess_im(image):
    image = cv2.resize(image, (28, 28)) # Resize image
    image = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2YUV))[0] # Convert image to grayscale
    image = cv2.equalizeHist(image) # Equalize image
    image = normalize_im(image)
    return np.expand_dims(image, axis=2)



def open_ncs_device():

    # Look for enumerated NCS device(s); quit program if none found.
    devices = mvnc.EnumerateDevices()
    if len( devices ) == 0:
        print( "No devices found" )
        quit()

    # Get a handle to the first enumerated device and open it
    device = mvnc.Device( devices[0] )
    device.OpenDevice()

    return device


def load_graph( device ):

    # Read the graph file into a buffer
    with open('./output_ncs/traffic_inference_ncs.graph', mode='rb') as f:
        blob = f.read()

    # Load the graph buffer into the NCS
    graph = device.AllocateGraph(blob)

    return graph


def infer_image(graph, img, processed):

    # The first inference takes an additional ~20ms due to memory 
    # initializations, so we make a 'dummy forward pass'.
    graph.LoadTensor(processed.astype(np.float16), '')
    output, userobj = graph.GetResult()

    # Load the image as a half-precision floating point array
    graph.LoadTensor(processed.astype(np.float16), '')

    # Get the results from NCS
    output, userobj = graph.GetResult()

    # # Sort the indices of top predictions
    order = output.argsort()[::-1][0]
    print("prediction:", LABELS[order])

    # # Get execution time
    # inference_time = graph.GetGraphOption( mvnc.GraphOption.TIME_TAKEN )

    # # Print the results
    # print( "\n==============================================================" )
    # print( "Top predictions for", ntpath.basename( ARGS.image ) )
    # print( "Execution time: " + str( numpy.sum( inference_time ) ) + "ms" )
    # print( "--------------------------------------------------------------" )
    # for i in range( 0, NUM_PREDICTIONS ):
    #     print( "%3.1f%%\t" % (100.0 * output[ order[i] ] )
    #            + labels[ order[i] ] )
    # print( "==============================================================" )

    # # If a display is available, show the image on which inference was performed
    # if 'DISPLAY' in os.environ:
    skimage.io.imshow(img)
    skimage.io.show()


if __name__ == '__main__':
    device = open_ncs_device()
    graph = load_graph(device)

    n = 10
    x = load_data()
    x_test = x[np.random.choice(len(x), n)]

    LABELS = load_labels()

    for img in x_test:
        processed = preprocess_im(img)
        infer_image(graph, img, processed)
