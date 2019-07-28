import cv2
import glob
import os
import numpy as np
import pandas as pd
import tensorflow as tf

import skimage.io
import mvnc.mvncapi as mvnc

LABELS = []

# OpenCV object for video capture
CAMERA = None


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


def load_graph(device):

    # Read the graph file into a buffer
    with open('./output_ncs/traffic_inference_ncs.graph', mode='rb') as f:
        blob = f.read()

    # Load the graph buffer into the NCS
    graph = device.AllocateGraph(blob)

    return graph


def infer_image(graph, processed):

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
    # skimage.io.imshow(img)
    # skimage.io.show()


def close_ncs_device( device, graph ):
    graph.DeallocateGraph()
    device.CloseDevice()
    CAMERA.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Create a VideoCapture object
    CAMERA = cv2.VideoCapture(0)

    # Set camera resolution
    CAMERA.set(cv2.CAP_PROP_FRAME_WIDTH, 620)
    CAMERA.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    device = open_ncs_device()
    graph = load_graph(device)
    LABELS = load_labels()

    if not CAMERA.isOpened():
        print("Could not open webcam")
        exit()

    while CAMERA.isOpened():
        # read frame from camera
        status, frame = CAMERA.read()
        frame_flipped = cv2.flip(frame, 1)

        x1, y1, x2, y2 = 100, 100, 300, 300
        img_cropped = frame[y1:y2, x1:x2]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.imshow("live webcam", frame_flipped)

        processed = preprocess_im(img_cropped)
        infer_image(graph, processed)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    close_ncs_device(device, graph)