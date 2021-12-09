import numpy as np
import pims

def make_layer_matrices(semantic_output):

    '''
    Description:
    Turns output from semantic segementation into an input usable for layer 
    replacement.

    Inputs:
        semantic_output: an m x n np matrix array that contains layer numbers 

    Outputs:
        layer_matrices: a 3 dimensional l x m x n np array that contains 1s and 0s

        unique_layers: a 1 dimensional np array of size l with label numbers that correspond
        to each layer matrix 


    '''
    
    unique_layers = np.unique(semantic_output)
    amount_layers = unique_layers.shape[0]

    zerosArray = np.zeros(semantic_output.shape)
    onesArray = np.ones(semantic_output.shape)

    layer_matrices = np.empty((amount_layers, semantic_output.shape[0], semantic_output.shape[1]))
    
    for index, layer_number in enumerate(unique_layers):
        layer_matrix = np.where((semantic_output == layer_number),
        onesArray,
        zerosArray)

        layer_matrices[index] = layer_matrix

    
    return layer_matrices, unique_layers
        #TODO: add layer matrix to larger return value then return after loop


def open_file(file_path):

    '''
    Description:
    first method called after pressing 'segment' in front-end

    Inputs:
        file_path: string detailing path of file.
        Extension already approved

    Outputs:
    an f x m x n array
    '''

    images = pims.open(file_path)
    #TODO: Verify that its readable and send ERROR if its not

    outputArray = np.empty((images.shape[:2]))

    for frame in images:

        x = 1
        #perform semantic segmentation, give outputed layer to outputArray

    return outputArray


def test_pims(filepath):
    #do things here to test out
    print("commencing testing")

    images = pims.open(filepath)

    print(images[0].shape)
