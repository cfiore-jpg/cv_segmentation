import numpy as np
import pims


def make_layer_matrices(semantic_output):
    """
    Description:
    Turns output from semantic segmentation into an input usable for layer
    replacement.

    Inputs:
        semantic_output: an f x m x n np matrix array that contains layer numbers for all frames

    Outputs:
        layer_matrices: a 4 dimensional l x f x m x n np array that contains 1s and 0s

        unique_layers: a 1 dimensional np array of size l with label numbers that correspond
        to each layer matrix
    """

    unique_layers = np.unique(semantic_output)
    amount_layers = unique_layers.shape[0]

    zeros_array = np.zeros(semantic_output.shape)
    ones_array = np.ones(semantic_output.shape)

    layer_matrices = np.empty((amount_layers, semantic_output.shape[0], semantic_output.shape[1],
                               semantic_output.shape[2]))

    for layer_index, layer_number in enumerate(unique_layers):
        for frame_index, frame in enumerate(semantic_output):
            layer_matrix = np.where((frame == layer_number),
                                    ones_array,
                                    zeros_array)

            layer_matrices[layer_index][frame_index] = layer_matrix

    return layer_matrices, unique_layers
    # TODO: add layer matrix to larger return value then return after loop


def open_file(file_path):
    """
    Description:
    first method called after pressing 'segment' in front-end. SEGMENTATION OCCURS HERE

    Inputs:
        file_path: string detailing path of file.
        Extension already approved

    Outputs:
    an f x m x n np array containing segmentation output.
    """

    images = pims.open(file_path)
    # TODO: Verify that its readable and send ERROR if its not

    outputArray = np.empty((images.shape[:2]))

    for frame in images:
        x = 1
        # perform semantic segmentation, give outputed layer to outputArray

    return outputArray


def test_pims(filepath):
    # do things here to test out
    print("commencing testing")

    images = pims.open(filepath)

    print(images[0].shape)


# TODO: incorporate a pims-pipeline method

def pre_layer_replace():
    """
   Description:

   Inputs:

   Outputs:

   """

    # TODO: fill this out. layer_replacement.py gets used here finally
