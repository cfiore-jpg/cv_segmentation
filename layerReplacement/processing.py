import numpy as np
import pims
from skimage.transform import resize

# import layer_replacement

primary_input = None


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

    global primary_input
    primary_input = file_path

    output_array = np.empty((len(images), images.frame_shape[0], images.frame_shape[1]))

    for frame in images:
        frame = frame_process(frame)
        frame = segment_resize(frame)
        # perform semantic segmentation, give outputed layer to outputArray

    return output_array


def test_pims(filepath):
    # do things here to test out
    print("commencing test_pims")

    global primary_input
    primary_input = filepath
    images = pims.open(filepath)

    images = frame_process(images)

    made_image = resize(images[0], (128, 128, 3))

    print("feature1")
    print(images)
    print("feature2")
    print(resize(images[0], (128, 128, 3)))
    print("feature3")
    print(made_image.shape)

    print("end testPIMS")


def test2():
    print(primary_input)


def pre_layer_replace():
    """
   Description:

   Inputs:
   #TODO: figure out form of inputs for secondary-inputs

   Outputs:

   """

    # TODO: fill this out. layer_replacement.py gets used here finally


@pims.pipeline
def frame_process(frame):
    # Eliminates 4th dimensions if a png or gif

    new_frame = frame[:, :, :3]
    if frame.shape[2] == 4:
        return frame[:, :, :3]
    else:
        return frame


@pims.pipeline
def segment_resize(frame):
    frame = frame_process(frame)
    return resize(frame, (128, 128, 3))
