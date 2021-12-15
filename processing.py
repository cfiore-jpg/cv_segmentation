import numpy as np
import pims
from skimage.transform import resize
from backend.predict import inference
from backend.nets.SegNet import *
import backend.core.config as cfg
from layerReplacement.read_labels import read_labels

from layerReplacement.layer_replacement import layer_replace

primary_input = None
layer_matrices = None
unique_layers = None
number_to_label = read_labels("./layerReplacement/labels.json")


def open_file(file_path):
    """
    Description:
    first method called after pressing 'segment' in front-end. SEGMENTATION OCCURS HERE

    Inputs:
        file_path: string detailing path of file.
        Extension already approved

    Outputs:
    a 1 dimensional np array of size l with label numbers that correspond to each
    layer matrix
    """
    images = pims.open(file_path)

    global primary_input
    primary_input = file_path

    # obtaining f x m x n output array
    output_array = get_segmented_layers(images)

    global layer_matrices
    global unique_layers
    # passing output_array to make_layer_matrices, which takes in f x m x n
    # and outputs l x f x m x n in 1s and 0s
    layer_matrices, unique_layers = make_layer_matrices(output_array)

    return unique_layers


def isolate_segment(frame):
    """
    Description:

    Inputs:
        frame: an m x n x 3 "Frame" that extends np array

    Outputs:
        an m x n array of the processed segmented image
    """

    model = SegNet_VGG16(cfg.input_shape, cfg.num_classes)
    model.load_weights("backend/segnet_weights.h5")

    result = inference(model, frame)

    # maybe reformat??

    return result


def get_segmented_layers(images):
    """
    Description:

    Inputs:
        images: a pims file with f frame, and m x n x 3 "Frames" that extend np array
        will loop through f

    Outputs:
        an f x m x n np array containing segmentation output.
    """

    # TODO: change output_array size if desired -- future application
    output_array = np.empty((len(images), 320, 320))

    for i in range(len(images)):  # loop through every frame
        frame = frame_process(images[i])
        layer_matrix = isolate_segment(frame)
        output_array[i] = layer_matrix

    return output_array


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
    return resize(frame, (320, 320, 3))


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


def pre_layer_replace(layer_dict):
    """
   Description:

   Inputs:
   #TODO: figure out form of inputs for secondary-inputs
   layer_dict: A dictionary of size l

   Outputs:

   """
    secondary_filepaths = []
    print("layer_matrices")
    print(layer_matrices)
    print("unique_layers")
    print(unique_layers)

    if unique_layers is not None:
        for layer_number in unique_layers:
            secondary_input_list = layer_dict[number_to_label[layer_number]]
            print(secondary_input_list)

            if secondary_input_list[0] is "Nothing":
                secondary_filepaths.append(primary_input)
            elif secondary_input_list[0] is "Video":
                secondary_filepaths.append(secondary_input_list[1])
            elif secondary_input_list[0] is "Image":
                secondary_filepaths.append(secondary_input_list[1])

    print("PRE LAYER REPALCING")
    layer_replace(layer_matrices, secondary_filepaths)

    # TODO: fill this out. layer_replacement.py gets used here finally


if __name__ == '__main__':
    x = 1
