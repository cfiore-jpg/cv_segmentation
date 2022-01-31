import numpy as np
import pims as pim
from skimage.transform import resize
from layerReplacement.read_labels import read_labels
from layerReplacement.layer_replacement import layer_replace
from matplotlib import pyplot as plt
from pickletools import uint8
import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
from gluoncv.data.transforms.presets.segmentation import test_transform

primary_input = None
layer_matrices_global = None
unique_layers_global = None


def open_file(file_path):
    """
    Description:
    first method called after pressing 'segment' in front-end. SEGMENTATION OCCURS HERE

        file_path: string detailing path of file.
        Extension already approved

    Outputs:
    a 1 dimensional np array of size l with label numbers that correspond to each
    layer matrix
    """
    images = pim.open(file_path)

    global primary_input
    primary_input = file_path

    # obtaining f x m x n output array
    output_array = get_segmented_layers(images)

    global layer_matrices_global
    global unique_layers_global
    # passing output_array to make_layer_matrices, which takes in f x m x n
    # and outputs l x f x m x n in 1s and 0s
    layer_matrices_global, unique_layers_global = make_layer_matrices(output_array)

    return unique_layers_global


def isolate_segment(frame, model):
    """
    Description:

    Inputs:
        frame: an m x n x 3 "Frame" that extends np array

    Outputs:
        an m x n array of the processed segmented image
    """

    ctx = ctx = mx.cpu(0)

    realFrame = mx.nd.array(frame, ctx=ctx)
    img = test_transform(realFrame, ctx)
    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()


    return predict


def get_segmented_layers(images):
    """
    Description:

    Inputs:
        images: a pims file with f frames, and m x n x 3 "Frames" that extend np array
        will loop through f

    Outputs:
        an f x m x n np array containing segmentation output.
    """


    # TODO: change output_array size if desired -- future application --YEAHHHH
    output_array = np.empty((len(images), images[0].shape[0], images[0].shape[1]))
    model = gluoncv.model_zoo.get_model('psp_resnet101_ade', pretrained=True)

    for i in range(len(images)):  # loop through every frame
        frame = frame_process(images[i])
        #THIS IS WHERE SEGMENTING HAPPENS change here or add "options"
        layer_matrix = isolate_segment(frame, model)
        output_array[i] = layer_matrix
    return output_array


@pim.pipeline
def frame_process(frame):
    # Eliminates 4th dimensions if a png or gif

    new_frame = frame[:, :, :3]
    if frame.shape[2] == 4:
        return frame[:, :, :3]
    else:
        return frame


@pim.pipeline
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

    for frame_index, frame in enumerate(semantic_output):
        fig = plt.figure()
        for layer_index, layer_number in enumerate(unique_layers):
        #for frame_index, frame in enumerate(semantic_output):
            layer_matrix = np.where((frame == layer_number),
                                    ones_array,
                                    zeros_array)
            layer_matrices[layer_index][frame_index] = layer_matrix[0]
            # now set up matplot to show layers
            ax = fig.add_subplot(1, len(unique_layers), layer_index + 1)
            ax.set_title(layer_number)
            plt.imshow(layer_matrix[0])
    plt.show()

    return layer_matrices, unique_layers

def show_layer(layer):
    print(layer)


def pre_layer_replace(layer_dict):
    """
   Description:

   Inputs:
   #TODO: figure out form of inputs for secondary-inputs
   layer_dict: A dictionary of size l

   Outputs:

   """
    secondary_filepaths = []
    # print(layer_matrices)
    if layer_matrices_global is not None:
        print("layer_matrices is not none")
        print(layer_matrices_global.shape)
        print(unique_layers_global)

    if unique_layers_global is not None:
        for layer_number in unique_layers_global:
            print(layer_number)
            secondary_input_list = layer_dict[layer_number] #TODO:figure out if this change is doing most of the error work
            print(layer_dict[layer_number])

            if secondary_input_list[0] == "Nothing":
                secondary_filepaths.append(primary_input)
            elif secondary_input_list[0] == "video":

                secondary_filepaths.append(secondary_input_list[1])
            elif secondary_input_list[0] == "image":
                secondary_filepaths.append(secondary_input_list[1])

    final_video = layer_replace(layer_matrices_global, secondary_filepaths)

    plt.imshow(final_video[0])
    plt.show()
    for i in range(len(final_video)):
        str1 = "static/data/output/frame"
        str2 = str(i)
        str3 = ".jpeg"
        plt.imsave(str1 + str2 + str3, final_video[i])


if __name__ == '__main__':
    x = 1
