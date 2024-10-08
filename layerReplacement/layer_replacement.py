import numpy as np
from skimage.transform import resize
import pims
from matplotlib import pyplot as plt


def get_processed_matrix(layer_matrix, image_matrix):
    """
    Description:

    Inputs:
        image_matrix: An m x n x 3 numpy array where m and n are the dimensions of
        the image.

        layer_matrix: An m x n numpy matrix. Only contains 0s and 1s

    Outputs:
        An m x n x 3. Will have values identical to image_matrix, but only where
        layer_matrix had 1s in the same spot.

    """
    # plt.imshow(layer_matrix)
    # plt.show()

    processed = image_matrix
    processed[:, :, 0] = np.multiply(layer_matrix, image_matrix[:, :, 0])
    processed[:, :, 1] = np.multiply(layer_matrix, image_matrix[:, :, 1])
    processed[:, :, 2] = np.multiply(layer_matrix, image_matrix[:, :, 2])
    # plt.imshow(processed)
    # plt.show()

    return processed


def naive_layer_frames(layer_matrices, filepath):
    """
        Description:

        Inputs:
            frame_image_matrices: a string that contains a filepath to an image/video
            of size f x m x n x 3
            f is each frame,
            m is width and n is height of image, 3 is the RGB dimensions
            if it is an image, f will simply be length 1

            layer_matrices: A 3 dimensional np matrix of size f x m x n
            f is each frame,
            m is width and n is height of image.
            Contains matrices of 1s and 0s


        Outputs:
            a f x m x n x 3 matrix that represents the image sequence of the specific layer

    """

    images = pims.open(filepath)

    frame_count = len(images) 

    output_frames = np.empty(
        (layer_matrices.shape[0], layer_matrices.shape[1], layer_matrices.shape[2], 3))

    for i in range(len(layer_matrices)):  # for each frame
        if frame_count == 1:  # if there is only 1 frame == just an image, so keep using only that
            image_frame = images[0]  # TODO: optimize this?
        elif i < frame_count:  # else, its a video, so check that secondary input has not fallen short
            image_frame = images[i]
        else:  # if it has, then just freeze frame last frame of secondary input
            image_frame = images[-1]

        frame_layer_matrix = layer_matrices[i]
        image_frame = resize(image_frame, (frame_layer_matrix.shape[0], frame_layer_matrix.shape[1]))
        processed_matrix = get_processed_matrix(frame_layer_matrix, image_frame)
        output_frames[i] = processed_matrix

    return output_frames


def layer_replace(layer_matrices, secondary_filepaths):
    """
    Description:

    Inputs:
    layer_matrices: an l x f x m x n np array the contains m x n matrices of 1s and 0s for each layer
                    of each frame

    secondary_filepaths: a 1-D python list of strings that contain the filepaths to all final images/videos
                         for layer replacement
    Outputs:
    An f x m x n x 3 image sequence that represents the final video

    """

    # produce a 5 dimensional l x f x m x n x 3 np array that will contain the final video data
    final_video = np.empty((layer_matrices.shape[0], layer_matrices.shape[1], layer_matrices.shape[2],
                            layer_matrices.shape[3], 3))

    for layer_index, each_layer in enumerate(layer_matrices):  # for each layer

        layer_sequence = naive_layer_frames(each_layer, secondary_filepaths[layer_index])
        final_video[layer_index] = layer_sequence

    # compress final video to a 4 dimensional array that is a final sequence of images
    final_video = np.sum(final_video, axis=0)

    return final_video
