import numpy as np
from skimage.transform import resize
import pims

#TODO:CONSIDER TO (IF AN INPUT IS GREYSCALE) TO JUST CHANGE TO RGB FORMAT
#TODO: i think we shoudl just assume (and hence modify to fit) an rgb format for all
#images and videos




def get_processed_matrix(image_matrix, layer_matrix):
    '''
    Description:

    Inputs:
        image_matrix: An m x n x 3 numpy array where m and n are the dimensions of
        the image.

        layer_matrix: An m x n numpy matrix. Only contains 0s and 1s

    Outputs:
        An m x n x 3. Will have values identical to image_matrix, but only where
        layer_matrix had 1s in the same spot.

    '''
    #TODO:here assuming that image matrix is rgb, be extensible and check if grey or RGBA\
    #TODO:return in similar format? -- hmmmm might fuck with addition at end of process(?)
    processed = image_matrix
    processed[:][:][0] = np.matmul(layer_matrix, image_matrix[:][:][0])
    processed[:][:][1] = np.matmul(layer_matrix, image_matrix[:][:][1])
    processed[:][:][2] = np.matmul(layer_matrix, image_matrix[:][:][2])
    return processed

def naive_image(image_matrices, layer_matrices):
    '''
    Description:

    Inputs:
        layer_matrices: A 3 dimensional np array of size l x m x n, the highest
        dimension (l) being the number of layers/labels obtained from semantic
        segmentation. Each matrix (m x n) is a matrix of 1s and 0s

        #TODO: figure out this next parameter, basically maybe a list of different
        secondary inputs (null | m x n x ? image | m x n x ? x f video)

        image_matrices: a 4 dimensional np array of size l x m x n x 3. It contains the
        logic of what image will replace the given layer. Resizing assumed to have occured eariler


    Outputs:
        a m x n x 3 matrix that contains values of final image
    '''

    #TODO: of size l x m x n x 3
    processed_matrices = np.empty((image_matrices.shape))

    for i in range(layer_matrices.shape[0]): #loop through layers -- main loop


        #first retrieve specific image and layer
        layer_matrix = layer_matrices[i]
        image_matrix = image_matrices[i]

        #then send to processing method
        processed_matrix = get_processed_matrix(image_matrix, layer_matrix)

        #and add to processed matrices array
        processed_matrices[i] = processed_matrix

    #unsure if proper axis, i think its right TODO: confirm this
    summed_image = np.sum(processed_matrices, axis=0)

    return summed_image

#TODO:review written code, maybe start extending over to multiple frames
#"naive_image" could get called for each frame in a video, -- if sequence is only
#one frame (basically just an image), then just return one summed image

def naive_video(frame_image_matrices, frame_layer_matrices):
    '''
        Description:

        Inputs:
            frame_image_matrices: A 5 dimensional np matrix of size f x l x m x n x 3
            f is each frame, l is the amount of layers,
            m is width and n is height of image, 3 is the RGB dimensions

            layer_matrices: A 4 dimensional np matrix of size f x l x m x n
            f is each frame, l is the amount of layers,
            m is width and n is height of image. 
            Contains matrices of 1s and 0s

            
        Outputs:
            a f x m x n x 3 matrix that represents the image sequence of the completed
            video

    '''

    output_frames = np.empty((frame_image_matrices.shape[0], frame_image_matrices.shape[2], frame_image_matrices.shape[3], 3))

    for i in range(frame_layer_matrices.shape[0]): #for every frame
        #naive_image gets called here preferably
        frame = naive_image(frame_image_matrices[i], frame_layer_matrices[i])
        output_frames[i] = frame

    return output_frames

def pre_layerReplace():
    '''
    Description:
    Method called immediately after front end receives confirmation that all the
    secondary inputs are ready.
    Transforms inputs obtained from front-end into proper inputs for back-end
    (such as, resizing, converting videos to image sequences, maybe more)
    to then call naive_video with modified and organized parameters

    Inputs:


    Outputs:

    '''




    #TODO: need to start implementing with PyAV and creating a front end
    #And maybe also PIMS? -- probablty also PIMS actually
