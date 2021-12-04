import numpy as np
from skimage.transform import resize

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

def naive_image(layer_matrices, image_matrices):
    '''
    Description:

    Inputs:
        layer_matrices: A 3 dimensional np array of size l x m x n, the highest 
        dimension (l) being the number of layers/labels obtained from semantic 
        segmentation. Each matrix (m x n) is a matrix of 1s and 0s
        
        #TODO: figure out this parameter, basically maybe a list of different 
        secondary inputs (null | m x n x ? image | m x n x ? x f video)

        image_matrices: a 4 dimensional np array of size l x m x n x 3. It contains the 
        logic of what image will replace the given layer. Resizing assumed to have occured eariler


    Outputs:
        a m x n x 3 matrix that contains values of final image
    '''

    #TODO: add another dimension?
    processed_matrices = np.empty((image_matrices.shape))

    for i in range(layer_matrices.shape[0]): #loop through layers -- main loop
        #TODO: because of the , even with video, it might be worth doing 
        # one layer at a time

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

def naive_video(layer_matrices): 
'''
    Description:

    Inputs:
        layer_matrices: An f x l x m x n 

    Outputs:

    '''

    for i in range(layer_matrices.shape[0]): #for every frame
        #naive_image gets called here preferably
    

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
    