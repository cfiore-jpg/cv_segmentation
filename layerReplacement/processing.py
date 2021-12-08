import numpy as np

def make_layer_matrices(semantic_output):

     '''
    Description:


    Inputs:
        semantic_output: an m x n np matrix array that contains layer numbers 

    Outputs:
        a 3 dimensional l x m x n np array that contains 1s and 0s

    '''

    unique_layers = np.unique(semantic_output)

    amount_layers = unique_layers.shape[0]



    for layer_number in unique_layers:

