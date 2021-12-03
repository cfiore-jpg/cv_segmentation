'''
Description:
    Extract the uniform data for training and testing.
'''
import numpy as np
import skimage.io
import scipy.io
import os



def selectSameShapeDataID(img_path):
    imgs_filename = os.listdir(img_path)
    imgs_filename = [each for each in imgs_filename if "2007_" not in each]
    res_list = []
    for f in imgs_filename:
        img_data =  skimage.io.imread("{}/{}".format(img_path, f))
        if img_data.shape[0] == 442 and img_data.shape[1] == 500:
            res_list.append(f.split(".")[0])
    print("The number of imges with the same shape : {}".format(len(res_list)))
    np.save("./same_shape_img_id.npy", res_list)


def extractData(img_path, annotation_path, selected_img_id):
    selected_img_id = list(np.load(selected_img_id, allow_pickle=True))
    # parse filenames
    annot_filename = os.listdir(annotation_path)
    annot_filename = [each for each in annot_filename]
    annot_id = [each.split(".")[0] for each in annot_filename]

    imgs_filename = os.listdir(img_path)
    imgs_filename = [each for each in imgs_filename]
    img_id = [each.split(".")[0] for each in imgs_filename]

    selected_img_id = [each for each in selected_img_id if "2007_" not in each and each in annot_id and each in img_id]
    print("{} images to be loaded...".format(len(selected_img_id)))

    imgs_filename = ["{}.jpg".format(each) for each in selected_img_id]
    annot_filename = ["{}.mat".format(each) for each in selected_img_id]

    # images loading
    img_data = np.asarray([
        skimage.io.imread("{}/{}".format(img_path, each))
        for each in imgs_filename
    ])
    # annotation loading
    annot_data = np.asarray([
        scipy.io.loadmat("{}/{}".format(annotation_path, each))["LabelMap"]
        for each in annot_filename
    ])
    return img_data, annot_data


if __name__ == '__main__':
    img_path = "../data/dataset/VOC2010/JPEGImages/"
    selectSameShapeDataID(img_path)
    # ------------------------------------
    annotation_path = "../data/annotations/"
    selected_img_id = "./same_shape_img_id.npy"
    img_data, annot_data = extractData(img_path, annotation_path, selected_img_id)
    print("Img data shape : ", img_data.shape)
    print("Annot data shape : ", annot_data.shape)
    np.save("./selected_img_data.npy", img_data)
    np.save("./selected_annot_data.npy", annot_data)

