# cv_segmentation


## Quick Start  

1.  Make sure you have installed ```flask```, ```tensorflow2```, and ```opencv``` through:  

```
pip install Flask  
pip install tensorflow  
pip install opencv-python  
```
2. Make sure that the pretrained model weights are located in the backend folder. The weights can be downloaded from: https://github.com/Runist/SegNet-keras/releases/download/v0.1/segnet_weights.h5.
```

3.  Running Frontend GUI. Make sure ```Flask``` is installed locally. In the cv_segmentation (root) folder, run the following three commands:  
  1.) export FLASK_APP=server  
  2.) export FLASK_ENV=development  
  3.) flask run  
The frontend module will automatically load the trained model.  

## Retrain the SegNet Model  

All the scripts related to SegNet model training are in the ```backend``` directory. We have a trained model that can be shared after requests. 
But one can still re-train the model as follows:  

1.  Run ```python backend/ExtractData.py``` to download dataset for training.  
2.  Run ```python backend/train.py```. This will train the model for 80 epochs and save the best model weights as ```backend/best_weights.h5```.
