# Training Yolo on LEGO

*This repository contains experiments of transfer learning using YOLO on a new synthetical LEGO data set*
*ROUGH AND UNDOCUMENTED!*
### Prerequisite
 * Tensorflow
### Organisation
 * _dumps_weights_to_pkl.py_ save tiny yolo weights obtained from *darkflow* to a pkl
 * _train.py_ use those weights to retrain the network defined in _tiny_yolo.py_
 * _load.py_ contains function to save and create Tensorflow graphs including a *freeze_graph(folder)* that makes a graph as a tensorflow checkpoint reusable
 * _test.py_ apply a 'frozen graph' to a test image

### References and Acknowledgements
This code is based on the foillowing paper
Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." arXiv preprint arXiv:1506.02640 (2015).
website : http://pjreddie.com/darknet/yolo/

Some python functions and are taken from or modified from : 
https://github.com/thtrieu/darkflow
