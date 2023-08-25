# Football-Player-Tracking using YOLOv5 and ByteTrack

A computer vision based project that uses "YOLO" object detection series and "ByteTrack" tracking, this project aims at detecting football players, referees, goalkeepers and a football upon feeding an input clip. In this project, we make use of Roboflow's open source datasets to train, test and validate the model. Since YOLO is trained in the COCO model, classes such as 'Football Player','Football','Referee','GoalKeeper' cannot be identified by the COCO model. Hence, for training the custom model, we make use of pre-trained weights in a file named 'best.pt' that is downloaded during the process. The input clips are downloaded from Kaggle (DFL Bundesliga Data Shootout). The annotation of the dataset used to obtain the weights is done using Roboflow. This project makes use of GPU and cannot be run on CPU runtime. For reference, use NVIDIA System Management Interface to determine the GPU Specifications. In some cases, NVIDIA CUDA drivers installation could be necessary for the GPU. ByteTrack is used for tracking. All the relevant links are mentioned below in this document. 


#Requisites

* Notebook: Google Colab
* Operating System: Windows 10, macOS 10.13, or Linux (Ubuntu 18.04+).
* GPU Drivers: NVIDIA drivers (version 460.32.03 or higher).
* CUDA Toolkit: CUDA 11.2 (for NVIDIA GPUs) .
* GPU Support: Ensure that your GPU supports CUDA and/or OpenCL.
* Dependencies: Install the required libraries as specified in the requirements.txt files.


YOLOv5 GitHub Repository: https://github.com/ultralytics/yolov5
ByteTrack Repository: https://github.com/ifzhang/ByteTrack
Kaggle-DFL Bundesliga Data Shootout: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data
Roboflow Notebooks: https://github.com/roboflow/notebooks
NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
