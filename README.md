# SCAN: Semantic Clustering by Adopting Nearest neighbors - Reproducibility Test

The SCAN approach is published in the paper [SCAN: Learning to Classify Images without Labels](https://arxiv.org/abs/2005.12320)

SCAN is an unsupervised approach for image classification introduced by Van Gansbeke, et al. It has two main steps followed by a third fine-tuning step:
1. Pretext Step: Feature representations are learned through a pretext task (SimCLR is used here), and then the nearest neighbors of each image are mined according to feature similarity.
2. SCAN Semantic Clustering Step: The semantically meaningful nearest neighbors found in the pretext step are used as a prior for semantic clustering. An image and and its nearest neighbors are classified together using the novel SCAN loss function. 
3. Self-labeling Step: Classification errors caused by neighbors with less-confident predictions are corrected by applying a threshold to filter them out and the network is updated using a weighted cross entropy loss.

This repo contains a reimplementation of SCAN for a reproduction study aimed to reproduce performance results on CIFAR10 to within the range of the published results. Parts of the original implementation are reused. 

Setup Instructions:
Open the SCAN_repro Colab notebook and run the sections in order. Connect the notebook to GPU and mount to Google drive to store datasets and file. The notebook imports the setup files provided here containing required functions. Run the steps in order. Alternatively, if previous model weights are available, each step in the approach can be run independently.

This SCAN reimplementation achieved classification accuracy of 87.2% on CIFAR10.

