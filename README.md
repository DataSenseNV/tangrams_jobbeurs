[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)


# TangrIAm Project
#### Real-time tangram form detection from a live video stream.

The project is a partnership between Exploradôme museum, OCTO Technology and Microsoft and it aims to introduce the concept and application of artificial intelligence to young children. The specific application developed for the project is to apply object detection to live tangram solving.

A tangram is a dissection puzzle consisting of seven flat polygons (5 triangles, 1 square and 1 parallelogram) which are combined to obtain a specific shape. The objective is to replicate a pattern (given only an outline) using all seven pieces without overlap.

The goal of the game of tangram is to replicate the outline shown on the target card using a set of seven individual polygons (triangles, square or parallelogram). 
Tangram can be played either by multiple teams at the same time, racing to be the first to finish the outline. 

This project's objective is to detect the similarity between the shape drawn by each player at any point in time and the twelve possible target classes. 
The predictions are to be in real-time from a live video feed of the game board.

Within the framework of the project, 12 tangram selected shapes act as classes for the classifier:

- Boat (Bateau)
- Bow (Bol)
- Bridge (Pont)
- Cat (Chat)
- Fox (Renard)
- Hammer (Marteau)
- Heart (Coeur)
- House (Maison)
- Mountain (Montagne)
- Rabbit (Lapin)
- Swan (Cygne)
- Turtle (Tortue)


## Installation

```shell
git clone https://github.com/k13var/exploradome_tangram/
cd exploradome_tangram
git checkout numpy---team-4
python -m venv venv/
source venv/Scripts/activate # Windows
source venv/bin/activate # Mac
pip install -r requirements.txt
```

## Execution


### With an external webcam
```shell
# connect an external webcam and launch the script, specifying the board side of interest (in this example right)
python3 main.py --mode 1 --side right
```

### With a video saved locally
```shell
# launch the script using a video saved locally, specifying the board side of interest (in this example left)
python3 main.py --mode /videos/coeur.pm4 --side left
```

### With an image
```shell
# launch the app with an image on the right side, specifying the board side of interest (in this example right)
python3 main.py --mode /videos/coeur.jpg --side right
```

## Approach & Objective

Our approach has been to avoid the use of ML and DL techniques and leverage OpenCV's native capabilities to detect the individual shapes on the board and estimate their likeness to each of the twelve classes, using distances between the shapes' centroids.

## Image processing steps

- Split the video feed into two halves (left and right side)
- Identify all the edges on the board
- Keep only the contours corresponding to regular geometric shapes (triangle, square, parallelogram).
- Compute centroids of each geometric shape.
- Calculate distances between each pair of geometric shapes (resulting in 21 data points if all pieces are on the board).
- Compare this distance scorecard to the distance scorecard of each of the 12 target outlines (using an RMSE distance metric over the 21 distance readings)
- Transform RMSE into a probability distribution.

### Pre-processing

 1) Our original image

<p align="center"><img width=30% src="https://github.com/Nohossat/exploradome_tangram/blob/numpy---team-4/data/imgs/original.JPG"></p>

 2) Detect edges
 
<p align="center"><img width=30% src="https://github.com/Nohossat/exploradome_tangram/blob/numpy---team-4/data/imgs/canny_edge.JPG"></p>

3) Find geometric shapes 
 
 <p align="center"><img width=30% src="https://github.com/Nohossat/exploradome_tangram/blob/numpy---team-4/data/imgs/Shapes_only.JPG"></p>


## Classification results

Frames per seconds : 40-50 fps


### Prediction results in an ideal testing environment
The below results are valid for "final" predictions, ie the class predicted by the program once the correct form was finalized. 

<p align="center"><img width=60% src="https://github.com/Nohossat/exploradome_tangram/blob/numpy---team-4/data/imgs/confusion_matrix_clean_dataset.png")</p>


<p align="center"><img width=60% src="https://github.com/Nohossat/exploradome_tangram/blob/numpy---team-4/data/imgs/metrics_clean_test.png")</p>


### Prediction results in a more challenging testing environment
Results below were obtained using a more challenging testing dataset (incomplete form, slight mistake in the final form, objects between the camera and the board, challenging light conditions...) and are probably closer to the expected production environment.

<p align="center"><img width=60% src="https://github.com/Nohossat/exploradome_tangram/blob/numpy---team-4/data/imgs/confusion_matrix.png")</p>


<p align="center"><img width=60% src="https://github.com/Nohossat/exploradome_tangram/blob/numpy---team-4/data/imgs/classification_metrics.png")</p>

## Project metainformation

#### Dataset

Our approach, since it does not rely on machine/deep learning, does not require extensive training. A single imgage per class was sufficient to create the twelve benchmarks against which each form is compared.

Our data collection was hence designed to obtain a test set of images that would be as diverse as possible (in terms of orientation, image obstruction, light conditions...) to ensure maximize robustness in our software. For testing purposes, the dataset was enriched with changed lighting conditions (both brighter and darker).

#### Attempts and challenges

* A key issue has been the robustness of the preprocessing pipeline. Originally, the test dataset was too narrow to properly capture variations in light conditions.   
In early attempts, our preprocessing (clear isolation of the tangrams on the board), worked well for similar conditions with a subset of images similar to those used for calibration; however, it did not generalize well with other environments (different table orientation, light, shadows...).
To that end, changing the preprocessing pipeline (from a binary threshold to canny edge detection) yielded much more robust results.

* Our first approach was to use Hu Moments to compare the player's shape to each of the 12 classes,  while benefiting from  Hu Moments' invariance to rotation, scale and position. Hu Moments are a set of 7 non-human-interpretable features which can be used to characterize any shape. Their list can be found [here](https://en.wikipedia.org/wiki/Image_moment#Moment_invariants).

   However, this approach turned out to give poor predictions for shapes which are in the process of construction. Another problem was an overly sensitive reaction to camera obstruction (when a player move a piece of tangram, he can obstruct the camera while doing it) which motivates our switch to more robust data points taken from piece centroids' relative distance to each other.

   The results from this approach highlight how the Hu Moments approach results in near-perfect predictions in ideal conditions but does not generalize well as the absence or presence of a single geometrice form completely changes their values and hence the program's ability to match it with the appropriate target class.

<p align="center"> Clean dataset results with the Hu Moments approach </p>
<p align="center"><img width=60% src="https://github.com/Nohossat/exploradome_tangram/blob/numpy---team-4/data/imgs/hu_moments_metrics_clean_dataset.png")</p>
 
<p align="center"> Challenging dataset results with the Hu Moments approach</p>
<p align="center"><img width=60% src="https://github.com/Nohossat/exploradome_tangram/blob/numpy---team-4/data/imgs/hu_moments_mixed_dataset.png")</p>



# Team

[Nohossat](https://github.com/Nohossat)  
[Bastien](https://github.com/BasCR-hub)  
[Nicolas](https://github.com/nicolaszys)  
[Renata](https://github.com/renatakaczor)  
[Gauthier](https://github.com/yanggautier)  


