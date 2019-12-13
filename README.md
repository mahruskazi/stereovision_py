# Stereovision Py

Stereovision Py is a python based stereovision project that combines the benefits of stereopsis and object tracking into one simple to use solution. Here are some of the current features:

  - Camera calibration
  - Depthmap tuning
  - ORB object tracking

### Installation

Clone the repo and start a terminal session in the main project folder.
We first install the required dependencies; it is advisable to create a virtual environment first
```sh
$ pip3 install -r requirements.txt
```
### Usage

We first by setting the camera settings, open a file called camera_settings.txt in the folder called capture.
You can change which camera is opened by the program and what resolution we will be running.
```
left_camera = 0, right_camera = 1, height = 480, width = 640
```

Now you should be all set to run the tool!
```
$ python3 -m stereo_tool -h
usage: stereovision_py <command> [<args>]

                        Commands:
                        capture         Capture pairs of images to then be used for stereovision calibration
                        calibrate       Calibrate the cameras to create a rectified image pair
                        tune            Tune the stereovision depth map
                        tracker         Track a object in real time and measure the distance to it

A CLI to help calibrate stereovision, calculate distance from a depth map, and
track an object

positional arguments:
  command     Subcommand to run

optional arguments:
  -h, --help  show this help message and exit
```

To start calibration, we will need to first capture images

```
$ python3 -m stereo_tool capture -i 5
```
This will open up a left camera and right camera frame and then take images at the set interval (5 seconds). It is recommended to take around 50 image pairs with the chessboard pattern visible at different angles and orientations.
Once happy with the calibration set, we can start the calibration

```
$ python3 -m stereo_tool calibrate -p 50
```
This will try to find the chessboard corners in the image pairs, automatically skipping the 'bad' images. Press any key to move through the images, at the end the program will create a rectified pair that will be saved for reference.

Now we need to take a scene to tune the depth map on, place three objects spaced close and far from the camera
```
$ python3 -m stereo_tool tune --save_scene
```
Press 's' to save the scene or 'q' to quit

Once the scene is saved, we can tune the depth map
```
$ python3 -m stereo_tool tune
```
The idea is to play with the sliders until the closes object is shown in red while the further objects are a dark blue. Once you are happy with the outcome, save the settings. If the depth map is very noisy, it may benefit from running the calibration again with more image pairs.

Finally, we can start tracking
```
$ python3 -m stereo_tool tracker
```
It will provide a depth map and camera view for you to use. If you click on an object in the depth view, the distance to that object should be displayed. Additionally, by drawing a bounding box in the camera view around the object you want to track will produce an ORB tracking view.

Modify get_distance in tracker.py if the distance to the object is inaccurate

