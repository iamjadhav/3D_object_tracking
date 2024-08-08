# 3D Object Tracking

The 3D Object Tracking repository uses a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. 
Also, we need to know how to detect objects in an image using the YOLO deep-learning framework.
And finally, we need to know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at how this can be done!

<img src="images/course_code_structure.png" width="779" height="414" />

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
  * Install Git LFS before cloning this Repo.
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* Boost >= 1.74
  * Linux/Mac/Windows: Follow official instuctions.
* Matplot++ >= 1.0
  * Linux/Mac/Windows: Follow official instuctions.

## Basic Build Instructions: Linux

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking FAST BRIEF MAT_BF SEL_KNN`. (With default or custom arguments)

## Basic Build Instructions: Windows (VS19)

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake ..`
4. Run: 3D_object_tracking.exe (can change the arguments from launch.vs.json)

## Visualization

Time To Collision for all 78 datapoints

<img src="docs/graphs_entire_dataset/ttc_lidar_camera_FAST_BRISK.png" width="960" height="493" />


## Arguments

1. Detector Type
2. Descriptor Type
3. Matcher Type
4. Selector Type

E.g. `"args": [
        "FAST",
        "BRIEF",
        "MAT_BF",
        "SEL_KNN"
      ]`

## FP.1 - Match 3D Objects

I used a brute-force approach for this method. Making use of std::map, std::multimap, std::set, and std::multiset I matched the best bounding boxes between previous and current frames.
Then I assigned these matches to the best bounding box matches vector of the current frame which are used to fuse lidar points together with the matches.

## FP.2 - Compute Lidar-based TTC

To calculate the lidar based TTC values, I used the mean and median X values of both clouds and got unstable values. So I switched to the IQR rule to find outliers and remove them.
I implemented a IQR method to obtain lower and upper thresholds of the X values, then using these thresholds to remove the lower and upper outlier cloud points containing the 
outlier X values.

## FP.3 - Associate Keypoint Correspondences with Bounding Boxes

To associating keypoint correspondences to the bounding boxes which enclose them I used the `_Rect_.contains()` method to check for presence of the point in the bounding box. 
After that, I calculated the euclidean distances between points in the filtered matches. Then I employed the IQR rule again to find lower and upper outlier distances and use
those to remove the bounding box matches associated with those distances.

## FP.4 - Compute Camera-based TTC

Camera TTC computation is similar to the exercise in previous lessons, taking the distances between matches, calculation distance ratios, and meadian distances and then calculating the 
camera ttc with the formula.

## FP.5 - Performance Evaluation: 1

1. The very first "off" lidar TTC values appear between 5th and 9th image pairs, deviating from 7-29-9.
2. Similar variations are present throughout the 78 image pairs.
3. My rudientary understanding is that,
					#### min X val (Previous Cloud) - min X val (Current Cloud)	        <<<--->>>	        #### Lidar TTC value
					a. sufficiently large                                      	        <<<--->>>	 	a. decent readings
					b. very large                                              	        <<<--->>>	 	b. small readings
					c. very large                                              	        <<<--->>>	 	c. large readings
					d. negative                                                	        <<<--->>>	 	d. negative readings
4. These variations arise due to various disadvantages of the constant velocity model in practical scenarios, acceleration or deceleration (slight changes in velocity of the preceding or ego vehicle),
   non-uniform motion (of the preceding or the ego vehicle).
3. After image 45, the ego car essentially stands in traffic, which I think registers wildly reflective lidar readings resulting in large and negative lidar TTC values. Looking at the readings,
   the minimum X values of previous lidar point clouds are less than the current ones which leads to negative denominator in the TTC formula.
   For Camera TTC, the vehicle standing in traffic leads to exactly same point correspondences giving a median distance ratio of 1 and -inf TTC values.

## FP.6 - Performance Evaluation: 2

1. The first "off" camera TTC value appears after the 20th image pair, where it deviates by 2~3 units for the best performing detector-descriptor combination FAST-BRIEF (in terms of stable ttc values,
   speed, and performance in general) which is mostly because the preceding vehicle is coming to a hault with decreasing velocity.
2. After image 45, the ego car essentially stands in traffic, which leads to exactly same point correspondences giving a median distance ratio of 1 and -inf TTC values or large negative ones in case
   several other combinations.
3. FAST-FREAK, AKAZE-AKAZE combinations also work well in terms of reliable and stable ttc values.
4. Other combinations display off values once in a few pairs with also the -inf/huge values in traffic.
5. Any combinations involving a SIFT in either way, gives extremely unreliable and unrealistic values which makes sense given its gradient-based working.
6. In general, camera values appear to be a bit more stable than Lidar values in this Constant Velocity Model probably because the lidar values are very precise and produce sensitive TTC values.

## Documentation

1. For reference of the ttc values refer to the `output_values_new.txt` file in the workspace.
2. And for visual inspection, the graphs in `docs` directory would be helpful.