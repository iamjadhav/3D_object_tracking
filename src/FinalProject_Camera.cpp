#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <boost/circular_buffer.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"
// #include <matplot/matplot.h>  // works
#include <matplotlibcpp.h>

using namespace std;
namespace plt = matplotlibcpp;

// write ttcs to files
void writeValuesToFile(const std::vector<double>& lidar, const std::vector<double>& camera, std::string params) {
	// Open the file for writing in append mode
	std::ofstream outputFile("../output_values_new.txt", std::ios::app);
	std::cout << std::fixed;
	std::cout << std::setprecision(2);

	// Check if the file is opened successfully
	if (outputFile.is_open()) {
		// Write the values to the file for category 1
		if (outputFile.tellp() == 0) {
			outputFile << std::left << std::setw(7) << "Image" << std::setw(15) << "Lidar" << std::setw(7) << "Camera_" + params + "\n\n";
		}

		for (size_t i = 0; i < std::max(lidar.size(), camera.size()); ++i) {
			outputFile << std::setw(7) << i + 1 << std::fixed << std::setprecision(2);;

			if (i < lidar.size()) {
				outputFile << std::setw(15) << lidar[i];
			}
			else {
				outputFile << std::setw(15) << " ";
			}

			if (i < camera.size()) {
				outputFile << std::setw(15) << camera[i];
			}
			outputFile << "\n";
		}
		outputFile << "\n";

		// Close the file
		outputFile.close();
		std::cout << "Values appended to output_values.txt successfully." << std::endl;
	}
	else {
		std::cerr << "Unable to open the file for writing." << std::endl;
	}
}


// MAIN PROGRAM //
int main(int argc, const char* argv[]) {

	// Args
	std::cout << "\n<<<--- Final Project --->>>" << std::endl;
	//std::cout << argc << " " << argv[4] << std::endl;

	// argv[0] -> exe path
	// argv[1] -> Detector type; MUST be one of modernDets
	// argv[2] -> Descriptor type; MUST be one of modernDescrs
	// argv[3] -> Matcher type; MUST be one of MAT_BF or MAT_FLANN
	// argv[4] -> Selector type: MUST be one of SEL_NN or SEL_KNN
	std::string detectorType, descriptorType, matcherType, selectorType;

	std::vector<std::string> modernDets = { "SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT" };
	std::vector<std::string> modernDescrs = { "BRISK", "BRIEF", "ORB", "AKAZE", "FREAK", "SIFT" };
	// args check
	try {
		if ((argc < 2) || (argc != 5)) {
			throw std::logic_error("\nNo/Insufficient arguments passed !!");
		}
		std::cout << "\nAll arguments passed !!" << std::endl;
		detectorType = static_cast<std::string>(argv[1]);
		descriptorType = static_cast<std::string>(argv[2]);
		matcherType = static_cast<std::string>(argv[3]);
		selectorType = static_cast<std::string>(argv[4]);
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	if (std::find(modernDets.begin(), modernDets.end(), detectorType) != modernDets.end()) {
		std::cout << "\nDetector Type is valid." << std::endl;
	}
	else {
		std::cout << "\nInvalid Detector Type !! Using default detector..." << std::endl;
		detectorType = "FAST";
	}
	if (std::find(modernDescrs.begin(), modernDescrs.end(), descriptorType) != modernDescrs.end()) {
		std::cout << "Descriptor Type is valid." << std::endl;
	}
	else {
		std::cout << "Invalid Descriptor Type !! Using default descriptor..." << std::endl;
		descriptorType = "BRIEF";
	}
	if (!(matcherType.compare("MAT_BF") == 0) && !(matcherType.compare("MAT_FLANN") == 0)) {
		std::cout << "Invalid Matcher Type !! Using default matcher..." << std::endl;
		matcherType = "MAT_BF";
	}
	else {
		std::cout << "Matcher Type is valid." << std::endl;
	}
	if (!(selectorType.compare("SEL_NN") == 0) && !(selectorType.compare("SEL_KNN") == 0)) {
		std::cout << "Invalid Selector Type !! Using default selector..." << std::endl;
		selectorType = "SEL_KNN";
	}
	else {
		std::cout << "Selector Type is valid." << std::endl;
	}

	// Disable logging and set precision
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
	std::cout << std::fixed;
	std::cout << std::setprecision(3);

	// INIT VARIABLES AND DATA STRUCTURES //

	// data location
	std::string dataPath = "../";

	// camera
	std::string imgBasePath = dataPath + "images/";
	std::string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
	std::string imgFileType = ".png";
	int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
	int imgEndIndex = 77;   // last file index to load
	int imgStepWidth = 1;
	int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

	// object detection
	std::string yoloBasePath = dataPath + "dat/yolo/";
	std::string yoloClassesFile = yoloBasePath + "coco.names";
	std::string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
	std::string yoloModelWeights = yoloBasePath + "yolov3.weights";

	// Lidar
	std::string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
	std::string lidarFileType = ".bin";

	// calibration data for camera and lidar
	cv::Mat P_rect_00(3, 4, cv::DataType<double>::type); // 3x4 projection matrix after rectification
	cv::Mat R_rect_00(4, 4, cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
	cv::Mat RT(4, 4, cv::DataType<double>::type); // rotation matrix and translation std::vector

	RT.at<double>(0, 0) = 7.533745e-03; RT.at<double>(0, 1) = -9.999714e-01; RT.at<double>(0, 2) = -6.166020e-04; RT.at<double>(0, 3) = -4.069766e-03;
	RT.at<double>(1, 0) = 1.480249e-02; RT.at<double>(1, 1) = 7.280733e-04; RT.at<double>(1, 2) = -9.998902e-01; RT.at<double>(1, 3) = -7.631618e-02;
	RT.at<double>(2, 0) = 9.998621e-01; RT.at<double>(2, 1) = 7.523790e-03; RT.at<double>(2, 2) = 1.480755e-02; RT.at<double>(2, 3) = -2.717806e-01;
	RT.at<double>(3, 0) = 0.0; RT.at<double>(3, 1) = 0.0; RT.at<double>(3, 2) = 0.0; RT.at<double>(3, 3) = 1.0;

	R_rect_00.at<double>(0, 0) = 9.999239e-01; R_rect_00.at<double>(0, 1) = 9.837760e-03; R_rect_00.at<double>(0, 2) = -7.445048e-03; R_rect_00.at<double>(0, 3) = 0.0;
	R_rect_00.at<double>(1, 0) = -9.869795e-03; R_rect_00.at<double>(1, 1) = 9.999421e-01; R_rect_00.at<double>(1, 2) = -4.278459e-03; R_rect_00.at<double>(1, 3) = 0.0;
	R_rect_00.at<double>(2, 0) = 7.402527e-03; R_rect_00.at<double>(2, 1) = 4.351614e-03; R_rect_00.at<double>(2, 2) = 9.999631e-01; R_rect_00.at<double>(2, 3) = 0.0;
	R_rect_00.at<double>(3, 0) = 0; R_rect_00.at<double>(3, 1) = 0; R_rect_00.at<double>(3, 2) = 0; R_rect_00.at<double>(3, 3) = 1;

	P_rect_00.at<double>(0, 0) = 7.215377e+02; P_rect_00.at<double>(0, 1) = 0.000000e+00; P_rect_00.at<double>(0, 2) = 6.095593e+02; P_rect_00.at<double>(0, 3) = 0.000000e+00;
	P_rect_00.at<double>(1, 0) = 0.000000e+00; P_rect_00.at<double>(1, 1) = 7.215377e+02; P_rect_00.at<double>(1, 2) = 1.728540e+02; P_rect_00.at<double>(1, 3) = 0.000000e+00;
	P_rect_00.at<double>(2, 0) = 0.000000e+00; P_rect_00.at<double>(2, 1) = 0.000000e+00; P_rect_00.at<double>(2, 2) = 1.000000e+00; P_rect_00.at<double>(2, 3) = 0.000000e+00;

	// misc
	double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
	int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
	//std::vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
	boost::circular_buffer<DataFrame> dataBuffer(dataBufferSize);
	bool bVis = false;            // visualize results
	std::vector<double> ttcs_lidar, ttcs_camera;  // visualize in a graph

	// MAIN LOOP OVER ALL IMAGES //

	for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex += imgStepWidth) {
		// LOAD IMAGE INTO BUFFER //

		// assemble filenames for current index
		std::ostringstream imgNumber;
		imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
		std::string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

		// load image from file 
		cv::Mat img = cv::imread(imgFullFilename);

		// push image into data frame buffer
		DataFrame frame;
		frame.cameraImg = img;
		dataBuffer.push_back(frame);
		//std::cout << "\nData buffer size: " << dataBuffer.size() << " \n" << std::endl;

		//std::cout << "#1 : LOAD IMAGE INTO BUFFER done" << std::endl;


		// DETECT & CLASSIFY OBJECTS //

		float confThreshold = 0.2;
		float nmsThreshold = 0.4;
		detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
					  yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

		// // helful to debug yolo carBB and lidar carBB boxId matches
		//std::cout << "\nImage: " << imgIndex << " bb boxid,tl,br: " << std::endl;
		//for (auto it = ( dataBuffer.end() - 1 )->boundingBoxes.begin(); it != ( dataBuffer.end() - 1 )->boundingBoxes.end(); ++it) {
		//	std::cout << '\t' << it->boxID << '\t' << it->roi.tl() << '\t' << it->roi.br() << std::endl;
		//}

		//std::cout << "#2 : DETECT & CLASSIFY OBJECTS done" << std::endl;

		// CROP LIDAR POINTS //

		// load 3D Lidar points from file
		std::string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
		std::vector<LidarPoint> lidarPoints;
		loadLidarFromFile(lidarPoints, lidarFullFilename);

		// remove Lidar points based on distance properties
		// assuming level road surface
		float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
		cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);

		(dataBuffer.end() - 1)->lidarPoints = lidarPoints;

		//std::cout << "#3 : CROP LIDAR POINTS done" << std::endl;


		// CLUSTER LIDAR POINT CLOUD //

		// associate Lidar points with camera-based ROI
		float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
		clusterLidarWithROI((dataBuffer.end() - 1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

		// Visualize 3D objects
		bVis = false;
		if (bVis) {
			show3DObjects((dataBuffer.end() - 1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
		}
		bVis = false;

		//std::cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << std::endl;

		//continue; // skips directly to the next image without processing what comes beneath

		// DETECT IMAGE KEYPOINTS //

		// convert current image to grayscale
		cv::Mat imgGray;
		cv::cvtColor((dataBuffer.end() - 1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

		// extract 2D keypoints from current image
		std::vector<cv::KeyPoint> keypoints; // create empty feature list for current image
		//std::string detectorType = "SHITOMASI";

		if (detectorType.compare("SHITOMASI") == 0) {
			detKeypointsShiTomasi(keypoints, imgGray, false);
		}
		else if (detectorType.compare("HARRIS") == 0) {
			detKeypointsHarris(keypoints, imgGray, false);
		}
		else {
			detKeypointsModern(keypoints, imgGray, detectorType, false);
		}

		// optional : limit number of keypoints (helpful for debugging and learning); not helpful as we already have bounding boxes from YOLO
		bool bLimitKpts = false;
		if (bLimitKpts) {
			int maxKeypoints = 50;

			if (detectorType.compare("SHITOMASI") == 0) { // there is no response info, so keep the first 50 as they are sorted in descending quality order
				keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
			}
			cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
			std::cout << " NOTE: Keypoints have been limited!" << std::endl;
		}

		// push keypoints and descriptor for current frame to end of data buffer
		(dataBuffer.end() - 1)->keypoints = keypoints;

		//std::cout << "#5 : DETECT KEYPOINTS done" << std::endl;


		// EXTRACT KEYPOINT DESCRIPTORS //

		cv::Mat descriptors;
		//std::string descriptorType = "BRISK"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT

		if ((descriptorType.compare("AKAZE") == 0) && !(detectorType.compare("AKAZE") == 0)) {
			std::cerr << "\n!!! AKAZE descriptor can only be used with AKAZE detector !!!" << std::endl;
			std::cout << "\n!!! Setting detector type to FAST !!!" << std::endl;
			detectorType = "FAST";
		}
		else if ((descriptorType.compare("ORB") == 0) && (detectorType.compare("SIFT") == 0)) {
			std::cerr << "\n!!! ORB descriptor does not work with SIFT detector !!!" << std::endl;
			std::cout << "\n!!! Setting detector type to FAST !!!" << std::endl;
			detectorType = "FAST";
		}

		descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);

		// push descriptors for current frame to end of data buffer
		(dataBuffer.end() - 1)->descriptors = descriptors;

		//std::cout << "#6 : EXTRACT DESCRIPTORS done" << std::endl;


		if (dataBuffer.size() > 1) // wait until at least two images have been processed
		{

			// MATCH KEYPOINT DESCRIPTORS //

			std::vector<cv::DMatch> matches;
			//std::string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
			std::string descriptorType2 = descriptorType.compare("SIFT") == 0 ? "DES_HOG" : "DES_BINARY";
			//std::string descriptorType2 = "DES_BINARY"; // DES_BINARY, DES_HOG
			//std::string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN

			matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
							 (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
							 matches, descriptorType2, matcherType, selectorType);

			// store matches in current data frame
			(dataBuffer.end() - 1)->kptMatches = matches;
			//std::cout << "kptmatches size: " << matches.size() << std::endl;  // around 3k

			//std::cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << std::endl;


			// TRACK 3D OBJECT BOUNDING BOXES //

			std::map<int, int> bbBestMatches;
			matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end() - 2), *(dataBuffer.end() - 1)); // associate bounding boxes between current and previous frame using keypoint matches

			// store matches in current data frame
			(dataBuffer.end() - 1)->bbMatches = bbBestMatches;

			//std::cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << std::endl;
			//continue;
			std::cout << "\n3D Object matching done between Previous Image: " << imgIndex - 1 << " <<-->> Current Image: " << imgIndex << "\n" << std::endl;

			// COMPUTE TTC ON OBJECT IN FRONT //

			// loop over all BB match pairs
			for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1) {
				// find bounding boxes associates with current match
				BoundingBox* prevBB = nullptr, * currBB = nullptr;
				for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2) {
					if (it1->second == it2->boxID) // check whether current match partner corresponds to this BB
					{
						currBB = &(*it2);
						//std::cout << "Current BB boxId: " << it2->boxID << '\t';
						break;
					}
					//else { std::cout << "No match for current BB" << std::endl; }
				}

				for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2) {
					if (it1->first == it2->boxID) // check whether previous match partner corresponds to this BB
					{
						prevBB = &(*it2);
						//std::cout << "Previous BB boxId: " << it2->boxID;
						break;
					}
					//else { std::cout << "No match for previous BB" << std::endl; }
				}

				//if (currBB != nullptr && prevBB != nullptr) {
				//	std::cout << " | Current BB Lidar Points: " << currBB->lidarPoints.size() << " & " <<
				//		"Previous BB Lidar Points : " << prevBB->lidarPoints.size() << std::endl;
				//}

				//continue;

				// compute TTC for current match
				if (currBB != nullptr && prevBB != nullptr) {
					if (currBB->lidarPoints.size() > 20 && prevBB->lidarPoints.size() > 20) {   // only compute TTC if we have enough Lidar points

						double ttcLidar;

						//std::cout << "Current BB Lidar Points: " << currBB->lidarPoints.size() << " & " <<
						//	"Previous BB Lidar Points : " << prevBB->lidarPoints.size() << std::endl;

						computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar, ttcs_lidar);

						double ttcCamera;
						clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);
						computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera, ttcs_camera);

						bVis = false;
						if (bVis) {
							cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
							showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
							cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);

							char str[200];
							sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
							putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255));

							std::string windowName = "Final Results : TTC";
							cv::namedWindow(windowName, 4);
							cv::imshow(windowName, visImg);
							std::cout << "Press key to continue to next frame" << std::endl;
							cv::waitKey(0);
						}
						bVis = false;
					}
				}  // eof TTC computation
			}  // eof loop over all BB matches            
		}  //eof 3D object matching
	} // eof loop over all images

	 //continue;

	// data for performance evaluation
	///*
	std::string params = static_cast<std::string>(argv[1]) + "_" + static_cast<std::string>(argv[2]);
	writeValuesToFile(ttcs_lidar, ttcs_camera, params);
	//*/
  
	// visualization
	std::vector<double> images;
	for (int i = imgStartIndex; i < imgEndIndex - 1; ++i) {
      images.push_back(i);
    }

    plt::plot(images, ttcs_lidar, "g");
	plt::plot(images, ttcs_camera, "r");

	plt::title("Time To Collision: Lidar Vs Camera : ");
	plt::xlabel("frames");
	plt::ylabel("ttc (s)");
	plt::legend();
	
	// plt::show();
	plt::save("../output.png");
	
	/*
	// visualization -> needs CMake > 3.13
	std::vector<double> images = linspace(0, imgEndIndex - imgStartIndex, imgEndIndex - imgStartIndex);

	auto plot_lidar = plot(images, ttcs_lidar, "g--o");
	hold(on);
	auto plot_cam = plot(images, ttcs_camera, "r--o");
	title("Time To Collision: Lidar Vs Camera : " + params);
	xlabel("frames");
	ylabel("ttc (s)");
	::matplot::legend({ plot_lidar, plot_cam }, { "Lidar", "Camera" });

	show();
	 */

	return 0;
}
