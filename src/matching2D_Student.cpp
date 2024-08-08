#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint>& kPtsSource, std::vector<cv::KeyPoint>& kPtsRef, cv::Mat& descSource, cv::Mat& descRef,
	std::vector<cv::DMatch>& matches, std::string descriptorType2, std::string matcherType, std::string selectorType) {
	// configure matcher
	bool crossCheck = selectorType.compare("SEL_NN") == 0 ? true : false;
	cv::Ptr<cv::DescriptorMatcher> matcher;

	if (matcherType.compare("MAT_BF") == 0) {
		int normType = descriptorType2.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
		matcher = cv::BFMatcher::create(normType, crossCheck);
		}
	else if (matcherType.compare("MAT_FLANN") == 0) {
		// OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
		if ((descSource.type() != CV_32F) || (descRef.type() != CV_32F)) {  // check both descriptor types
			descSource.convertTo(descSource, CV_32F);
			descRef.convertTo(descRef, CV_32F);
			}
		matcher = cv::FlannBasedMatcher::create();
		}

	// perform matching task
	if (selectorType.compare("SEL_NN") == 0) { // nearest neighbor (best match; k=1)
		matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
		}
	else if (selectorType.compare("SEL_KNN") == 0) { // k nearest neighbors (k=2)
		std::vector<std::vector<cv::DMatch>> knn_matches;
		matcher->knnMatch(descSource, descRef, knn_matches, 2);

		const float distance_threshold = 0.8f;
		for (int i = 0; i < knn_matches.size(); i++) {
			if ((knn_matches[i][0].distance / knn_matches[i][1].distance) < distance_threshold) {
				matches.push_back(knn_matches[i][0]);
				}
			}
		}

	//std::cout << matches.size() << " keypoints matched with " << matcherType << " and " << selectorType << " !!" << std::endl;
	}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, cv::Mat& descriptors, std::string descriptorType) {
	// select appropriate descriptor
	cv::Ptr<cv::DescriptorExtractor> extractor;
	if (descriptorType.compare("BRISK") == 0) {

		int threshold = 30;        // FAST/AGAST detection threshold score.
		int octaves = 3;           // detection octaves (use 0 to do single scale)
		float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

		extractor = cv::BRISK::create(threshold, octaves, patternScale);
		}
	else if (descriptorType.compare("BRIEF") == 0) {
		extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
		}
	else if (descriptorType.compare("SIFT") == 0) {
		// extractor = cv::SIFT::create();
		cv::Ptr<cv::xfeatures2d::SIFT> extractor = cv::xfeatures2d::SIFT::create();
		}
	else if (descriptorType.compare("ORB") == 0) {
		extractor = cv::ORB::create();
		}
	else if (descriptorType.compare("AKAZE") == 0) {
		extractor = cv::AKAZE::create();
		}
	else if (descriptorType.compare("FREAK") == 0) {
		extractor = cv::xfeatures2d::FREAK::create();
		}
	else {
		std::cerr << "\n!!! Invalid descriptor type!!!" << std::endl;
		}

	// perform feature description
	extractor->compute(img, keypoints, descriptors);
	}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, bool bVis) {
	// compute detector parameters based on image size
	int blockSize = 6;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
	double maxOverlap = 0.0; // max. permissible overlap between two features in %
	double minDistance = (1.0 - maxOverlap) * blockSize;
	int maxCorners = static_cast<double>((img.rows)) * static_cast<double>((img.cols)) / max(1.0, minDistance); // max. num. of keypoints

	double qualityLevel = 0.05; // minimal accepted quality of image corners
	double k = 0.04;

	// Apply corner detection
	//auto t = static_cast<double>(cv::getTickCount());
	std::vector<cv::Point2f> corners;
	cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

	// add corners to result vector
	for (auto it = corners.begin(); it != corners.end(); ++it) {
		cv::KeyPoint newKeyPoint;
		newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
		newKeyPoint.size = blockSize;
		keypoints.push_back(newKeyPoint);
		}
	//t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
	//std::cout << "Shi-Tomasi detection with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;

	// visualize results
	if (bVis) {
		cv::Mat visImage = img.clone();
		cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		string windowName = "Shi-Tomasi Corner Detector Results";
		cv::namedWindow(windowName, 6);
		imshow(windowName, visImage);
		cv::waitKey(0);
		}
	}


// Detect keypoints in image using the traditional Harris detector
void detKeypointsHarris(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, bool bVis) {
	// Detector parameters
	int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered; even
	int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd); to have a center pixel for NMS
	int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
	double k = 0.04;       // Harris parameter (see equation for details)

	// Detect Harris corners and normalize output
	cv::Mat dst, dst_norm, dst_norm_scaled;
	dst = img.clone();
	//auto t = static_cast<double>(cv::getTickCount());
	cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
	cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
	cv::convertScaleAbs(dst_norm, dst_norm_scaled);

	//std::vector<cv::KeyPoint> final_keypts;

	auto maximumOverlap = 0.0;
	// floor because the kernel will always be even-sized to have a center pixel
	int window_radius = std::floor(apertureSize / 2);  // increase overlap threshold for finer keypoints

	for (int row = window_radius; row <= dst_norm.rows - window_radius - 1; row++) {
		for (int col = window_radius; col <= dst_norm.cols - window_radius - 1; col++) {

			int harris_val = (int)dst_norm.at<float>(row, col);

			if (harris_val > minResponse) {
				float max_response = 0.0;
				for (int w_y = row - window_radius; w_y <= row + window_radius; w_y++) {
					for (int w_x = col - window_radius; w_x <= col + window_radius; w_x++) {
						// std::cout << "(" << w_y << "," << w_x << ") ";
						float w_val = (float)dst_norm.at<float>(w_y, w_x);
						if (max_response < w_val) {
							max_response = w_val;
							}
						else {
							// this is where the suppression happens within the window and in turn the image
							dst_norm.at<float>(w_y, w_x) = 0.0;
							}
						}
					}
				// move this before window for loops for better runtime
				if (dst_norm.at<float>(row, col) == max_response) {
					// std::cout << max_response << " ";
					cv::KeyPoint new_keypt;
					new_keypt.pt = cv::Point2f(col, row);
					new_keypt.response = max_response;
					new_keypt.size = 2 * apertureSize;
					keypoints.push_back(new_keypt);
					}
				}
			}
		}

	std::sort(keypoints.begin(), keypoints.end(),
		[](const cv::KeyPoint& a, const cv::KeyPoint& b) {
		return (a.pt.x < b.pt.x) || (a.pt.x == b.pt.x && a.pt.y < b.pt.y);
		});

	keypoints.erase(std::unique(keypoints.begin(), keypoints.end(),
		[](const cv::KeyPoint& a, const cv::KeyPoint& b) {
		return a.pt == b.pt;
		}), keypoints.end());

	//t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
	//std::cout << "Harris detection with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;

	// visualize results
	if (bVis) {
		cv::Mat visImage = img.clone();
		cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		string windowName = "Harris Corner Detector Results";
		cv::namedWindow(windowName, 6);
		imshow(windowName, visImage);
		cv::waitKey(0);
		}
	}


// Detect keypoints in image using the modern detectors
void detKeypointsModern(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, std::string detectorType, bool bVis) {

	if (detectorType.compare("SIFT") == 0) {
		// SIFT detector: gb
		//cv::Mat imgGray;
		//cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
		cv::Ptr<cv::xfeatures2d::SIFT> detectorSIFT = cv::xfeatures2d::SIFT::create();
		//cv::SiftFeatureDetector detectorSIFT;
		// cv::Ptr<cv::Feature2D> detectorSIFT = cv::SIFT::create();
		//cv::Ptr<cv::Feature2D> detectorSIFT = cv::SIFT::create(700, 3, 0.03, 10, 1.6);

		// explicitly typed initializer idiom
		// auto to avoid "invisible" proxy class type
		//auto t1 = static_cast<double>(cv::getTickCount());
		detectorSIFT->detect(img, keypoints);
		//t1 = (static_cast<double>(cv::getTickCount()) - t1) / cv::getTickFrequency();
		//std::cout << "SIFT detector with n = " << keypoints.size() << " keypoints in " << 1000 * t1 / 1.0 << " ms" << std::endl;
		}
	else if (detectorType.compare("FAST") == 0) {

		// Apply FAST corner detection: neither binary nor gb
		int blockSize = 6;
		cv::Ptr<cv::FeatureDetector> fast = cv::FastFeatureDetector::create();
		//cv::Ptr<cv::FeatureDetector> fast = cv::FastFeatureDetector::create(50, true, cv::FastFeatureDetector::TYPE_9_16);
		fast->detect(img, keypoints);
		}
	else if (detectorType.compare("BRISK") == 0) {

		// BRISK detector: binary
		cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
		detector->detect(img, keypoints);
		}
	else if (detectorType.compare("ORB") == 0) {
		// ORB detection: binary
		cv::Ptr<cv::Feature2D> detectORB = cv::ORB::create();
		detectORB->detect(img, keypoints);
		}
	else if (detectorType.compare("AKAZE") == 0) {
		// AKAZE detector: binary
		cv::Ptr<cv::Feature2D> detectorAKAZE = cv::AKAZE::create();
		detectorAKAZE->detect(img, keypoints);
		}
	else {
		std::cerr << "\n!!! Invalid detector type!!!" << std::endl;
		}
	// visualize results
	if (bVis) {
		cv::Mat visImage = img.clone();
		cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		string windowName = "Harris Corner Detector Results";
		cv::namedWindow(windowName, 6);
		imshow(windowName, visImage);
		cv::waitKey(0);
		}
	}
