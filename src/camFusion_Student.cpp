#include <iostream>
#include <algorithm>
#include <set>
#include <unordered_map>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


 /* start: utils */
 
// Median of a vector
double calculateMedian(std::vector<double>& data) {
	size_t size = data.size();
	std::sort(data.begin(), data.end());

	if (size % 2 == 0) {
		return (data[size / 2 - 1] + data[size / 2]) / 2.0;
	}
	else {
		return data[size / 2];
	}
}


// Calculate lower and upper thresholds for a data vector using 1.5*IQR rule
void calculateThresholdsUsingIQR(std::vector<double>& data, double& lowerThreshold, double& upperThreshold) {
	size_t size = data.size();
	size_t index_Q1 = size / 4;
	size_t index_Q3 = 3 * size / 4;
	double q1, q3;

	if (size % 4 == 0) {  // direct 4 quartiles for even range; take average
		q1 = (data[index_Q1 - 1] + data[index_Q1]) / 2.0;
		q3 = (data[index_Q3 - 1] + data[index_Q3]) / 2.0;
	}
	else {
		q1 = data[index_Q1];  // for even range c++ already uses floor() so fine for array indices
		q3 = data[index_Q3];
	}

	lowerThreshold = q1 - 1.5 * (q3 - q1);
	upperThreshold = q3 + 1.5 * (q3 - q1);
}


/* end: utils */

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox>& boundingBoxes, std::vector<LidarPoint>& lidarPoints, float shrinkFactor, cv::Mat& P_rect_xx, cv::Mat& R_rect_xx, cv::Mat& RT) {
	// loop over all Lidar points and associate them to a 2D bounding box
	cv::Mat X(4, 1, cv::DataType<double>::type);
	cv::Mat Y(3, 1, cv::DataType<double>::type);

	//bool skipLidarPoint = false;
	for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1) {

		// assemble vector for matrix-vector-multiplication
		X.at<double>(0, 0) = it1->x;
		X.at<double>(1, 0) = it1->y;
		X.at<double>(2, 0) = it1->z;
		X.at<double>(3, 0) = 1;

		// project Lidar point into camera
		Y = P_rect_xx * R_rect_xx * RT * X;
		cv::Point pt;
		// pixel coordinates
		pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
		pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

		std::vector<std::vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
		for (std::vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2) {
			// shrink current bounding box slightly to avoid having too many outlier points around the edges
			cv::Rect smallerBox;
			smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
			smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
			smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
			smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

			// check wether point is within current bounding box
			if (smallerBox.contains(pt)) {
				enclosingBoxes.push_back(it2);
			}

		} // eof loop over all bounding boxes

		// check wether point has been enclosed by one or by multiple boxes
		if (enclosingBoxes.size() == 1) {
			// add Lidar point to bounding box
			enclosingBoxes[0]->lidarPoints.push_back(*it1);
		}

	} // eof loop over all Lidar points
}

/*
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox>& boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait) {
	// create topview image
	cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

	for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1) {
		// create randomized color for current 3D object
		cv::RNG rng(it1->boxID);
		cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

		// plot Lidar points into top view image
		int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
		float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
		for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2) {
			// world coordinates
			float xw = (*it2).x; // world position in m with x facing forward from sensor
			float yw = (*it2).y; // world position in m with y facing left from sensor
			xwmin = xwmin < xw ? xwmin : xw;
			ywmin = ywmin < yw ? ywmin : yw;
			ywmax = ywmax > yw ? ywmax : yw;

			// top-view coordinates
			int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
			int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

			// find enclosing rectangle
			top = top < y ? top : y;
			left = left < x ? left : x;
			bottom = bottom > y ? bottom : y;
			right = right > x ? right : x;

			// draw individual point
			cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
		}

		// draw enclosing rectangle
		cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

		// augment object with some key data
		char str1[200], str2[200];
		sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
		putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
		sprintf(str2, "xmin=%2.2f m, ywmax=%2.2f m, ywmin=%2.2f m", xwmin, ywmax, ywmin);
		//std::cout << "Lidar Min X: " << xwmin << " m" << std::endl;
		putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
	}

	// plot distance markers
	float lineSpacing = 2.0; // gap between distance markers
	int nMarkers = floor(worldSize.height / lineSpacing);
	for (size_t i = 0; i < nMarkers; ++i) {
		int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
		cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
	}

	// display image
	string windowName = "3D Objects";
	cv::namedWindow(windowName, 0);
	cv::imshow(windowName, topviewImg);

	if (bWait) {
		cv::waitKey(0); // wait for key to be pressed
	}
}


// Calculate euclidean distance between two points
double calculateEuclideanDistance(const cv::Point& p1, const cv::Point& p2) {
	double dx = p2.x - p1.x;
	double dy = p2.y - p1.y;
	return std::sqrt(dx * dx + dy * dy);
}

// Filter matches based on distance
bool isDistanceWithinThreshold(const cv::Point& kptPrev, const cv::Point& kptCurr, double lowerT, double upperT) {
	double distance = calculateEuclideanDistance(kptPrev, kptCurr);
	return distance >= lowerT && distance <= upperT;
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox& boundingBox, std::vector<cv::KeyPoint>& kptsPrev, std::vector<cv::KeyPoint>& kptsCurr, std::vector<cv::DMatch>& kptMatches) {

	//
	//std::cout << "\nTotal keypoints previous frame and current frame: " << kptsPrev.size() << " | " << kptsCurr.size() << std::endl;
	//std::cout << "BoxID and total matches: " << boundingBox.boxID << " | " << kptMatches.size() << std::endl;

	// use a vector to store currBB match distances
	std::vector<double> distances;

	// eff?: yes, 4x faster
	auto condition = [&boundingBox, &kptsPrev, &kptsCurr](const cv::DMatch& match) {
		const cv::Point kptPrev = kptsPrev[match.queryIdx].pt;
		const cv::Point kptCurr = kptsCurr[match.trainIdx].pt;
		return boundingBox.roi.contains(kptPrev) && boundingBox.roi.contains(kptCurr);
	};

	std::copy_if(kptMatches.begin(), kptMatches.end(), std::back_inserter(boundingBox.kptMatches), condition);

	//std::cout << "currBB matches: " << boundingBox.kptMatches.size();

	for (auto it = boundingBox.kptMatches.begin(); it != boundingBox.kptMatches.end(); it++) {
		cv::Point kptPrev = kptsPrev[it->queryIdx].pt;
		cv::Point kptCurr = kptsCurr[it->trainIdx].pt;
		double distance = calculateEuclideanDistance(kptPrev, kptCurr);
		distances.push_back(distance);
		//std::cout << " " << distance;
	};

	double lowerT = 0.0, upperT = 0.0;
	std::sort(distances.begin(), distances.end());

	calculateThresholdsUsingIQR(distances, lowerT, upperT);

	// erase points based on lower and upper IQR thresholds
	boundingBox.kptMatches.erase(std::remove_if(boundingBox.kptMatches.begin(), boundingBox.kptMatches.end(),
												[kptsPrev, kptsCurr, lowerT, upperT](const cv::DMatch& match) {
													cv::Point kptPrev = kptsPrev[match.queryIdx].pt;
													cv::Point kptCurr = kptsCurr[match.trainIdx].pt;
													double distance = calculateEuclideanDistance(kptPrev, kptCurr);

											 return !isDistanceWithinThreshold(kptPrev, kptCurr, lowerT, upperT);
										 }),
								 boundingBox.kptMatches.end());
}



// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint>& kptsPrev, std::vector<cv::KeyPoint>& kptsCurr,
					  std::vector<cv::DMatch> kptMatches, double frameRate, double& TTC, std::vector<double>& ttcs_camera, cv::Mat* visImg) {
	//std::cout << "kptsprev size: " << kptsPrev.size() << "\tkptscurr size: " << kptsCurr.size() << "\tkptsmatches size: " << kptMatches.size() << std::endl;
	// compute distance ratios between all matched keypoints
	std::vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame

	for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1) { // outer keypoint loop
		// train set -> keypoints from current image (second image)
		// query set -> keypoints from previous image (first image)
		cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
		cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

		for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2) { // inner keypoint loop

			double minDist = 100.0; // min. required distance
			//cv::KeyPoint kpInnerCurr, kpInnerPrev;
			cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
			cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

			//std::cout << "Inner train index: " << it1->trainIdx << "\tInner query index: " << it1->queryIdx << std::endl;

			// compute distances and distance ratios
			double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
			double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

			//std::cout << "\ndist curr: " << distCurr << "\tdist prev: " << distPrev << std::endl;

			if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) { // avoid division by zero
				double distRatio = distCurr / distPrev;
				distRatios.push_back(distRatio);
			}
		} // eof inner loop over all matched kpts
	}     // eof outer loop over all matched kpts

	// only continue if list of distance ratios is not empty
	if (distRatios.size() == 0) {
		TTC = NAN;
		return;
	}

	// compute camera-based TTC from distance ratios
	double medianDistRatio = 0.0;
	medianDistRatio = calculateMedian(distRatios);
	//std::cout << "median dist ratio: " << medianDistRatio << std::endl;

	double dT = 1 / frameRate;
	TTC = -dT / (1 - medianDistRatio);
	ttcs_camera.push_back(TTC);
	std::cout << "Camera TTC: " << TTC << "\n" << std::endl;
}


void computeTTCLidar(std::vector<LidarPoint>& lidarPointsPrev,
					 std::vector<LidarPoint>& lidarPointsCurr, double frameRate, double& TTC, std::vector<double>& ttcs_lidar) {

	//std::cout << "Total prev and curr lidar points: " << lidarPointsPrev.size() << " | " << lidarPointsCurr.size() << std::endl;

	double dT = 0.0;
	// not robust yet: done
	// find closest distance to Lidar points within ego lane
	double minXPrev = 1e9, minXCurr = 1e9;

	//for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); it++)
	//	std::cout << "x: " << it->x << " ";

	double lowerTprev, upperTprev, lowerTcurr, upperTcurr;
	std::vector<double> prevXvals, currXvals;

	// extract x values from prev cloud
	auto getX = [](const LidarPoint& point) {
		return point.x;
	};

	std::transform(lidarPointsPrev.begin(), lidarPointsPrev.end(), std::back_inserter(prevXvals), getX);

	//std::cout << "\nx extracted";
	//for (auto it = prevXvals.begin(); it != prevXvals.end(); it++)
	//	std::cout << "x: " << *it << " ";

	std::transform(lidarPointsCurr.begin(), lidarPointsCurr.end(), std::back_inserter(currXvals), getX);

	//std::cout << lidarPointsPrev.size() << " " << prevXvals.size() << std::endl;
	//std::cout << lidarPointsCurr.size() << " " << currXvals.size() << std::endl;

	std::sort(prevXvals.begin(), prevXvals.end());
	std::sort(currXvals.begin(), currXvals.end());

	//std::cout << "\nsorted";
	//for (auto it = prevXvals.begin(); it != prevXvals.end(); it++)
	//	std::cout << "x: " << *it << " ";

	calculateThresholdsUsingIQR(prevXvals, lowerTprev, upperTprev);
	calculateThresholdsUsingIQR(currXvals, lowerTcurr, upperTcurr);

	// erase points based on lower and upper IQR thresholds
	lidarPointsPrev.erase(std::remove_if(lidarPointsPrev.begin(), lidarPointsPrev.end(),
										 [&lowerTprev, &upperTprev](const LidarPoint& prevPoint) {
											 return prevPoint.x <= lowerTprev || prevPoint.x >= upperTprev;
										 }),
						  lidarPointsPrev.end());

	lidarPointsCurr.erase(std::remove_if(lidarPointsCurr.begin(), lidarPointsCurr.end(),
										 [&lowerTcurr, &upperTcurr](const LidarPoint& currPoint) {
											 return currPoint.x <= lowerTcurr || currPoint.x >= upperTcurr;
										 }),
						  lidarPointsCurr.end());

	// find closest X values in outlier-free clouds
	for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it) {
		//std::cout << typeid(it).name() << std::endl;  // std::type_info object
		if (std::abs(it->y) <= 1.5 && it->x > 0) {  // pts in front; avoid reflections
			minXPrev = minXPrev > it->x ? it->x : minXPrev;
		}
	}

	for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it) {
		if (std::abs(it->y) <= 1.5 && it->x > 0) {  // pts in front; avoid reflections
			minXCurr = minXCurr > it->x ? it->x : minXCurr;
		}
	}

	//std::cout << "\nminPrev: " << minXPrev << std::endl;
	//std::cout << "\nminCurr: " << minXCurr << std::endl;

	// compute TTC from both measurements
	dT = 1 / frameRate;
	TTC = minXCurr * dT / (minXPrev - minXCurr);  // use std::abs??
	ttcs_lidar.push_back(TTC);
	std::cout << "Lidar TTC: " << TTC << std::endl;
}


void matchBoundingBoxes(std::vector<cv::DMatch>& matches, std::map<int, int>& bbBestMatches, DataFrame& prevFrame, DataFrame& currFrame) {
	// outer loop over current and previous frame
	// find which bbs keypoints are enclosed in, store boxids in map?
	// find and match bbs in map, then sort descending?

	//std::cout << "Total matches: " << matches.size() << std::endl;

	// brute-force solution
	std::multimap<int, int> totalBBMatches;
	for (auto it1 = matches.begin(); it1 != matches.end(); ++it1) {  // 3k+
		// extract keypoint coordinates
		cv::Point keypointPrev = prevFrame.keypoints[it1->queryIdx].pt;
		cv::Point keypointCurr = currFrame.keypoints[it1->trainIdx].pt;
		int nMatches = 0;
		// search for corresponding boxes in both frames
		for (auto it2 = prevFrame.boundingBoxes.begin(); it2 != prevFrame.boundingBoxes.end(); ++it2) {  // 10+
			bool foundPrev = it2->roi.contains(keypointPrev);
			if (foundPrev) {
				for (auto it3 = currFrame.boundingBoxes.begin(); it3 != currFrame.boundingBoxes.end(); ++it3) {  // 10+
					if (it3->roi.contains(keypointCurr)) {
						//std::cout << keypointPrev.x << "," << keypointPrev.y << "~" << keypointCurr.x << "," << keypointCurr.y << std::endl;
						//std::cout << it2->boxID << "," << it3->boxID << std::endl;
						totalBBMatches.insert(std::pair<int, int>(it2->boxID, it3->boxID));
						//bbBestMatches.insert(std::pair<int, int>(it2->boxID, it3->boxID));  // map() stores only the first match
					}
				}  // eof over all bbs in current frame
			}  // ~ previous frame
		}
	}  // eof over all matches

	//std::cout << "\ntotalMatches: " << totalBBMatches.size() << std::endl;

	// map with multiset for values to find the most frequest value i.e. the matching currBB
	std::map<int, std::multiset<int>> m2;

	// crucial step
	//int count = 0;
	for (auto& it : totalBBMatches) {
		m2[it.first].insert(it.second);
		//count++;
	}

	// sorted map
	// pair.first = int Key
	auto t = static_cast<double>(cv::getTickCount());
	for (const auto& pair : m2) {
		//std::cout << "Key: " << pair.first;
		int maxVal = 0;
		int maxCount = -1e8;
		// pair.second = multiset for Key
		for (const auto& value : pair.second) {
			//std::cout << value << " ";
			// could optimize
			int currCount = pair.second.count(value);
			if (currCount > (int)(0.5 * pair.second.size())) {
				maxCount = currCount;
				maxVal = value;
				break;
			}
			if (maxCount < currCount) {
				maxCount = currCount;
				maxVal = value;
			}
		}
		//std::cout << ", Max Value: " << maxVal << " with count " << maxCount;
		//std::cout << " (Multiset Count: " << pair.second.size() << ")" << std::endl;
		bbBestMatches.insert(std::pair<int, int>(pair.first, maxVal));
	}

	t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
	//std::cout << "Sorted to " << std::endl;

	//for (auto match = totalBBMatches.begin(); match != totalBBMatches.end(); ++match) {
	//	std::cout << '\t' << match->first << '\t' << match->second << std::endl;
	//}

	//std::cout << '\n' << std::endl;

	//std::cout << "bbBestMatches: " << bbBestMatches.size() << " in " << 1000 * t / 1.0 << " ms" << std::endl;

	//std::cout << "\nprevious box" << "   current box" << std::endl;
	//for (auto match = bbBestMatches.begin(); match != bbBestMatches.end(); ++match) {
	//	std::cout << '\t' << match->first << '\t' << match->second << std::endl;
	//}

}
