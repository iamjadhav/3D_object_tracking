
#ifndef camFusion_hpp
#define camFusion_hpp

#include <stdio.h>
#include <vector>
#include <opencv2/core.hpp>
#include "dataStructures.h"


void clusterLidarWithROI(std::vector<BoundingBox>& boundingBoxes, std::vector<LidarPoint>& lidarPoints, float shrinkFactor, cv::Mat& P_rect_xx, cv::Mat& R_rect_xx, cv::Mat& RT);
void clusterKptMatchesWithROI(BoundingBox& boundingBox, std::vector<cv::KeyPoint>& kptsPrev, std::vector<cv::KeyPoint>& kptsCurr, std::vector<cv::DMatch>& kptMatches);
void matchBoundingBoxes(std::vector<cv::DMatch>& matches, std::map<int, int>& bbBestMatches, DataFrame& prevFrame, DataFrame& currFrame);

void show3DObjects(std::vector<BoundingBox>& boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait = true);

void computeTTCCamera(std::vector<cv::KeyPoint>& kptsPrev, std::vector<cv::KeyPoint>& kptsCurr,
					  std::vector<cv::DMatch> kptMatches, double frameRate, double& TTC, std::vector<double>& ttcs_camera, cv::Mat* visImg = nullptr);
bool isDistanceWithinThreshold(const cv::Point& kptPrev, const cv::Point& kptCurr, double lowerT, double upperT);
double calculateEuclideanDistance(const cv::Point& p1, const cv::Point& p2);
double ransacLidar(std::vector<LidarPoint>& lidarPoints);
void computeTTCLidar(std::vector<LidarPoint>& lidarPointsPrev,
					 std::vector<LidarPoint>& lidarPointsCurr, double frameRate, double& TTC, std::vector<double>& ttcs_lidar);
double calculateMedian(std::vector<double>& data);
void calculateThresholdsUsingIQR(std::vector<double>& data, double& lowerThreshold, double& upperThreshold);
#endif /* camFusion_hpp */
