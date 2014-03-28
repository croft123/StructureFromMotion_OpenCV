
#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

#include "Common.h"
using namespace std;
using namespace cv;

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> LinearLSTriangulation(cv::Point3d u,		//homogenous image point (u,v,1)
								   cv::Matx34d P,		//camera 1 matrix
								   cv::Point3d u1,		//homogenous image point in 2nd camera
								   cv::Matx34d P1		//camera 2 matrix
								   );

#define EPSILON 0.00001
/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point3d u,	//homogenous image point (u,v,1)
											cv::Matx34d P,			//camera 1 matrix
											cv::Point3d u1,			//homogenous image point in 2nd camera
											cv::Matx34d P1			//camera 2 matrix
											);
double TriangulatePoints(const vector<KeyPoint>& pt_set1, 
						const vector<KeyPoint>& pt_set2, 
						const Mat& K,
						const Mat& Kinv,
						const Matx34d& P,
						const Matx34d& P1,
						vector<CloudPoint>& pointcloud,
						vector<KeyPoint>& correspImg1Pt,
						const Mat& distcoeff);
bool TestTriangulation(const vector<CloudPoint>& pcloud, const Matx34d& P, vector<uchar>& status);