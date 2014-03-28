#include "Common.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <vector>
#include <stdio.h>

using namespace std;
using namespace cv;







void FeatureMatching(const Mat& img_1, 
				   const Mat& img_2, 
				   vector<KeyPoint>& keypts1,
				   vector<KeyPoint>& keypts2,
				   vector<KeyPoint>& keypts1_good,
				   vector<KeyPoint>& keypts2_good,
				   	vector<DMatch>* matches,
					int method);