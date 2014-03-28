#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <vector>
#include <stdio.h>

#include "Common.h"

using namespace std;
using namespace cv;

void saveXYZimages(const Mat& img_1, 
						vector<CloudPoint>& pointcloud,
						vector<KeyPoint>& correspImg1Pt,
						string filepath,
						Mat& X,
						Mat& Y,
						Mat& Z);