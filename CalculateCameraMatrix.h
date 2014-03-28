#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Common.h"
using namespace std;
using namespace cv;

bool FindCameraMatrices(const Mat& K, 
						const Mat& Kinv,
						const Mat& F,
						Matx34d& P,
						Matx34d& P1,
						const Mat& distcoeff,
						const vector<KeyPoint>& imgpts1,
						const vector<KeyPoint>& imgpts2,
						vector<KeyPoint>& imgpts1_good,
						vector<KeyPoint>& imgpts2_good,
						vector<DMatch>& matches,
						vector<CloudPoint>& outCloud);














