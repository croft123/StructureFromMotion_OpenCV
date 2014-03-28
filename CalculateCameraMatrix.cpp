#include "CalculateCameraMatrix.h"
#include "Triangulation.h"
#include "Common.h"



#include <vector>
#include <iostream>

#include <opencv2/calib3d/calib3d.hpp>


using namespace cv;
using namespace std;




bool FindCameraMatrices(const Mat& K, 
						const Mat& Kinv,
						const Mat& F,
						Matx34d& P,
						Matx34d& P1,
						const Mat& discoeff,
						const vector<KeyPoint>& imgpts1,
						const vector<KeyPoint>& imgpts2,
						vector<KeyPoint>& imgpts1_good,
						vector<KeyPoint>& imgpts2_good,
						vector<DMatch>& matches,
						vector<CloudPoint>& outCloud)
{
	Mat_<double> E = K.t() * F * K; // Essential Matrix
	if(fabsf(determinant(E)) > 1e-05) {
		cout << "det(E) != 0 : " << determinant(E) << "\n";
		return false;
	}
	Mat_<double> R1(3,3);
	Mat_<double> R2(3,3);
	Mat_<double> t1(1,3);
	Mat_<double> t2(1,3);
	SVD svd(E);
	Matx33d W(0,-1,0,	
		1,0,0,
		0,0,1);
	Matx33d Wt(0,1,0,
		-1,0,0,
		0,0,1);
	R1 = svd.u * Mat(W) * svd.vt; //Rotation solution 1
	R2 = svd.u * Mat(Wt) * svd.vt; //Rotation solution 2
	t1 = svd.u.col(2); //Translatiion solution 1
	t2 = -svd.u.col(2); //Translation solution 2

	P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
				R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
				R1(2,0),	R1(2,1),	R1(2,2),	t1(2)); // Camara Matrix
	P = Matx34d(1,0,0,0,
				0,1,0,0,
				0,0,1,0);

	cout << "Testing P1 " << endl << Mat(P1) << endl;

	vector<CloudPoint> pcloud,pcloud1; 
	vector<KeyPoint> corresp;

	double reproj_error1 = TriangulatePoints(imgpts1_good,imgpts2_good,K,Kinv,P,P1,pcloud,corresp,discoeff);
	double reproj_error2 = TriangulatePoints(imgpts2_good,imgpts1_good,K,Kinv,P1,P,pcloud1,corresp,discoeff);
	vector<uchar> tmp_status;
	if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) 
	{
				P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t2(0),
							 R1(1,0),	R1(1,1),	R1(1,2),	t2(1),
							 R1(2,0),	R1(2,1),	R1(2,2),	t2(2));
				cout << "Testing P1 "<< endl << Mat(P1) << endl;

				pcloud.clear(); pcloud1.clear(); corresp.clear();
				reproj_error1 = TriangulatePoints(imgpts1_good,imgpts2_good,K,Kinv,P,P1,pcloud,corresp,discoeff);
				reproj_error2 = TriangulatePoints(imgpts2_good,imgpts1_good,K,Kinv,P1,P,pcloud1,corresp,discoeff);
				
				if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {

					
					P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t1(0),
								 R2(1,0),	R2(1,1),	R2(1,2),	t1(1),
								 R2(2,0),	R2(2,1),	R2(2,2),	t1(2));
					//cout << "Testing P1 "<< endl << Mat(P1) << endl;

					pcloud.clear(); pcloud1.clear(); corresp.clear();
					reproj_error1 = TriangulatePoints(imgpts1_good,imgpts2_good,K,Kinv,P,P1,pcloud,corresp,discoeff);
					reproj_error2 = TriangulatePoints(imgpts2_good,imgpts1_good,K,Kinv,P1,P,pcloud1,corresp,discoeff);
					
					if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
						P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t2(0),
									 R2(1,0),	R2(1,1),	R2(1,2),	t2(1),
									 R2(2,0),	R2(2,1),	R2(2,2),	t2(2));
						cout << "Testing P1 "<< endl << Mat(P1) << endl;

						pcloud.clear(); pcloud1.clear(); corresp.clear();
						reproj_error1 = TriangulatePoints(imgpts1_good,imgpts2_good,K,Kinv,P,P1,pcloud,corresp,discoeff);
						reproj_error2 = TriangulatePoints(imgpts2_good,imgpts1_good,K,Kinv,P1,P,pcloud1,corresp,discoeff);
						
						if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
							cout << "Err is too big." << endl; 
							return false;
						}
					}				
				}			
			}
			for (unsigned int i=0; i<pcloud.size(); i++) {
				outCloud.push_back(pcloud[i]);
			}

}





















