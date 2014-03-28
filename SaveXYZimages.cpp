
#include "SaveXYZimages.h"
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
#include <fstream>

#include <iostream>
#include <set>
using namespace std;
using namespace cv;

void saveXYZimages(const Mat& img_1, 
						vector<CloudPoint>& pointcloud,
						vector<KeyPoint>& correspImg1Pt,
						string filepath,
						Mat& X,
						Mat& Y,
						Mat& Z
	)
{
	string filenameImage = "C.bmp";
	string filenameX = "X.byt";
	string filenameY = "Y.byt";
	string filenameZ = "Z.byt";
	string FullPathImage = filepath + filenameImage;
	string FullPathX = filepath + filenameX;
	string FullPathY = filepath + filenameY;
	string FullPathZ = filepath + filenameZ;
	imwrite(FullPathImage,img_1);



	X = Mat::zeros(X.rows, X.cols, CV_32FC1);
	Y = Mat::zeros(Y.rows, Y.cols, CV_32FC1);
	Z = Mat::zeros(Z.rows, Z.cols, CV_32FC1);
	vector<Point2f> pts1;
	KeyPointsToPoints(correspImg1Pt, pts1);

	for (unsigned int i=0; i<pointcloud.size(); i++) 
	{
		int xcor = pts1[i].x;
		int ycor = pts1[i].y;
		if(ycor > Y.rows || xcor > X.cols)
		{
			int test = 1;
		}
		
		
		float* Xi =X.ptr<float>(ycor);
		Xi[xcor] = (float) pointcloud[i].pt.x;

		float* Yi =Y.ptr<float>(ycor);
		Yi[xcor-1] = (float) pointcloud[i].pt.y;

		float* Zi =Z.ptr<float>(ycor);
		Zi[xcor] = (float) pointcloud[i].pt.z;
		
		/*
		Mat B = X(Range(ycor,ycor+1),Range(xcor,xcor+1));
		cout << "coordinate X "<< endl << B<< endl;
		cout << "coordinate X "<< endl << Xi[xcor] << endl;
		*/
		
	}

	double Nindex = X.rows * X.cols;
	
	FILE *fs_x;
	fs_x = fopen(FullPathX.c_str(),"wb");
	for(int i=0;i<X.rows;i++)
	{
		for(int j=0;j<X.cols;j++)
		{
			float* Xr =X.ptr<float>(i);
			fwrite(&Xr[j],sizeof(float),1,fs_x);
			if(Xr[j] != 0)
				int test = 1;
		}
	}
	fclose(fs_x);

	FILE *fs_y;
	fs_y = fopen(FullPathY.c_str(),"wb");
	for(int i=0;i<Y.rows;i++)
	{
		for(int j=0;j<Y.cols;j++)
		{
			float* Yr =Y.ptr<float>(i);
			fwrite(&Yr[j],sizeof(float),1,fs_y);
		}
	}
	fclose(fs_y);

	FILE *fs_z;
	fs_z = fopen(FullPathZ.c_str(),"wb");
	for(int i=0;i<Z.rows;i++)
	{
		for(int j=0;j<Z.cols;j++)
		{
			float* Zr =Z.ptr<float>(i);
			fwrite(&Zr[j],sizeof(float),1,fs_z);
		}
	}
	fclose(fs_z);

	


	//imwrite(FullPathX,X);
	//imwrite(FullPathY,Y);
	//imwrite(FullPathZ,Z);
	

}