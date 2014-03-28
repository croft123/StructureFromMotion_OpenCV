/**
 * @file SURF_FlannMatcher
 * @brief SURF detector + descriptor + FLANN Matcher
 * @author A. Huaman
 */

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/core/core.hpp>

#include "FeatureMatching.h"
#include "CalculateCameraMatrix.h"
#include "Triangulation.h"
#include "Common.h"
#include "SaveXYZimages.h"


#include <stdlib.h>
//#include <windows.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <math.h>



using namespace std;
using namespace cv;


void readme();

float imgdata[2448][3264][3];
float texture[2448][3264][3];
int width=0, height=0, rx = 0, ry = 0;  
int eyex = 30, eyez = 20, atx = 100, atz = 50; 
int eyey = -15;
float scalar = 0.1;        //scalar of converting pixel color to float coordinates 
vector<CloudPoint> pointcloud;
float allx = 0.0;
float ally = 0.0;
float allz = 0.0;



void special(int key, int x, int y)  
{  
    switch(key)  
    {  
    case GLUT_KEY_LEFT:  
        ry-=5;  
        glutPostRedisplay();  
        break;  
    case GLUT_KEY_RIGHT:  
        ry+=5;  
        glutPostRedisplay();  
        break;  
    case GLUT_KEY_UP:  
        rx+=5;  
        glutPostRedisplay();  
        break;  
    case GLUT_KEY_DOWN:  
        rx-=5;  
        glutPostRedisplay();  
        break;  
    }  
}  
  
//////////////////////////////////////////////////////////////////////////  

void renderScene(void) {  
  
    glClear (GL_COLOR_BUFFER_BIT);  
    glLoadIdentity();// Reset the coordinate system before modifying   
    gluLookAt (eyex, eyey, eyez, allx, ally, allz, 0.0, 1.0, 0.0);    //  
    glRotatef(ry, 0.0, 1.0, 0.0); //rotate about the z axis            // 
	//glRotatef(ry, allx, ally, 0); //rotate about the z axis  
    //glRotatef(rx-180, 1.0, 0.0, 0.0); //rotate about the y axis  
	glRotatef(rx, 1.0, 0.0, 0.0); //rotate about the y axis
	//glRotatef(rx, allx, ally, 0); //rotate about the y axis
  
    float x,y,z;  
  
    glPointSize(1.0);   
    glBegin(GL_POINTS);//GL_POINTS 
	for(int i=0;i<pointcloud.size();i++)
	{
		glColor3f(255,255,255);
		x = -(pointcloud[i].pt.x - allx)/scalar;        // 
		y = -(pointcloud[i].pt.y - ally)/scalar;     
		z = (pointcloud[i].pt.z - allz)/scalar;  
		glVertex3f(x,y,z); 
	}
	/*
    for (int i=0;i<height;i++){   
        for (int j=0;j<width;j++){  
            glColor3f(texture[i][j][0]/255, texture[i][j][1]/255, texture[i][j][2]/255);    //  
            x=-imgdata[i][j][0]/scalar;        // 
            y=-imgdata[i][j][1]/scalar;   
            z=imgdata[i][j][2]/scalar;   
            glVertex3f(x,y,z);   
        }  
    }  
	*/
    glEnd();  
    glFlush();  
}  
  
//////////////////////////////////////////////////////////////////////////  

void reshape (int w, int h) {  
    glViewport (0, 0, (GLsizei)w, (GLsizei)h);  
    glMatrixMode (GL_PROJECTION);  
    glLoadIdentity ();  
    gluPerspective (60, (GLfloat)w / (GLfloat)h, 1.0, 5000.0);    // 
    glMatrixMode (GL_MODELVIEW);  
}  





////Function Main
int main( int argc, char** argv )
{
 // if( argc != 3 )
 //{ readme(); return -1; }

	//Mat img_1 = imread( argv[1], IMREAD_GRAYSCALE );
	//Mat img_2 = imread( argv[2], IMREAD_GRAYSCALE );
	string filename1 = "C:\\OpenCV_Project\\SFM_Exp\\Building5.JPG";
	string filename2 = "C:\\OpenCV_Project\\SFM_Exp\\Building6.JPG";
	Mat img_1 = imread(filename1);
	Mat img_2 = imread(filename2);
	std::vector<KeyPoint> keypoints_1, keypoints_2,keypts1_good,keypts2_good, corr;
	std::vector< DMatch > matches;
	width = img_1.cols;
	height = img_1.rows;

	if( !img_1.data || !img_2.data )
	{ std::cout<< " --(!) Error reading images " << std::endl; return -1; } /// Read in Images

	// Start Feature Matching
	int Method = 1;
	FeatureMatching(img_1,img_2,keypoints_1,keypoints_2,keypts1_good,keypts2_good,&matches,Method); // matched featurepoints
	 // Calculate Matrices
	vector<Point2f> pts1,pts2;
	vector<uchar> status;


	vector<KeyPoint> imgpts1_tmp,imgpts1_good,imgpts2_good;
	vector<KeyPoint> imgpts2_tmp;
	GetAlignedPointsFromMatch(keypoints_1, keypoints_2, matches, imgpts1_tmp, imgpts2_tmp);
	KeyPointsToPoints(imgpts1_tmp, pts1);
	KeyPointsToPoints(imgpts2_tmp, pts2);
	double minVal,maxVal;
	cv::minMaxIdx(pts1,&minVal,&maxVal);

	Mat F = findFundamentalMat(pts1, pts2, FM_RANSAC, 0.006*maxVal, 0.99, status);
	double status_nz = countNonZero(status); 
	double status_sz = status.size();
	double kept_ratio = status_nz / status_sz;

	vector<DMatch> new_matches;
	cout << "F keeping " << countNonZero(status) << " / " << status.size() << endl;	
	for (unsigned int i=0; i<status.size(); i++) {
		if (status[i]) 
		{
			imgpts1_good.push_back(imgpts1_tmp[i]);
			imgpts2_good.push_back(imgpts2_tmp[i]);

			new_matches.push_back(matches[i]);

			//good_matches_.push_back(DMatch(imgpts1_good.size()-1,imgpts1_good.size()-1,1.0));
		}
	}	
	
	cout << matches.size() << " matches before, " << new_matches.size() << " new matches after Fundamental Matrix\n";
	matches = new_matches; //keep only those points who survived the fundamental matrix

	Mat img_matches;
	drawMatches( img_1, keypoints_1, img_2, keypoints_2,
		matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );		
	//-- Show detected matches
	imshow( "Feature Matches", img_matches );
	waitKey(0);
	destroyWindow("Feature Matches");
	imwrite("C:\\OpenCV_Project\\SFM_Exp\\Image_Matches.jpg",img_matches);



	/////////////////////
	Mat K,Kinv,discoeff; // Read from calibration file

	string filename = "C:\\OpenCV_Project\\camera_calibration\\result.xml";
	FileStorage fs(filename, FileStorage::READ);
	FileNode n = fs.getFirstTopLevelNode();
	fs["Camera_Matrix"] >> K;
	fs["Distortion_Coefficients"] >> discoeff;
	cout << "K " << endl << Mat(K) << endl;
	Kinv = K.inv();

	Matx34d P, P1;
	
	

	bool CM = FindCameraMatrices(K,Kinv,F,P,P1,discoeff,imgpts1_tmp,imgpts2_tmp,imgpts1_good,imgpts2_good,matches,pointcloud);
	
	// Reconstruct 3D
	//double mse = TriangulatePoints(keypts1_good,keypts2_good,K,Kinv,P,P1,pointcloud,keypts1_good,discoeff);

	// Write points to file
	Mat X(img_1.rows,img_1.cols,CV_32FC1);
	Mat Y(img_1.rows,img_1.cols,CV_32FC1);
	Mat Z(img_1.rows,img_1.cols,CV_32FC1);
	string filepath = "C:\\OpenCV_Project\\SFM_Exp\\";
	saveXYZimages(img_1,pointcloud,imgpts1_good,filepath,X,Y,Z);

	double Nindex = X.rows * X.cols;


	for(int i=0;i<pointcloud.size();i++ )
	{
		allx += pointcloud[i].pt.x;
		ally += pointcloud[i].pt.y;
		allz += pointcloud[i].pt.z;
	}
	allx = 1.0 * allx/(float)pointcloud.size();
	ally = 1.0 * ally/(float)pointcloud.size();
	allz = 1.0 * allz/(float)pointcloud.size();



	/*
	for(int i=0;i<X.rows;i++)
	{
		for(int j=0;j<X.cols;j++)
		{
			float* Xr =X.ptr<float>(i);
			imgdata[i][j][0] = Xr[j];
			float* TXr = img_1.ptr<float>(i);
			texture[i][j][0] = TXr[j];
		}
	}


	for(int i=0;i<Y.rows;i++)
	{
		for(int j=0;j<Y.cols;j++)
		{
			float* Yr =Y.ptr<float>(i);
			imgdata[i][j][1] = Yr[j];
			float* TYr = img_1.ptr<float>(i);
			texture[i][j][1] = TYr[j];
		}
	}


	for(int i=0;i<Z.rows;i++)
	{
		for(int j=0;j<Z.cols;j++)
		{
			float* Zr =Z.ptr<float>(i);
			imgdata[i][j][2] = Zr[j];
			float* TZr = img_1.ptr<float>(i);
			texture[i][j][2] = TZr[j];
		}
	}
	*/

	//////// OpenGL Draw

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_SINGLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(1024,768);
	glutCreateWindow("3D Reconstruct Model");
	glutReshapeFunc (reshape);            // 窗口变化时重构图像  
	glutDisplayFunc(renderScene);        // 显示三维图像  
	glutSpecialFunc(special);                // 响应方向键按键消息  
	glutPostRedisplay();    
	glutMainLoop();  

	//cvWaitKey(0);

	return 0;
}

/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SURF_FlannMatcher <img1> <img2>" << std::endl; }