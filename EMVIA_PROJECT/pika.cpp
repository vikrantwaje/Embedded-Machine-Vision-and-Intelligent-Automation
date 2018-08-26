//#define _GNU_SOURCE
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <sched.h>
#include <time.h>

#include<sys/sysinfo.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>
#include <string>
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <vector>
#include <semaphore.h>

bool peds = 0;
bool cars = 0;
bool signs = 0;
bool lanes = 0;
bool box = 0;
bool show = 0;
bool store = 0;

using namespace cv;
using namespace std;

char input_file_name[30];
char output_file_name[20];

int frame_count = 1;

String cars_cascade_name = "cars_my2.xml";
CascadeClassifier cars_cascade;

String stop_cascade_name = "stop.xml";
CascadeClassifier stop_cascade;

HOGDescriptor hog;

Point lineintersection( Point A, Point B, Point C, Point D);

float slope( Point A, Point B);

void detectAndDisplay( Mat frame );

void lanedetect( Mat src);

void pedestrian_detect(Mat img);

void roadsign_detect( Mat frame );


int main(int argc, const char * argv[])
{

  for(int i=0; i<argc; i++)
	{
		if(string(argv[i]) == "--peds")
			peds = 1;

    if(string(argv[i]) == "--box")
  		box = 1;

		if(string(argv[i]) == "--cars")
			cars = 1;

		if(string(argv[i]) == "--signs")
			signs = 1;

		if(string(argv[i]) == "--lanes")
			lanes = 1;

    if(string(argv[i]) == "--show")
    	show = 1;

    if(string(argv[i]) == "--store")
      store = 1;

	}

  strcpy(input_file_name,argv[1]);

  VideoCapture cap(input_file_name);

  Mat frame;

  if( !cars_cascade.load( cars_cascade_name ) )
  {
     printf("Error loading cars cascade\n");
      //return -1;
  }


  hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());


  if( !stop_cascade.load( stop_cascade_name ) )
  {
     printf("Error loading stop sign cascade\n");
      //return -1;
  }



  while(1)
  {
    cap >> frame;

    if(lanes == 1)
      lanedetect(frame);

    if(cars == 1)
      detectAndDisplay(frame);

    if(signs == 1)
      roadsign_detect(frame);

    if(peds == 1)
      pedestrian_detect(frame);

    imshow("frame",frame);
    char c=waitKey(1);
    if(c == 27)	break;

    if(store == 1)
    {
      sprintf(output_file_name,"image%05d.ppm",frame_count);

      cout<<"Writing frame "<<frame_count<<endl;

      imwrite(output_file_name,frame);

    }

    frame_count++;

  }

}



Point lineintersection( Point A, Point B, Point C, Point D)
{
			double a1 = B.y - A.y;
			double b1 = A.x - B.x;
			double c1 = a1*(A.x) + b1*(A.y);

			double a2 = D.y - C.y;
			double b2 = C.x - D.x;
			double c2 = a2*(C.x) + b2*(C.y);

			double determinant = a1*b2 - a2*b1;

			if(determinant == 0)
			{
					// The lines are parallel
					return Point(0,0);
			}
			else
			{
					double x = (b2*c1 - b1*c2)/determinant;
					double y = (a1*c2 - a2*c1)/determinant;

					return Point(x,y);
			}

}


float slope( Point A, Point B)
{
		float s=1.0;
		float dx;
		float dy;

		if(B.x == A.x)
			return 1000;
		else
		{
			dx = (B.y - A.y);
			dy = (B.x - A.x);
			s = dx/dy;
		}

		// cout << "dx = "<<dx<<endl;
		// cout << "dy = "<<dy<<endl;
		// cout << "Slope = "<<s<<endl;

		return s;

}


void detectAndDisplay( Mat frame )
{

	std::vector<Rect> cars;
	Mat frame_gray;

	float hypot;

	cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

//	imshow("hist",frame_gray);

	  cars_cascade.detectMultiScale( frame_gray, cars, 1.1, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

		for( size_t i = 0; i < cars.size(); i++ )
		{
			Point center( cars[i].x + cars[i].width*0.5, cars[i].y + cars[i].height*0.5 );
			//ellipse( frame, center, Size( cars[i].width*0.5, cars[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 0 ), 1, 8, 0 );

			hypot = sqrt((cars[i].width)^2 + (cars[i].height)^2);

//			cout<<"H = "<<hypot<<endl;
			if(box == 1){
			if(hypot>=12)
			{
				rectangle( frame, cars[i].tl(), cars[i].br(), cv::Scalar(0,0,255), 2);
			}
			else if(hypot>= 5)
			{
				rectangle( frame, cars[i].tl(), cars[i].br(), cv::Scalar(0,255,255), 2);
			}
			else
			{
				rectangle( frame, cars[i].tl(), cars[i].br(), cv::Scalar(0,255,100), 2);
			}
			putText( frame, "Car", Point( cars[i].x, (cars[i].y)-5), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0,255,100), 2);
		}
	}
	if(show == 1)
	imshow("out",frame);
}


void lanedetect( Mat src)
{

	Mat gray,dst,hsv,yellow_mask,white_mask,yw_mask,out_and,img_blur,canny_out,img_roi,myROI,hough_out;


	int c,r;

	c = src.cols;
	r = src.rows;

	//Display the image
	if(show == 1)
	imshow("Input",src);

	//Converting to grayscale
	cvtColor(src,gray,COLOR_BGR2GRAY);

	//imshow("Grayscale",gray);

	cvtColor(src,hsv,COLOR_BGR2HSV);

	//Display the hsv output
	//imshow("HSV",hsv);

	inRange( hsv, Scalar(10,80,60), Scalar(40,200,150), yellow_mask);

	//imshow("yellow mask",yellow_mask);

//	inRange( src, Scalar(120,120,120), Scalar(255,255,255), white_mask);

	inRange( gray, 130, 255, white_mask);

	//imshow("white mask",white_mask);

	bitwise_or(yellow_mask,white_mask,yw_mask);

	if(show == 1)
	imshow("Yellow and White Mask",yw_mask);

	bitwise_and(yw_mask,gray,out_and);

	//imshow("and",out_and);

	GaussianBlur(out_and,img_blur,Size(5,5),0,0);

	//imshow("Blur",img_blur);

	Canny(img_blur,canny_out,50,100);

	if(show == 1)
	imshow("Canny",canny_out);

	Mat mask;

	mask = Mat::zeros(src.size(),CV_8UC1);

	vector<Point> ROI_Vertices;
	vector<Point> ROI_Poly;

	ROI_Vertices.push_back(Point(0.45*c,0.5*r));
	ROI_Vertices.push_back(Point(0.55*c,0.5*r));
	ROI_Vertices.push_back(Point(0.75*c,0.8*r));
	ROI_Vertices.push_back(Point(0.25*c,0.8*r));

	approxPolyDP(ROI_Vertices, ROI_Poly, 1.0, true);

	fillConvexPoly(mask, &ROI_Poly[0], ROI_Poly.size(), 255, 8, 0);

	if(show == 1)
	imshow("mask_poly",mask);

	Rect ROI(c/4,0.5*r,c/2,(0.3*r));

	myROI = Mat::zeros(src.size(),CV_8UC1);

	//Rect WhereRec(c/4,0.5*r,c/2,0.3*r);

	//img_roi.copyTo(myROI(ROI));

	canny_out.copyTo(myROI,mask);

	if(show == 1)
	imshow("myROI",myROI);

	vector<Vec4i> linesP;
	HoughLinesP(myROI,linesP,3,CV_PI/180,80,25,40);

	Point left_A = Point(0.2*c,0.7*r);
	Point left_B = Point(0.45*c,0.7*r);
	Point right_A = Point(0.55*c,0.7*r);
	Point right_B = Point(0.8*c,0.7*r);
	Point cntr = Point(0.5*c,0.7*r);

	line( src, left_A, left_B, Scalar(0,255,100), 1, 1);
	line( src, right_A, right_B, Scalar(0,255,100), 1, 1);
	line( src, Point(0.5*c,0.6*r), Point(0.5*c,0.8*r), Scalar(255,255,255), 2, 1);

	double d_left, d_right, d;

	//Drawing the lines
	for(size_t i=0;i<linesP.size();i++){

		Vec4i l=linesP[i];

		if( slope(Point(l[0],l[1]),Point(l[2],l[3])) > 0.8 || slope(Point(l[0],l[1]),Point(l[2],l[3])) < -0.8)
		{
			if(lanes == 1){
		line(myROI,Point(l[0],l[1]),Point(l[2],l[3]),Scalar(255,0,0),3,1);
		line(src,Point(l[0],l[1]),Point(l[2],l[3]),Scalar(255,0,0),3,1);
				}
		Point left = lineintersection( left_A, left_B, Point(l[0],l[1]), Point(l[2],l[3]));

		if( left.x>=left_A.x && left.x<=left_B.x )
		{
				//circle(src,left,5,Scalar(0,0,255),-1);
				d_left = left.x - cntr.x;
		}
		else
		{
			d_left = 0;
		}

		Point right = lineintersection( right_A, right_B, Point(l[0],l[1]), Point(l[2],l[3]));

		if( right.x>=right_A.x && right.x<=right_B.x )
		{
			//	circle(src,right,5,Scalar(0,0,255),-1);
				d_right = right.x - cntr.x;
		}
		else
		{
			d_right = 0;
		}

		d = d_right + d_left ;

		//line(src, cntr, Point((0.5*c)+d,0.7*r), Scalar(255,255,255), 2, 1);

		if(d > 0)
		{
		//	cout << "Go Right" << endl;

		}
		else if (d < 0)
		{
		//	cout << "Go Left" << endl;
		}
	}

	}


	// line(src,Point(0.3*c,0.4*r),Point(0.3*c,0.8*r),Scalar(0,255,100),1,1);
	//
	// Point pika = lineintersection( Point(0.55*c,0.7*r), Point(0.8*c,0.7*r), Point(0.3*c,0.4*r), Point(0.3*c,0.8*r));
	//
	// circle(src, pika, 10, Scalar(255,57,98),-1);
	if(show == 1)
	imshow("myROI",myROI);

	//imshow("Hough Output",src);

}


void pedestrian_detect(Mat img)
{



	 		if(show == 1)
	 			imshow("input frame",img);

	// 		resize(img, img, Size(), 0.5, 0.5, INTER_LINEAR);

	// 		if(show == 1)
	// 			imshow("resized frame",img);

	// 		cvtColor(img,img,CV_BGR2GRAY);

	        vector<Rect> found, found_filtered;
	        hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 2);

	 		if(peds == 1)
	 		{

	 			cout<<"Number of Pedestrians :"<<found.size()<<endl;

	 		}

	        size_t i, j;
	        for (i=0; i<found.size(); i++)
	        {
	            Rect r = found[i];
	            for (j=0; j<found.size(); j++)
	                if (j!=i && (r & found[j])==r)
	                    break;
	            if (j==found.size())
	                found_filtered.push_back(r);
	        }
	        for (i=0; i<found_filtered.size(); i++)
	        {
		    Rect r = found_filtered[i];
	            r.x += cvRound(r.width*0.1);
		    r.width = cvRound(r.width*0.8);
		    r.y += cvRound(r.height*0.06);
		    r.height = cvRound(r.height*0.9);

			if(box == 1)
			{
		    	rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,100), 2);
		    }
			}

	 //       imshow("video capture", img);
	  /*
	        sprintf(file_name,"frame%04d.pgm",frame_count);

			frame_count++;

			imwrite(file_name,gray);
	    */
}


void roadsign_detect( Mat frame )
{

	std::vector<Rect> cars;
	Mat frame_gray;

	float hypot;

	cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

//	imshow("hist",frame_gray);

	  stop_cascade.detectMultiScale( frame_gray, cars, 1.1, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

		for( size_t i = 0; i < cars.size(); i++ )
		{
			Point center( cars[i].x + cars[i].width*0.5, cars[i].y + cars[i].height*0.5 );
			//ellipse( frame, center, Size( cars[i].width*0.5, cars[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 0 ), 1, 8, 0 );

			hypot = sqrt((cars[i].width)^2 + (cars[i].height)^2);

//			cout<<"H = "<<hypot<<endl;
			if(box == 1){
			if(hypot>=12)
			{
				rectangle( frame, cars[i].tl(), cars[i].br(), cv::Scalar(0,0,255), 2);
			}
			else if(hypot>= 5)
			{
				rectangle( frame, cars[i].tl(), cars[i].br(), cv::Scalar(0,255,255), 2);
			}
			else
			{
				rectangle( frame, cars[i].tl(), cars[i].br(), cv::Scalar(0,255,100), 2);
			}
			putText( frame, "STOP", Point( cars[i].x, (cars[i].y)-5), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0,255,100), 2);
		}
	}
	if(show == 1)
	imshow("out",frame);
}
