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

#define NUM_THREADS (5)
#define NUM_CPUS (1)


#define NSEC_PER_SEC (1000000000)
#define NSEC_PER_MSEC (1000000)
#define NSEC_PER_MICROSEC (1000)
#define DELAY_TICKS (1)
#define ERROR (-1)
#define OK (0)


bool peds = 0;
bool cars = 0;
bool signs = 0;
bool lanes = 0;
bool box = 0;
bool show = 0;

using namespace cv;
using namespace std;

typedef struct
{
    int threadIdx;
} threadParams_t;


// POSIX thread declarations and scheduling attributes
//
pthread_t threads[NUM_THREADS];
threadParams_t threadParams[NUM_THREADS];
pthread_attr_t rt_sched_attr[NUM_THREADS];
int rt_max_prio, rt_min_prio;
struct sched_param rt_param[NUM_THREADS];
struct sched_param main_param;
pthread_attr_t main_attr;
pid_t mainpid;

int numberOfProcessors;


sem_t sem_cap_lane,sem_cap_veh,sem_cap_sig,sem_cap_ped,sem_lane,sem_veh,sem_sig,sem_ped;

char input_file_name[30];
char output_file_name[20];

int frame_count = 1;

VideoCapture cap;
Mat global_frame;

String cars_cascade_name = "cars_my2.xml";
CascadeClassifier cars_cascade;

String stop_cascade_name = "stop.xml";
CascadeClassifier stop_cascade;

HOGDescriptor hog;

void print_scheduler(void)
{
   int schedType;

   schedType = sched_getscheduler(getpid());

   switch(schedType)
   {
     case SCHED_FIFO:
           printf("Pthread Policy is SCHED_FIFO\n");
           break;
     case SCHED_OTHER:
           printf("Pthread Policy is SCHED_OTHER\n");
       break;
     case SCHED_RR:
           printf("Pthread Policy is SCHED_OTHER\n");
           break;
     default:
       printf("Pthread Policy is UNKNOWN\n");
   }

}


int delta_t(struct timespec *stop, struct timespec *start, struct timespec *delta_t)
{
  int dt_sec=stop->tv_sec - start->tv_sec;
  int dt_nsec=stop->tv_nsec - start->tv_nsec;

  if(dt_sec >= 0)
  {
    if(dt_nsec >= 0)
    {
      delta_t->tv_sec=dt_sec;
      delta_t->tv_nsec=dt_nsec;
    }
    else
    {
      delta_t->tv_sec=dt_sec-1;
      delta_t->tv_nsec=NSEC_PER_SEC+dt_nsec;
    }
  }
  else
  {
    if(dt_nsec >= 0)
    {
      delta_t->tv_sec=dt_sec;
      delta_t->tv_nsec=dt_nsec;
    }
    else
    {
      delta_t->tv_sec=dt_sec-1;
      delta_t->tv_nsec=NSEC_PER_SEC+dt_nsec;
    }
  }

  return(1);
}


Point lineintersection( Point A, Point B, Point C, Point D);

float slope( Point A, Point B);

void detectAndDisplay( Mat frame );

void lanedetect( Mat src);

void pedestrian_detect(Mat img);

void roadsign_detect( Mat frame );

void *thread0(void *threadp)
{

  int sum=0, i, cpucore;
  pthread_t thread;
  cpu_set_t cpuset;
  struct timespec start_time_0 = {0, 0};
  struct timespec finish_time_0 = {0, 0};
  struct timespec thread_dt_0 = {0, 0};
  threadParams_t *threadParams = (threadParams_t *)threadp;


  thread=pthread_self();
  cpucore=sched_getcpu();

  CPU_ZERO(&cpuset);
  pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);


  cout<<"<Thread 0 Starting>"<< endl;
//    for(int i=0;i<1000000000;i++);

VideoCapture cap(input_file_name);

cap >> global_frame;

int xi;

clock_gettime(CLOCK_REALTIME, &start_time_0);


while(1)
{

  //cout<<"waiting cap"<<endl;
  sem_wait(&sem_cap_lane);
  sem_wait(&sem_cap_veh);
  sem_wait(&sem_cap_sig);
  sem_wait(&sem_cap_ped);
  //cout<<xi<<endl;

  xi++;

  cap >> global_frame;

//  imshow("global frame",global_frame);

//  waitKey(1);
  //
  // Mat frame;
  //
  // frame = global_frame;
  //
  // lanedetect(frame);


  //cout<<"posting from cap"<<endl;
  sem_post(&sem_lane);
  sem_post(&sem_veh);
  sem_post(&sem_sig);
  sem_post(&sem_ped);

  // imshow("Vehicles",frame);
  //   char c=waitKey(1);
  //   if(c == 27)	break;
}

clock_gettime(CLOCK_REALTIME, &finish_time_0);
delta_t(&finish_time_0, &start_time_0, &thread_dt_0);


    printf("Thread 0 finished \n");


/*
        printf("\nThread idx=%d, sum[0...%d]=%d\n",
               threadParams->threadIdx,
               ((threadParams->threadIdx)+1)*100, sum);

        printf("Thread idx=%d ran on core=%d, affinity contained:", threadParams->threadIdx, cpucore);
        for(i=0; i<numberOfProcessors; i++)
            if(CPU_ISSET(i, &cpuset))  printf(" CPU-%d ", i);
        printf("\n");
*/
/*
printf("Thread idx= 0 ran on core=%d, affinity contained:",  cpucore);
for(i=0; i<numberOfProcessors; i++)
		if(CPU_ISSET(i, &cpuset))  printf(" CPU-%d ", i);
printf("\n");
*/


  //      printf("\nThread idx=%d ran %ld sec, %ld msec (%ld microsec)\n", threadParams->threadIdx, thread_dt.tv_sec, (thread_dt.tv_nsec / NSEC_PER_MSEC), (thread_dt.tv_nsec / NSEC_PER_MICROSEC));

  //      pthread_exit(&sum);

}

void *thread1(void *threadp)
{

    int sum=0, i, cpucore;
    pthread_t thread;
    cpu_set_t cpuset;
    struct timespec start_time_1 = {0, 0};
    struct timespec finish_time_1 = {0, 0};
    struct timespec thread_dt_1 = {0, 0};
    threadParams_t *threadParams = (threadParams_t *)threadp;


    thread=pthread_self();
    cpucore=sched_getcpu();

    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);


    cout<<"<Thread 1 Starting>"<<endl;



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


    clock_gettime(CLOCK_REALTIME, &start_time_1);

int xi=0;

    while(1)
    {

      //cout<<"waiting lane"<<endl;
      sem_wait(&sem_lane);

      //cout<<"<T1>"<<xi<<endl;

      xi++;

    	frame = global_frame;

      //global_frame.copyTo(frame);

    //	detectAndDisplay( frame );

      if(lanes == 1)
    	 lanedetect(frame);

    //	pedestrian_detect(frame);

    //	roadsign_detect(frame);

  //	imshow("Final Output", global_frame);

if( peds == 0 && cars == 0 && signs == 0 && lanes == 1)
{
      imshow("Lanes",frame);
    	  char c=waitKey(1);
    	  if(c == 27)	break;
}
      //cout<<"posting cap from lane"<<endl;
      sem_post(&sem_cap_lane);

    }


    clock_gettime(CLOCK_REALTIME, &finish_time_1);
    delta_t(&finish_time_1, &start_time_1, &thread_dt_1);

    destroyAllWindows();

/*
     printf("\nThread idx= pika ran %ld sec, %ld msec (%ld microsec)\n", 	 thread_dt_1.tv_sec, (thread_dt_1.tv_nsec / NSEC_PER_MSEC), (thread_dt_1.tv_nsec / NSEC_PER_MICROSEC));

    float frate = 1000000/((thread_dt_1.tv_sec * 1000) + (thread_dt_1.tv_nsec / NSEC_PER_MSEC));

    printf("\n Frame rate = %f",frate);



    printf("Thread 1 finished \n");

/*
            printf("\nThread idx=%d, sum[0...%d]=%d\n",
                   threadParams->threadIdx,
                   ((threadParams->threadIdx)+1)*100, sum);

    printf("Thread idx= 1 ran on core=%d, affinity contained:", cpucore);
    for(i=0; i<numberOfProcessors; i++)
        if(CPU_ISSET(i, &cpuset))  printf(" CPU-%d ", i);
    printf("\n");
*/

    pthread_exit(&sum);

}

void *thread2(void *threadp)
{

    int sum=0, i, cpucore;
    pthread_t thread;
    cpu_set_t cpuset;
    struct timespec start_time = {0, 0};
    struct timespec finish_time = {0, 0};
    struct timespec thread_dt = {0, 0};
    threadParams_t *threadParams = (threadParams_t *)threadp;

    clock_gettime(CLOCK_REALTIME, &start_time);

    thread=pthread_self();
    cpucore=sched_getcpu();

    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);

    cout<<"<Thread 2 Starting>"<<endl;

    Mat frame;

    int xi;

    while(1)
    {

      //cout<<"waiting veh"<<endl;
      sem_wait(&sem_veh);

      //cout<<"<T2>"<<xi<<endl;

      xi++;

      frame = global_frame;

      //global_frame.copyTo(frame);

      if(cars == 1)
    	 detectAndDisplay( frame );

    //  lanedetect(frame);

    //	pedestrian_detect(frame);

    //	roadsign_detect(frame);

    //	imshow("Final Output", global_frame);

if(peds == 0 && cars == 1)
{
      imshow("Vehicles",frame);
        char c=waitKey(1);
        if(c == 27)	break;
}
      //cout<<"posting cap from veh"<<endl;
      sem_post(&sem_cap_veh);
    }

    destroyAllWindows();

    printf("Thread 2 finished \n");


/*
            printf("\nThread idx=%d, sum[0...%d]=%d\n",
                   threadParams->threadIdx,
                   ((threadParams->threadIdx)+1)*100, sum);
*/
/*
            printf("Thread idx= 2 ran on core=%d, affinity contained:", cpucore);
            for(i=0; i<numberOfProcessors; i++)
                if(CPU_ISSET(i, &cpuset))  printf(" CPU-%d ", i);
            printf("\n");

            clock_gettime(CLOCK_REALTIME, &finish_time);
            delta_t(&finish_time, &start_time, &thread_dt);

            printf("\nThread idx= 2 ran %ld sec, %ld msec (%ld microsec)\n", thread_dt.tv_sec, (thread_dt.tv_nsec / NSEC_PER_MSEC), (thread_dt.tv_nsec / NSEC_PER_MICROSEC));
*/
            pthread_exit(&sum);

}

void *thread3(void *threadp)
{


    int sum=0, i, cpucore;
    pthread_t thread;
    cpu_set_t cpuset;
    struct timespec start_time = {0, 0};
    struct timespec finish_time = {0, 0};
    struct timespec thread_dt = {0, 0};
    threadParams_t *threadParams = (threadParams_t *)threadp;

    clock_gettime(CLOCK_REALTIME, &start_time);

    thread=pthread_self();
    cpucore=sched_getcpu();

    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);


    cout<<"<Thread 3 Starting>"<<endl;

    Mat frame;

    int xi;

    while(1)
    {

      //cout<<"waiting signs"<<endl;
      sem_wait(&sem_sig);

      //cout<<"<T3 >"<<xi<<endl;

      xi++;

      frame = global_frame;

      //global_frame.copyTo(frame);

    //  detectAndDisplay( frame );

    //  lanedetect(frame);

    //	pedestrian_detect(frame);

      if(signs == 1)
    	 roadsign_detect(frame);

    //	imshow("Final Output", global_frame);

if(cars == 0 &&  peds == 0 && signs == 1)
{
      imshow("Vehicles",frame);
        char c=waitKey(1);
        if(c == 27)	break;
}

      //cout<<"posting cap from signs"<<endl;
      sem_post(&sem_cap_sig);
    }

    destroyAllWindows();

    printf("Thread 3 finished \n");


/*
            printf("\nThread idx=%d, sum[0...%d]=%d\n",
                   threadParams->threadIdx,
                   ((threadParams->threadIdx)+1)*100, sum);
*/
/*
            printf("Thread idx= 3 ran on core=%d, affinity contained:", cpucore);
            for(i=0; i<numberOfProcessors; i++)
                if(CPU_ISSET(i, &cpuset))  printf(" CPU-%d ", i);
            printf("\n");

            clock_gettime(CLOCK_REALTIME, &finish_time);
            delta_t(&finish_time, &start_time, &thread_dt);

            printf("\nThread idx= 3 ran %ld sec, %ld msec (%ld microsec)\n", thread_dt.tv_sec, (thread_dt.tv_nsec / NSEC_PER_MSEC), (thread_dt.tv_nsec / NSEC_PER_MICROSEC));
*/
            pthread_exit(&sum);

}


void *thread4(void *threadp)
{

  int sum=0, i, cpucore;
  pthread_t thread;
  cpu_set_t cpuset;
  struct timespec start_time = {0, 0};
  struct timespec finish_time = {0, 0};
  struct timespec thread_dt = {0, 0};
  threadParams_t *threadParams = (threadParams_t *)threadp;

  clock_gettime(CLOCK_REALTIME, &start_time);

  thread=pthread_self();
  cpucore=sched_getcpu();

  CPU_ZERO(&cpuset);
  pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);

  cout<<"<Thread 4 Starting>"<<endl;

  Mat frame;

  int xi;

  while(1)
  {

    //cout<<"waiting ped"<<endl;
    sem_wait(&sem_ped);

    //cout<<"<T4 >"<<xi<<endl;

    xi++;

    frame = global_frame;

    //global_frame.copyTo(frame);

  //  detectAndDisplay( frame );

  //  lanedetect(frame);

    if(peds == 1)
  	   pedestrian_detect(frame);

  //  roadsign_detect(frame);

  //	imshow("Final Output", global_frame);

if(peds == 1)
{
    imshow("Vehicles",frame);
      char c=waitKey(1);
      if(c == 27)	break;
}

    //cout<<"posting cap from ped"<<endl;
    sem_post(&sem_cap_ped);
  }

  destroyAllWindows();

  printf("Thread 4 finished \n");


/*
          printf("\nThread idx=%d, sum[0...%d]=%d\n",
                 threadParams->threadIdx,
                 ((threadParams->threadIdx)+1)*100, sum);
*/

/*
          printf("Thread idx= 3 ran on core=%d, affinity contained:", cpucore);
          for(i=0; i<numberOfProcessors; i++)
              if(CPU_ISSET(i, &cpuset))  printf(" CPU-%d ", i);
          printf("\n");

          clock_gettime(CLOCK_REALTIME, &finish_time);
          delta_t(&finish_time, &start_time, &thread_dt);

          printf("\nThread idx= 3 ran %ld sec, %ld msec (%ld microsec)\n", thread_dt.tv_sec, (thread_dt.tv_nsec / NSEC_PER_MSEC), (thread_dt.tv_nsec / NSEC_PER_MICROSEC));
*/
          pthread_exit(&sum);


}

int main(int argc, const char * argv[])
{

  sem_init(&sem_cap_lane, 0, 1);
  sem_init(&sem_cap_veh, 0, 1);
  sem_init(&sem_cap_sig, 0, 1);
  sem_init(&sem_cap_ped, 0, 1);
  sem_init(&sem_lane, 0, 0);
  sem_init(&sem_veh, 0 ,0);
  sem_init(&sem_sig, 0, 0);
  sem_init(&sem_ped, 0, 0);

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
	}

  //input_file_name = malloc(sizeof(char)*30);

  //input_file_name = argv[1];

  strcpy(input_file_name,argv[1]);

	int rc;
	int i, scope, idx;
	cpu_set_t allcpuset;
	cpu_set_t threadcpu;
	int coreid;

	printf("This system has %d processors configured and %d processors available.\n", get_nprocs_conf(), get_nprocs());

	numberOfProcessors = get_nprocs_conf();
	printf("number of CPU cores=%d\n", numberOfProcessors);

	CPU_ZERO(&allcpuset);

	for(i=0; i < numberOfProcessors; i++)
			CPU_SET(i, &allcpuset);

	if(numberOfProcessors >= NUM_CPUS)
			printf("Using sysconf number of CPUS=%d, count in set=%d\n", numberOfProcessors, CPU_COUNT(&allcpuset));
	else
	{
			numberOfProcessors=NUM_CPUS;
			printf("Using DEFAULT number of CPUS=%d\n", numberOfProcessors);
	}

	mainpid=getpid();

	rt_max_prio = sched_get_priority_max(SCHED_FIFO);
	rt_min_prio = sched_get_priority_min(SCHED_FIFO);

	print_scheduler();
	rc=sched_getparam(mainpid, &main_param);
	main_param.sched_priority=rt_max_prio;
	rc=sched_setscheduler(getpid(), SCHED_FIFO, &main_param);
	if(rc < 0) perror("main_param");
	print_scheduler();


	pthread_attr_getscope(&main_attr, &scope);

	if(scope == PTHREAD_SCOPE_SYSTEM)
		printf("PTHREAD SCOPE SYSTEM\n");
	else if (scope == PTHREAD_SCOPE_PROCESS)
		printf("PTHREAD SCOPE PROCESS\n");
	else
		printf("PTHREAD SCOPE UNKNOWN\n");

	printf("rt_max_prio=%d\n", rt_max_prio);
	printf("rt_min_prio=%d\n", rt_min_prio);


	   for(i=0; i < NUM_THREADS; i++)
	   {
	       CPU_ZERO(&threadcpu);
	       coreid=i%numberOfProcessors;
	       printf("Setting thread %d to core %d\n", i, coreid);
	       CPU_SET(coreid, &threadcpu);
	       for(idx=0; idx<numberOfProcessors; idx++)
	           if(CPU_ISSET(idx, &threadcpu))  printf(" CPU-%d ", idx);
	       printf("\nLaunching thread %d\n", i);

	       rc=pthread_attr_init(&rt_sched_attr[i]);
	       rc=pthread_attr_setinheritsched(&rt_sched_attr[i], PTHREAD_EXPLICIT_SCHED);
	     //  rc=pthread_attr_setschedpolicy(&rt_sched_attr[i], SCHED_FIFO);
	       rc=pthread_attr_setaffinity_np(&rt_sched_attr[i], sizeof(cpu_set_t), &threadcpu);

	       rt_param[i].sched_priority=rt_max_prio-i-1;
	       pthread_attr_setschedparam(&rt_sched_attr[i], &rt_param[i]);

	       threadParams[i].threadIdx=i;
/*
	       if(i == 0)
	       {

	       pthread_create(&threads[i],   // pointer to thread descriptor
	                      (void *)0,     // use default attributes
	                      thread0, // thread function entry point
	                      (void *)&(threadParams[i]) // parameters to pass in
	                     );


	       }

	       if(i == 1)
	       {
	       pthread_create(&threads[i],   // pointer to thread descriptor
	                      (void *)0,     // use default attributes
	                      thread1, // thread function entry point
	                      (void *)&(threadParams[i]) // parameters to pass in
	                    );
	       }

	       if(i == 2)
	       {
	       pthread_create(&threads[i],   // pointer to thread descriptor
	                      (void *)0,     // use default attributes
	                      thread2, // thread function entry point
	                      (void *)&(threadParams[i]) // parameters to pass in
	                     );
	       }

	       if(i == 3)
	       {
	       pthread_create(&threads[i],   // pointer to thread descriptor
	                      (void *)0,     // use default attributes
	                      thread3, // thread function entry point
	                      (void *)&(threadParams[i]) // parameters to pass in
	                     );
	       }
*/
}

		pthread_create(&threads[0], &rt_sched_attr[0], thread0, NULL);
		pthread_create(&threads[1], &rt_sched_attr[1], thread1, NULL);
		pthread_create(&threads[2], &rt_sched_attr[2], thread2, NULL);
		pthread_create(&threads[3], &rt_sched_attr[3], thread3, NULL);
    pthread_create(&threads[4], &rt_sched_attr[4], thread4, NULL);

	   // for(i=0;i<NUM_THREADS;i++)
	   //     pthread_join(threads[i], NULL);

pthread_join(threads[0], NULL);
pthread_join(threads[1], NULL);
pthread_join(threads[2], NULL);
pthread_join(threads[3], NULL);
pthread_join(threads[4], NULL);

//cout <<"Finished"<<endl;


/*


	VideoCapture cap(argv[1]);

	Mat frame;

	if( !cars_cascade.load( cars_cascade_name ) )
	{
		 printf("Error loading cars cascade\n");
		 return -1;
	 }


	while(1)
	{

		cap >> frame;

		detectAndDisplay( frame );

		lanedetect(frame);

		imshow("Final Output", frame);

		char c=waitKey(1);
		if(c == 27)	break;
	}*/
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
