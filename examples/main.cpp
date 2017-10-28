#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/imgcodecs.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "memoryAccessor.h"
#include "streamCreator.h"
using namespace cv;
using namespace std;
using namespace cv::cuda;


enum AllocTypes{
	PINNED = 1,
	UNIFIED = 2,
	NORMAL = 3
};

template <typename T>
int debug(T value, bool isDebug){
	if(isDebug){
		cout << value << endl;
	}
	return 0;
}
int testConcurrentCannyVideo(AllocTypes alloctype, VideoCapture cap, int itrNum, bool isRender, Mat source, Mat dest, bool isDebug){
	GpuMat gSource[10];
	GpuMat gray[10];
	GpuMat gDest[10];

	Mat sources[10];
		
	cudaStream_t streamCUDA[10];
	int streamNum = 10;
	cv::cuda::Stream stream[10];
	

	for(int i=0; i<streamNum; i++){
		streamCUDA[i] = createStreamWithFlags();
		stream[i] = cv::cuda::StreamAccessor::wrapStream(streamCUDA[i]);
	}
	if(alloctype == PINNED){
		for(int i=0; i<10; i++){
			gSource[i] = GpuMat(source.size(), CV_8UC3);
			gray[i] = GpuMat(source.size(), CV_8UC1);
			gDest[i] = GpuMat(source.size(), CV_8UC1);
		
		}


		
		for(int i=0; i<itrNum; i++){
			cap >> source;
			for(int j=0; j<10; j++){
				gSource[j].upload(source, stream[j] );
				cuda::cvtColor(gSource[j], gray[j], CV_BGR2GRAY, 0, stream[j] );
				Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
				canny->detect( gray[j], gDest[j], stream[j]  );
				gDest[j].download(dest, stream[j]);
			}
			if(isRender){
				imshow("Video", dest);
				waitKey(30);
			}
					
		}	
	
	}
	else if(alloctype == UNIFIED){

		/*
		for(;;){
			cap >> source;
			cuda::cvtColor(gSource, gray, CV_BGR2GRAY, 0, stream[0]);
			Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
			canny->detect( gray, gDest, stream[1] );
		
			if(isRender){
				imshow("Video", dest);
				waitKey(30);
			}
			stream[1].waitForCompletion();				
		}
		*/
			
	
	}
	
	return 0;
}

int testConcurrentCannyImage(AllocTypes alloctype, int itrNum, bool isRender, Mat source, Mat dest, bool isDebug){
	GpuMat gSource[10];
	GpuMat gray[10];
	GpuMat gDest[10];

	Mat sources[10];
	Mat dests[10];	
	cudaStream_t streamCUDA[10];
	int streamNum = 10;
	cv::cuda::Stream stream[10];
	

	for(int i=0; i<streamNum; i++){
		streamCUDA[i] = createStreamWithFlags();
		stream[i] = cv::cuda::StreamAccessor::wrapStream(streamCUDA[i]);
	}
	if(alloctype == PINNED){
		for(int i=0; i<10; i++){
			source.copyTo(sources[i]);
			dest = Mat(source.size(), CV_8UC1);
			gSource[i] = GpuMat(source.size(), CV_8UC3);
			gray[i] = GpuMat(source.size(), CV_8UC1);
			gDest[i] = GpuMat(source.size(), CV_8UC1);
		
		}

		
		for(int i=0; i<itrNum; i++){
			//for(int j=0; j<10; j++){
				gSource[0].upload(sources[0], stream[0] );
				cuda::cvtColor(gSource[0], gray[0], CV_BGR2GRAY, 0, stream[1] );
				gSource[1].upload(sources[1], stream[0] );
				cuda::cvtColor(gSource[0], gray[0], CV_BGR2GRAY, 0, stream[2] );
				gSource[2].upload(sources[2], stream[5] );
				Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
				canny->detect( gray[0], gDest[0], stream[6]  );
				gSource[3].upload(sources[3], stream[5] );				
				gDest[0].download(dests[0], stream[3]);
				canny->detect( gray[1], gDest[1], stream[4]  );
				gDest[0].download(dests[0], stream[3]);
				
			//}
			if(isRender){
				imshow("Video", dest);
				waitKey(30);
			}
					
		}	
	
	}
	else if(alloctype == UNIFIED){
		

		/*
		for(;;){
			cap >> source;
			cuda::cvtColor(gSource, gray, CV_BGR2GRAY, 0, stream[0]);
			Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
			canny->detect( gray, gDest, stream[1] );
		
			if(isRender){
				imshow("Video", dest);
				waitKey(30);
			}
			stream[1].waitForCompletion();				
		}
		*/
			
	
	}
	
	return 0;
}

int testCannyVideo(AllocTypes alloctype, VideoCapture cap, int itrNum, bool isRender, Mat source, Mat dest, bool isDebug){
	GpuMat gSource;
	GpuMat gray;
	GpuMat gDest;
	
	cudaStream_t streamCUDA[4];
	int streamNum = 4;
	cv::cuda::Stream stream[4];
	for(int i=0; i<streamNum; i++){
		streamCUDA[i] = createStreamWithFlags();
		stream[i] = cv::cuda::StreamAccessor::wrapStream(streamCUDA[i]);
	}
	
	double wholeTime = 0;
	double timeSec = 0;
	if(alloctype == PINNED){
		gSource = GpuMat(source.size(), CV_8UC3);
		gray = GpuMat(source.size(), CV_8UC1);
		gDest = GpuMat(source.size(), CV_8UC1);
		
		for(int i=0; i<itrNum; i++){			
			cap >> source;
			const int64 startWhole = getTickCount();		
			gSource.upload(source);
			
				
	
			const int64 startCvt = getTickCount();
			cuda::cvtColor(gSource, gray, CV_BGR2GRAY, 0);
			timeSec = (getTickCount() - startCvt) / getTickFrequency();

			const int64 startCanny = getTickCount();
			Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
			canny->detect( gray, gDest);
			timeSec = (getTickCount() - startCanny) / getTickFrequency();
			
			//stream[1].waitForCompletion();
			
			gDest.download(dest);
			timeSec = (getTickCount() - startWhole) / getTickFrequency();
			
			wholeTime += timeSec;
			if(isRender){
				imshow("Video", dest);
				waitKey(30);
			}
					
		}

		debug("PINNED", isDebug);
		debug("Whole Time", isDebug);
		debug(wholeTime, isDebug);
		debug("Average", isDebug);
		debug(wholeTime/itrNum, isDebug);		
		debug(1/(wholeTime/itrNum), isDebug);
		
	}
	else if(alloctype == UNIFIED){
		debug(source.size(), isDebug);
		gSource = GpuMat(source.size(), CV_8UC3, source.data);
		gray = GpuMat(source.size(), CV_8UC1);
		gDest = GpuMat(source.size(), CV_8UC1, dest.data);

		for(int i=0; i<itrNum; i++){
			cap >> source;
			const int64 startWhole = getTickCount();	
			const int64 startCvt = getTickCount();			
			cuda::cvtColor(gSource, gray, CV_BGR2GRAY, 0);
			timeSec = (getTickCount() - startCvt) / getTickFrequency();

			const int64 startCanny = getTickCount();			
			Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
			canny->detect( gray, gDest);
			timeSec = (getTickCount() - startCanny) / getTickFrequency();

			timeSec = (getTickCount() - startWhole) / getTickFrequency();
			
			wholeTime += timeSec;					
			if(isRender){
				imshow("Video", dest);
				waitKey(30);
			}
			//stream[1].waitForCompletion();				
		}
		debug("UNIFIED", isDebug);
		debug("Whole Time", isDebug);
		debug(wholeTime, isDebug);
		debug("Average", isDebug);
		debug(wholeTime/itrNum, isDebug);		
		debug(1/(wholeTime/itrNum), isDebug);				
	
	}
	else if(alloctype == NORMAL){
		gSource = GpuMat(source.size(), CV_8UC3);
		gray = GpuMat(source.size(), CV_8UC1);
		gDest = GpuMat(source.size(), CV_8UC1);
		
		for(int i=0; i<itrNum; i++){			
			cap >> source;
			const int64 startWhole = getTickCount();		
			gSource.upload(source);

				
	
			const int64 startCvt = getTickCount();
			cuda::cvtColor(gSource, gray, CV_BGR2GRAY, 0);
			timeSec = (getTickCount() - startCvt) / getTickFrequency();

			const int64 startCanny = getTickCount();
			Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
			canny->detect( gray, gDest);
			timeSec = (getTickCount() - startCanny) / getTickFrequency();
			
			//stream[1].waitForCompletion();
			
			gDest.download(dest);
			timeSec = (getTickCount() - startWhole) / getTickFrequency();
			
			wholeTime += timeSec;
			if(isRender){
				imshow("Video", dest);
				waitKey(30);
			}
					
		}

		debug("NORMAL", isDebug);
		debug("Whole Time", isDebug);
		debug(wholeTime, isDebug);
		debug("Average", isDebug);
		debug(wholeTime/itrNum, isDebug);		
		debug(1/(wholeTime/itrNum), isDebug);	
	}
	
	return 0;
}


int testCannyImage(AllocTypes alloctype, int itrNum, bool isRender, Mat source, Mat dest, bool isDebug){
	GpuMat gSource;
	GpuMat gray;
	GpuMat gDest;
	
	cudaStream_t streamCUDA[4];
	int streamNum = 4;
	cv::cuda::Stream stream[4];
	for(int i=0; i<streamNum; i++){
		streamCUDA[i] = createStreamWithFlags();
		stream[i] = cv::cuda::StreamAccessor::wrapStream(streamCUDA[i]);
	}
	
	double wholeTime = 0;
	double timeSec = 0;
	if(alloctype == PINNED){
		gSource = GpuMat(source.size(), CV_8UC3);
		gray = GpuMat(source.size(), CV_8UC1);
		gDest = GpuMat(source.size(), CV_8UC1);
		
		for(int i=0; i<itrNum; i++){			
			const int64 startWhole = getTickCount();		
			gSource.upload(source);
			
				
	
			const int64 startCvt = getTickCount();
			cuda::cvtColor(gSource, gray, CV_BGR2GRAY, 0);
			timeSec = (getTickCount() - startCvt) / getTickFrequency();

			const int64 startCanny = getTickCount();
			Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
			canny->detect( gray, gDest);
			timeSec = (getTickCount() - startCanny) / getTickFrequency();
			
			//stream[1].waitForCompletion();
			
			gDest.download(dest);
			timeSec = (getTickCount() - startWhole) / getTickFrequency();
			
			wholeTime += timeSec;
			if(isRender){
				imshow("Video", dest);
				waitKey(30);
			}
					
		}

		debug("PINNED", isDebug);
		debug("Whole Time", isDebug);
		debug(wholeTime, isDebug);
		debug("Average", isDebug);
		debug(wholeTime/itrNum, isDebug);		
		debug(1/(wholeTime/itrNum), isDebug);
		
	}
	else if(alloctype == UNIFIED){
		debug(source.size(), isDebug);
		gSource = GpuMat(source.size(), CV_8UC3, source.data);
		gray = GpuMat(source.size(), CV_8UC1);
		gDest = GpuMat(source.size(), CV_8UC1, dest.data);

		for(int i=0; i<itrNum; i++){
			const int64 startWhole = getTickCount();	
			const int64 startCvt = getTickCount();			
			cuda::cvtColor(gSource, gray, CV_BGR2GRAY, 0);
			timeSec = (getTickCount() - startCvt) / getTickFrequency();

			const int64 startCanny = getTickCount();			
			Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
			canny->detect( gray, gDest);
			timeSec = (getTickCount() - startCanny) / getTickFrequency();

			timeSec = (getTickCount() - startWhole) / getTickFrequency();
			
			wholeTime += timeSec;					
			if(isRender){
				imshow("Video", dest);
				waitKey(30);
			}
			//stream[1].waitForCompletion();				
		}

		debug("UNIFIED", isDebug);
		debug("Whole Time", isDebug);
		debug(wholeTime, isDebug);
		debug("Average", isDebug);
		debug(wholeTime/itrNum, isDebug);		
		debug(1/(wholeTime/itrNum), isDebug);				
	
	}
	else if(alloctype == NORMAL){
		gSource = GpuMat(source.size(), CV_8UC3);
		gray = GpuMat(source.size(), CV_8UC1);
		gDest = GpuMat(source.size(), CV_8UC1);
		
		for(int i=0; i<itrNum; i++){			
			const int64 startWhole = getTickCount();		
			gSource.upload(source);
			
				
	
			const int64 startCvt = getTickCount();
			cuda::cvtColor(gSource, gray, CV_BGR2GRAY, 0);
			timeSec = (getTickCount() - startCvt) / getTickFrequency();

			const int64 startCanny = getTickCount();
			Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
			canny->detect( gray, gDest);
			timeSec = (getTickCount() - startCanny) / getTickFrequency();
			
			//stream[1].waitForCompletion();
			
			gDest.download(dest);
			timeSec = (getTickCount() - startWhole) / getTickFrequency();
			
			wholeTime += timeSec;
			if(isRender){
				imshow("Video", dest);
				waitKey(30);
			}
					
		}

		debug("NORMAL", isDebug);
		debug("Whole Time", isDebug);
		debug(wholeTime, isDebug);
		debug("Average", isDebug);
		debug(wholeTime/itrNum, isDebug);		
		debug(1/(wholeTime/itrNum), isDebug);	
	}
	
	return 0;
}


int testVideo(AllocTypes alloctype,const char* videoName, int itrNum, bool isRender, bool isDebug){


	
	VideoCapture cap = VideoCapture(videoName);
	int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT); 
	unsigned char *h_source;
	unsigned char *h_dest;	
	Size sourceSize(width, height);	
	Mat source(sourceSize, CV_8UC3);
	Mat dest(sourceSize, CV_8UC1);
	if(alloctype == PINNED){
			cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));
			//source = Mat(sourceSize, CV_8UC3);
			//dest = Mat(sourceSize, CV_8UC1);
	}	
	else if(alloctype == UNIFIED){
			//cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::UNIFIED));

			cuMallocManaged((void **)&h_source, height, width, 3);
			cuMallocManaged((void **)&h_dest,  height, width, 1);	
			source = Mat(sourceSize, CV_8UC3, h_source);
			dest = Mat(sourceSize, CV_8UC1, h_dest);
	
	}
	
	//testConcurrentCannyVideo(alloctype, cap, itrNum, isRender, source, dest, isDebug);
	testCannyVideo(alloctype, cap, itrNum, isRender, source, dest, isDebug);
	if(alloctype == UNIFIED){
		//cuFree(&(source.data));
	}
	return 0;
}


int testImage(AllocTypes alloctype,const char* imageName, int itrNum, bool isRender, bool isDebug, int width, int height){


	

	unsigned char *h_source;
	unsigned char *h_dest;	
	Size sourceSize(width, height);
	Mat source;
	Mat dest;
	imreadFixed(imageName, &source);

	if(alloctype == PINNED){
			cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));
			source = Mat(sourceSize, CV_8UC3);
			dest = Mat(sourceSize, CV_8UC1);
	}	
	else if(alloctype == UNIFIED){
			//cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::UNIFIED));
			cuMallocManaged((void **)&h_source, height, width, 3);
			cuMallocManaged((void **)&h_dest,  height, width, 1);
			debug(height, isDebug);
			debug(width, isDebug);	
			source = Mat(sourceSize, CV_8UC3, h_source);
			dest = Mat(sourceSize, CV_8UC1, h_dest);

	}
	imreadFixed(imageName, &source);
	//testConcurrentCannyImage(alloctype, itrNum, isRender, source, dest, isDebug);
	testCannyImage(alloctype, itrNum, isRender, source, dest, isDebug);
	
	if(alloctype == UNIFIED){
		//cuFree(source.data);
	}


	return 0;
}

int testMultipleConcurrentStream(){
	return 0;
}
int main(int argc, char** argv){
	
	string videoName[3] = { "NORWAY720P.mp4", "NORWAY1080P.mp4", "NORWAY2K.mp4"};
	string imageName[3] = {"720p.jpg", "1080p.jpg", "4k.jpg"};
	int width[3] = {1280, 1920, 3840};
	int height[3] = {720, 1080, 2160};
	int videoNum = sizeof(videoName)/sizeof(videoName[0]);
	int imageNum = sizeof(imageName)/sizeof(imageName[0]);
	string videoPath = "/home/ubuntu/dev_opencv3/tx1CVMemoryTest/";
	string imagePath = "/home/ubuntu/dev_opencv3/tx1CVMemoryTest/";
	string videoFile;
	string imageFile;
	int iterateNum = 100;
	bool isRender = false;
	bool isDebug = true;
	cv::compileCheck();
	
	int videoIndex = atoi( argv[1]);
	int imageIndex = atoi( argv[2]);
	int allocIndex = atoi(argv[3]);
	int chooseVideo = atoi(argv[4]);
	//testImage(PINNED, "/home/ubuntu/dev_opencv3/tx1CVMemoryTest/4k.jpg", iterateNum, isRender, isDebug, width[2], height[2]);
	//testVideo(PINNED, "/home/ubuntu/dev_opencv3/tx1CVMemoryTest/NORWAY1080P.mp4", 100, false, isDebug);

	AllocTypes alloc;
	debug(allocIndex, isDebug);
	if(allocIndex == 0){
		alloc = UNIFIED;
	}
	else if(allocIndex == 1){
		alloc = PINNED;
	}
	else if(allocIndex == 2){
		
		alloc = NORMAL;
	}
	videoFile = videoPath + videoName[videoIndex];
	debug("########################"+videoName[videoIndex]  + "################################", isDebug);
	if(chooseVideo == 1){
		testVideo(alloc, videoFile.c_str(), iterateNum, isRender, isDebug);
	}
	
	int i= imageIndex;
	imageFile = imagePath + imageName[imageIndex];
	debug("########################"+ imageName[imageIndex]  + "################################", isDebug);
	if(chooseVideo == 0){
		testImage(alloc, imageFile.c_str(), iterateNum, isRender, isDebug, width[i], height[i]);
	}
	

	
	

	
	/*
	Mat mat;
	imreadFixed(imageFile, &mat, 0);
	GpuMat gMat(mat.size(), mat.type(), mat.data);
	Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
    canny->detect( gMat, gMat );
    imshow("test", mat);
    waitKey(0);
	*/
	return 0;

}

