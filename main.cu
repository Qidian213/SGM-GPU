#include <iostream>
#include <numeric>
#include <sys/time.h>
#include <vector>
#include <stdlib.h>
#include <typeinfo>
#include <opencv2/opencv.hpp>

#include <numeric>
#include <stdlib.h>
#include <ctime>
#include <sys/types.h>
#include <stdint.h>
#include <linux/limits.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include "disparity_method.h"

static uint8_t *pparams; 

int main(int argc, char *argv[]) 
{
    bool first_alloc = true;
    uint8_t *d_mcost;
	init_disparity_method(first_alloc);

    cv::Mat h_im0 = cv::imread("../left.png");
    cv::Mat h_im1 = cv::imread("../right.png");
    		// Convert images to grayscale
	if (h_im0.channels()>1) 
	{
		cv::cvtColor(h_im0, h_im0, CV_RGB2GRAY);
	}

	if (h_im1.channels()>1) 
	{
		cv::cvtColor(h_im1, h_im1, CV_RGB2GRAY);
	}
	
	if(h_im0.rows % 4 != 0 || h_im0.cols % 4 != 0) {
	    int rows_s = h_im0.rows- (h_im0.rows % 4);
	    int cols_s = h_im0.cols- (h_im0.cols % 4);
	    cv::resize(h_im1,h_im1,cv::Size(cols_s,rows_s));
	    cv::resize(h_im0,h_im0,cv::Size(cols_s,rows_s));
	}

	pparams = new uint8_t[h_im0.rows*h_im0.cols*8];
	for(int i=0; i< h_im0.rows*h_im0.cols*8 ;i +=2 )
	{
	    pparams[i] = 5;
	    pparams[i+1] = 86;
	}
	cv::Mat disparity_im = compute_disparity_method(h_im0, h_im1, d_mcost,pparams,first_alloc);

/////write file
    const int type = disparity_im.type();
    const uchar depth = type & CV_MAT_DEPTH_MASK;
	if(depth == CV_8U) 
	{
		cv::imwrite("disp.png", disparity_im);
	} else 
	{
		cv::Mat disparity16(disparity_im.rows, disparity_im.cols, CV_16UC1);
		for(int i = 0; i < disparity_im.rows; i++) 
		{
			for(int j = 0; j < disparity_im.cols; j++) 
			{
				const float d = disparity_im.at<float>(i, j)*256.0f;
				disparity16.at<uint16_t>(i, j) = (uint16_t) d;
			}
		}
		cv::imwrite("disp.png", disparity16);
	}

	finish_disparity_method(first_alloc);

	return 0;
}

