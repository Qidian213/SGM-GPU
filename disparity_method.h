#ifndef DISPARITY_METHOD_H_
#define DISPARITY_METHOD_H_

#include <stdint.h>
#include <opencv2/opencv.hpp>
#include "util.h"
#include "configuration.h"
#include "costs.h"
#include "hamming_cost.h"
#include "median_filter.h"
#include "cost_aggregation.h"
#include <vector>

//void init_disparity_method(const uint8_t _p1ud, const uint8_t _p2ud,const uint8_t _p1lr, const uint8_t _p2lr,const uint8_t _p1du, const uint8_t _p2du,const uint8_t _p1rl, const uint8_t _p2rl) ;
void disparity_errors(cv::Mat estimation, cv::Mat gt_disp, int *n, int *n_err);
void init_disparity_method(bool &first_alloc); // p1ud, p2ud, p1lr, p2lr, p1du, p2du, p1rl, p2rl

cv::Mat compute_disparity_method(cv::Mat left, cv::Mat right, uint8_t *&d_mcost,uint8_t *pparams,bool &first_alloc);
void finish_disparity_method(bool &first_alloc);
static void free_memory();
                               
                               
#endif /* DISPARITY_METHOD_H_ */
