#include "disparity_method.h"

static cudaStream_t stream1, stream2, stream3;//, stream4, stream5, stream6, stream7, stream8;
static uint8_t *d_im0;
static uint8_t *d_im1;
static cost_t *d_transform0;
static cost_t *d_transform1;
static uint8_t *d_cost;
static uint8_t *d_disparity;
static uint8_t *d_disparity_filtered_uchar;
static uint8_t *h_disparity;
static uint8_t *d_mmcost;
static uint16_t *d_S;
static uint8_t *d_L0;
static uint8_t *d_L1;
static uint8_t *d_L2;
static uint8_t *d_L3;
static uint32_t cols, rows, size, size_cube_l,size_ppparams;
static uint8_t *pparamsgpu; 

void disparity_errors(cv::Mat estimation, cv::Mat gt_disp, int *n, int *n_err)
{
	int nlocal = 0;
	int nerrlocal = 0;
	
	if(!gt_disp.data) {
		std::cerr << "Couldn't read the gt_disp file " << std::endl;
		exit(EXIT_FAILURE);
	}
	if(estimation.rows != gt_disp.rows || estimation.cols != gt_disp.cols) {
		std::cerr << "Ground truth must have the same dimesions" << std::endl;
		exit(EXIT_FAILURE);
	}
	const int type = estimation.type();
	const uchar depth = type & CV_MAT_DEPTH_MASK;
	for(int i = 0; i < gt_disp.rows; i++) {
		for(int j = 0; j < gt_disp.cols; j++) {
			const uint16_t gt = gt_disp.at<uint16_t>(i, j);
			if(gt > 0) {
				const float gt_f = ((float)gt)/256.0f;
				float est;
				if(depth == CV_8U) {
					est = (float) estimation.at<uint8_t>(i, j);
				} else {
					est = estimation.at<float>(i, j);
				}
				const float err = fabsf(est-gt_f);
				const float ratio = err/fabsf(gt_f);
				if(err > ABS_THRESH && ratio > REL_THRESH) {
					nerrlocal++;
				}
				nlocal++;
			}
		}
	}
	*n += nlocal;
	*n_err += nerrlocal;
}

void init_disparity_method(bool &first_alloc)
{
	// Create streams
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream1));
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream2));
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream3));
	first_alloc = true;
    rows = 0;
    cols = 0;
}

cv::Mat compute_disparity_method(cv::Mat left, cv::Mat right, uint8_t *&d_mcost,uint8_t *pparams,bool &first_alloc)
{
	if(cols != left.cols || rows != left.rows)
	 {
		if(!first_alloc) 
		{
			free_memory();
		}
		first_alloc = false;
		cols = left.cols;
		rows = left.rows;
		size = rows*cols;
		size_cube_l = size*MAX_DISPARITY;    ///// MAX_DISPARITY =128
		size_ppparams = size*8;
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_transform0, sizeof(cost_t)*size));

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_transform1, sizeof(cost_t)*size));

		int size_cube = size*MAX_DISPARITY;
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_cost, sizeof(uint8_t)*size_cube));
        d_mcost = d_cost;
        
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_im0, sizeof(uint8_t)*size));

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_im1, sizeof(uint8_t)*size));

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_S, sizeof(uint16_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L0, sizeof(uint8_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L1, sizeof(uint8_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L2, sizeof(uint8_t)*size_cube_l));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L3, sizeof(uint8_t)*size_cube_l));

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity, sizeof(uint8_t)*size));
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disparity_filtered_uchar, sizeof(uint8_t)*size));
		
		CUDA_CHECK_RETURN(cudaMalloc((void **)&pparamsgpu, sizeof(uint8_t)*size_ppparams));
		h_disparity = new uint8_t[size];
		d_mmcost = new uint8_t[size_cube_l];
	}
	
	CUDA_CHECK_RETURN(cudaMemcpyAsync(pparamsgpu, pparams, sizeof(uint8_t)*size_ppparams, cudaMemcpyHostToDevice, stream1));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im0, left.ptr<uint8_t>(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im1, right.ptr<uint8_t>(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice, stream1));

	dim3 block_size;
	block_size.x = 32;
	block_size.y = 32;

	dim3 grid_size;
	grid_size.x = (cols+block_size.x-1) / block_size.x;
	grid_size.y = (rows+block_size.y-1) / block_size.y;

	CenterSymmetricCensusKernelSM2<<<grid_size, block_size, 0, stream1>>>(d_im0, d_im1, d_transform0, d_transform1, rows, cols);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
	{
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

	// Hamming distance
	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream1));
	HammingDistanceCostKernel<<<rows, MAX_DISPARITY, 0, stream1>>>(d_transform0, d_transform1, d_cost, rows, cols);
	err = cudaGetLastError();
	if (err != cudaSuccess) 
	{
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

	// Cost Aggregation
	const int PIXELS_PER_BLOCK = COSTAGG_BLOCKSIZE/WARP_SIZE; ////   64/32
	const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ/WARP_SIZE;

	CostAggregationKernelLeftToRight<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream2>>>(d_cost, d_L0, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, pparamsgpu);
	err = cudaGetLastError();
	if (err != cudaSuccess) 
	{
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

	CostAggregationKernelRightToLeft<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream3>>>(d_cost, d_L1,rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, pparamsgpu);
	err = cudaGetLastError();
	if (err != cudaSuccess) 
	{
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

	CostAggregationKernelUpToDown<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, pparamsgpu);
	err = cudaGetLastError();
	if (err != cudaSuccess) 
	{
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	CostAggregationKernelDownToUp<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(d_cost, d_L3, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, pparamsgpu);
	err = cudaGetLastError();
	if (err != cudaSuccess) 
	{
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

	MedianFilter3x3<<<(size+MAX_DISPARITY-1)/MAX_DISPARITY, MAX_DISPARITY, 0, stream1>>>(d_disparity, d_disparity_filtered_uchar, rows, cols);
	err = cudaGetLastError();
	if (err != cudaSuccess) 
	{
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}

	CUDA_CHECK_RETURN(cudaMemcpy(h_disparity, d_disparity_filtered_uchar, sizeof(uint8_t)*size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(d_mmcost,d_cost, sizeof(uint8_t)*size_cube_l, cudaMemcpyDeviceToHost));
    d_mcost = d_mmcost;
	cv::Mat disparity(rows, cols, CV_8UC1, h_disparity);
	return disparity;
}

static void free_memory() 
{
	CUDA_CHECK_RETURN(cudaFree(d_im0));
	CUDA_CHECK_RETURN(cudaFree(d_im1));
	CUDA_CHECK_RETURN(cudaFree(d_transform0));
	CUDA_CHECK_RETURN(cudaFree(d_transform1));
	CUDA_CHECK_RETURN(cudaFree(d_L0));
	CUDA_CHECK_RETURN(cudaFree(d_L1));
	CUDA_CHECK_RETURN(cudaFree(d_L2));
	CUDA_CHECK_RETURN(cudaFree(d_L3));
	CUDA_CHECK_RETURN(cudaFree(d_disparity));
	CUDA_CHECK_RETURN(cudaFree(d_disparity_filtered_uchar));
	CUDA_CHECK_RETURN(cudaFree(d_cost));

	delete[] h_disparity;
}

void finish_disparity_method(bool &first_alloc) 
{
	if(!first_alloc) 
	{
		free_memory();
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream1));
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream2));
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream3));
	}
}
