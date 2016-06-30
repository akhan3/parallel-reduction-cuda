/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation and
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution
 * of this software and related documentation without an express license
 * agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

/* Template project which demonstrates the basics on how to setup a project
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// includes, project
#include <cutil_inline.h>

// includes, kernels
#include <findmax_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);

extern "C"
float computeGold(float* idata, const unsigned int N);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    runTest(argc, argv);

//    cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char** argv)
{
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
//     if(cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
//         cutilDeviceInit(argc, argv);
//     else
//         cudaSetDevice(cutGetMaxGflopsDeviceId() );

    float N_float;
    sscanf(*(argv+1),  "%f", &N_float);
//     printf("N_float = %f\n", N_float);
    unsigned int N_orig = (int)N_float;
    unsigned int N = exp2(ceil(log2(N_float)));
    printf("N_orig=%d, N=%d\n", N_orig, N);

    // allocate host memory
    float* h_idata = (float*) malloc(sizeof(float) * N);
    // initalize the memory
    srand(time(NULL));
    printf("Host data array = [");
    for(unsigned int i = 0; i < N; ++i)
    {
        if(i >= N_orig)
            h_idata[i] = 0;
        else
//             h_idata[i] = (float)i;
            h_idata[i] = (float) rand()/RAND_MAX;
//         printf("\n  %d: %f, ", i, h_idata[i]);
    }
    printf("]\n");

    // allocate device memory
    float* d_idata;
    cutilSafeCall(cudaMalloc((void**) &d_idata, sizeof(float) * N));
    // copy host memory to device
    cutilSafeCall(cudaMemcpy(d_idata, h_idata, sizeof(float) * N, cudaMemcpyHostToDevice));

    // compute the reference solution
    unsigned int timer_cpu = 0;
    cutilCheckError(cutCreateTimer(&timer_cpu));
    cutilCheckError(cutStartTimer(timer_cpu));
    float ref_max = computeGold(h_idata, N);
    cutilCheckError(cutStopTimer(timer_cpu));


    const unsigned int max_threads_per_block = 512;
    unsigned int num_threads;
    unsigned int num_blocks;
    unsigned int threads_per_block;
    dim3  grid;
    dim3  threads;
    unsigned int timer_kernel = 0;

#if 1
    // "Cook the kernel"
    {
        // setup kernel execution parameters
        num_threads = ceil(N/2.0);
        num_blocks = ceil(1.0*num_threads/max_threads_per_block);
        threads_per_block = ceil(1.0*num_threads/num_blocks);
        grid = dim3(num_blocks, 1, 1);
        threads = dim3(threads_per_block, 1, 1);
        printf("N=%d, num_threads=%d, num_blocks=%d, threads_per_block=%d, gridDim=%d, blockDim=%d\n",
                N, num_threads, num_blocks, threads_per_block, grid.x, threads.x);

        // execute the kernel
        testKernel<<<grid, threads, sizeof(float)*threads_per_block>>>(d_idata, 2*threads_per_block);
        cudaThreadSynchronize();
    }
#endif

    cutilCheckError(cutCreateTimer(&timer_kernel));
    cutilCheckError(cutStartTimer(timer_kernel));

    do{
        // setup kernel execution parameters
        num_threads = ceil(N/2.0);
        num_blocks = ceil(1.0*num_threads/max_threads_per_block);
        threads_per_block = ceil(1.0*num_threads/num_blocks);
        grid = dim3(num_blocks, 1, 1);
        threads = dim3(threads_per_block, 1, 1);
        printf("N=%d, num_threads=%d, num_blocks=%d, threads_per_block=%d, gridDim=%d, blockDim=%d\n",
                N, num_threads, num_blocks, threads_per_block, grid.x, threads.x);

        // execute the kernel
        testKernel<<<grid, threads, sizeof(float)*threads_per_block>>>(d_idata, 2*threads_per_block);
        // check if kernel execution generated and error
        cutilCheckMsg("Kernel execution failed");

        N = num_blocks;
    } while(N != 1);

    cudaThreadSynchronize();
    cutilCheckError(cutStopTimer(timer_kernel));

    // copy result from device to host
    float cuda_max;
    cutilSafeCall(cudaMemcpy(&cuda_max, d_idata, sizeof(float), cudaMemcpyDeviceToHost));

    float time_cpu = cutGetTimerValue(timer_cpu);
    float time_kernel = cutGetTimerValue(timer_kernel);
    cutilCheckError(cutDeleteTimer(timer_cpu));
    cutilCheckError(cutDeleteTimer(timer_kernel));

    N = exp2(ceil(log2(N_float)));
    float bw_cpu = N*sizeof(float)/time_cpu*1000;
    float bw_kernel = N*sizeof(float)/time_kernel*1000;

    printf("\n");
    printf("Sequential C found maximum to be  %f in %f ms (BW=%f MB/s)\n", ref_max, time_cpu, bw_cpu/1024/1024);
    printf("Parallel CUDA found maximum to be %f in %f ms (BW=%f MB/s)\n", cuda_max, time_kernel, bw_kernel/1024/1024);
    printf("Speed-up factor = %f\n", time_cpu/time_kernel);
    printf("%s\n", (ref_max == cuda_max) ? "Results matched" : "ERROR: Results do not match!!");


    // cleanup memory
    free(h_idata);
    cutilSafeCall(cudaFree(d_idata));

    cudaThreadExit();
}
