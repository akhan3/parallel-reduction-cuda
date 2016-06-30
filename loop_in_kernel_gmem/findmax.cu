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

    cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char** argv)
{
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if(cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice(cutGetMaxGflopsDeviceId() );

    unsigned int timer = 0;
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));

    unsigned int N = 1024;
    unsigned int mem_size = sizeof(float) * N;
//     unsigned int step;

    // allocate host memory
    float* h_idata = (float*) malloc(mem_size);
    // initalize the memory
    srand (time(NULL));
    printf("Data array = [");
    for(unsigned int i = 0; i < N; ++i)
    {
        h_idata[i] = (float) rand()/RAND_MAX;
        printf("%f, ", h_idata[i]);
    }
    printf("]\n");

    // allocate device memory
    float* d_idata;
    cutilSafeCall(cudaMalloc((void**) &d_idata, mem_size));
    // copy host memory to device
    cutilSafeCall(cudaMemcpy(d_idata, h_idata, mem_size,
                                cudaMemcpyHostToDevice) );

    // allocate device memory for result
    float* d_odata;
    cutilSafeCall(cudaMalloc((void**) &d_odata, sizeof(float)));

    // setup execution parameters
    dim3  grid(1, 1, 1);
    dim3  threads(N/2, 1, 1);

    unsigned int timer_kernel = 0;
    cutilCheckError(cutCreateTimer(&timer_kernel));
    cutilCheckError(cutStartTimer(timer_kernel));
    // execute the kernel
    testKernel<<<grid, threads, mem_size>>>(d_idata, d_odata, N);
    cutilCheckError(cutStopTimer(timer_kernel));
    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    float cuda_max;
    cudaMemcpy(&cuda_max, d_idata, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Parallel CUDA found maximum to be %f in %f us\n", cuda_max, cutGetTimerValue(timer_kernel)*1e3);

    unsigned int timer_cpu = 0;
    cutilCheckError(cutCreateTimer(&timer_cpu));
    cutilCheckError(cutStartTimer(timer_cpu));
    // compute the reference solution
    float ref_max = computeGold(h_idata, N);
    cutilCheckError(cutStopTimer(timer_cpu));
    printf("Sequential C found maximum to be  %f in %f us\n", ref_max, cutGetTimerValue(timer_cpu)*1e3);
    printf("%s\n", (ref_max == cuda_max) ? "Results matched" : "ERROR: Results do not match!!");

    cutilCheckError(cutDeleteTimer(timer_kernel));
    cutilCheckError(cutDeleteTimer(timer_cpu));


// //     // allocate mem for the result on host side
// //     float* h_odata = (float*) malloc(mem_size);
// //     // copy result from device to host
// //     cutilSafeCall(cudaMemcpy(h_odata, d_odata, sizeof(float) * num_threads,
// //                                 cudaMemcpyDeviceToHost) );
// //
// //     cutilCheckError(cutStopTimer(timer));
// //     printf("Processing time: %f (ms)\n", cutGetTimerValue(timer));
// //     cutilCheckError(cutDeleteTimer(timer));
// //
// //     // check result
// //     if(cutCheckCmdLineFlag(argc, (const char**) argv, "regression"))
// //     {
// //         // write file for regression test
// //         cutilCheckError(cutWriteFilef("./data/regression.dat",
// //                                       h_odata, num_threads, 0.0));
// //     }
// //     else
// //     {
// //         // custom output handling when no regression test running
// //         // in this case check if the result is equivalent to the expected soluion
// //         CUTBoolean res = cutComparef(reference, h_odata, num_threads);
// //         printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");
// //     }
// //
// //     // cleanup memory
// //     free(h_idata);
// //     free(h_odata);
// //     free(reference);
// //     cutilSafeCall(cudaFree(d_idata));
// //     cutilSafeCall(cudaFree(d_odata));

    cudaThreadExit();
}
