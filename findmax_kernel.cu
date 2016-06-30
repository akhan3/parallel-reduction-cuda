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
 * Device code.
 */

#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>

//#define SDATA(index)      cutilBankChecker(sdata, index)
#define SDATA(index)      sdata[index]

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel(float* g_idata, unsigned int N)
{
    // shared memory
    // the size is determined by the host application
    extern  __shared__  float sdata[];

    volatile int index = 2*blockDim.x*blockIdx.x + threadIdx.x;
    volatile int index2 = index + N/2 ;

    SDATA(threadIdx.x) = (g_idata[index] > g_idata[index2]) ?
                          g_idata[index] : g_idata[index2];

    __syncthreads();

    N = N/2;

    if (threadIdx.x < 256)
        if(SDATA(threadIdx.x) < SDATA(threadIdx.x+256)) SDATA(threadIdx.x) = SDATA(threadIdx.x+256);

    __syncthreads();

    if (threadIdx.x < 128)
        if(SDATA(threadIdx.x) < SDATA(threadIdx.x+128)) SDATA(threadIdx.x) = SDATA(threadIdx.x+128);

    __syncthreads();

    if (threadIdx.x < 64)
        if(SDATA(threadIdx.x) < SDATA(threadIdx.x+64)) SDATA(threadIdx.x) = SDATA(threadIdx.x+64);

    __syncthreads();

    if (threadIdx.x < 32)
    {
        if(SDATA(threadIdx.x) < SDATA(threadIdx.x+32)) SDATA(threadIdx.x) = SDATA(threadIdx.x+32);
        if(SDATA(threadIdx.x) < SDATA(threadIdx.x+16)) SDATA(threadIdx.x) = SDATA(threadIdx.x+16);
        if(SDATA(threadIdx.x) < SDATA(threadIdx.x+ 8)) SDATA(threadIdx.x) = SDATA(threadIdx.x+ 8);
        if(SDATA(threadIdx.x) < SDATA(threadIdx.x+ 4)) SDATA(threadIdx.x) = SDATA(threadIdx.x+ 4);
        if(SDATA(threadIdx.x) < SDATA(threadIdx.x+ 2)) SDATA(threadIdx.x) = SDATA(threadIdx.x+ 2);
        if(SDATA(threadIdx.x) < SDATA(threadIdx.x+ 1)) SDATA(threadIdx.x) = SDATA(threadIdx.x+ 1);
    }

    if(threadIdx.x == 0) {
//         printf("gtid:%d.%d block_max=%f\n", blockIdx.x, threadIdx.x, SDATA(0));
        g_idata[blockIdx.x] = SDATA(0);
    }
}


#endif // #ifndef _TEMPLATE_KERNEL_H_
