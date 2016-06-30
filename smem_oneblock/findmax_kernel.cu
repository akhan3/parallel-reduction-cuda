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

#define SDATA( index)      cutilBankChecker(sdata, index)

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel( float* g_idata, float* g_odata, unsigned int N)
{
    // shared memory
    // the size is determined by the host application
    extern  __shared__  float sdata[];

    // access thread id
    const unsigned int tid = threadIdx.x;

    // copy the global data to shared memory
    SDATA(tid*2) = g_idata[tid*2];
    SDATA(tid*2+1) = g_idata[tid*2+1];

    unsigned int step;
    for(unsigned int i = N; i > 1; i = i/2) {
        step = N/i;
        if(tid < i/2) {
            // printf("threads: %d, step: %d\n", i/2, step);
            // do the comparison
            if(SDATA(tid*2*step+step) > SDATA(tid*2*step))
                SDATA(tid*2*step) = SDATA(tid*2*step+step);
        }
        __syncthreads();
    }







//     // read in input data from global memory
//     // use the bank checker macro to check for bank conflicts during host
//     // emulation
//     SDATA(tid) = g_idata[tid];
//     __syncthreads();
//
//     // perform some computations
//     SDATA(tid) = (float) num_threads * SDATA( tid);
//     __syncthreads();
//
//     // write data to global memory
//     g_odata[tid] = SDATA(tid);

}

#endif // #ifndef _TEMPLATE_KERNEL_H_
