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

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
float computeGold(float* idata, const unsigned int N);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////

float computeGold(float* idata, const unsigned int N) {
    unsigned int step = 1;
    for(unsigned int i = N; i > 1; i = i/2) {
        for(unsigned int j = 0; j < N; j += 2*step)
            if(idata[j+step] > idata[j])
                idata[j] = idata[j+step];
        step *= 2;
    }
    return *idata;
}
