#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);

		__global__ void kernWorkEffScanUpsweep(const int N, const int D, int *out, const int* in);

		__global__ void kernWorkEffScanDownsweep(const int N, const int D, int *out, const int* in);

		int nextPowerOfTwo(int in);

    }
}
