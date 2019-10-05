#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);
		__global__ void kernScanStep(const int N, const int D, int *odata, const int* idata);
		__global__ void kernInclusiveToExclusive(const int N, int *out, const int* in);
    }
}
