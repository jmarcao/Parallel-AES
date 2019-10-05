#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			// Allocate buffers on GPU and move data in
			const size_t ARR_LEN = n * sizeof(int);
			const int NUM_THREADS = n;
			const int NUM_BLOCKS = 1; // TODO: How to best comute Blocks/BlockSize?
			int* dev_odata;
			int* dev_tmp;

			// Allocate our arrays
			cudaMalloc(&dev_odata, ARR_LEN);
			cudaMalloc(&dev_tmp,   ARR_LEN);

			// Copy input to odata buffer
			// After each loop of the algorithm we will swap tmp and odata
			// So that the final result will always be located in the dev_odata buffer.
			cudaMemcpy(dev_odata, idata, ARR_LEN, ::cudaMemcpyHostToDevice);
			cudaMemcpy(dev_tmp,   idata, ARR_LEN, ::cudaMemcpyHostToDevice);

			// Algorithm adapted from GPU Gems 3, Section 39.2.1
            timer().startGpuTimer();
			for (int d = 1; d <= ilog2ceil(n); d++) {
				std::swap(dev_tmp, dev_odata);
				kernScanStep<<<NUM_BLOCKS, NUM_THREADS >>>(n, d, dev_odata, dev_tmp);
			}

			// Algorithm above produced inclusive scan, adjust to exclusive.
			std::swap(dev_tmp, dev_odata);
			kernInclusiveToExclusive<<<NUM_BLOCKS, NUM_THREADS>>>(n, dev_odata, dev_tmp);
			cudaMemset(dev_odata, 0, sizeof(int)); // Set first element to 0 (identity)

            timer().endGpuTimer();

			// Copy back to host and free memory
			cudaMemcpy(odata, dev_odata, ARR_LEN, ::cudaMemcpyDeviceToHost);
			cudaFree(dev_tmp);
			cudaFree(dev_odata);
        }

		__global__ void kernScanStep(const int N, const int D, int *out, const int* in) {
			int k = threadIdx.x + (blockIdx.x * blockDim.x);
			if (k >= N) {
				return;
			}

			if (k >= (int)powf(2, D - 1)) {
				out[k] = in[k - (int)powf(2, D - 1)] + in[k];
			}
			else {
				out[k] = in[k];
			}
		}

		__global__ void kernInclusiveToExclusive(const int N, int *out, const int* in) {
			int k = threadIdx.x + (blockIdx.x * blockDim.x);
			if (k >= N - 1) { // Modified condition, we do NOT want the last thread working.
				return;
			}

			out[k + 1] = in[k];
		}
    }
}
