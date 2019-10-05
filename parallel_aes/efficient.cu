#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <bitset>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		int nextPowerOfTwo(int in) {
			int out = 0;
			float log = log2(in);

			// If this is true, the number IS a power of 2
			if (ceil(log) == floor(log)) {
				out = in;
			}
			else {
				// Not a power of two, grab the next one up.
				out = 1;
				do {
					out = out << 1;
				} while (out < in);
			}

			return out;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			// Efficient algorithm uses balanded binary trees and two phases: upsweep and downsweep.
			// This can be performed inplace.

			// 0) Correct length to be Power of 2
			const int N = nextPowerOfTwo(n); // Returns 'n' if input is already a power of 2.
			
			// TODO: How to best comute Blocks/BlockSize?
			const int NUM_THREADS = n;
			const int NUM_BLOCKS = 1;

			// 1) Initialize Memory
			int* dev_data = 0;
			cudaMalloc(&dev_data, N * sizeof(int));
			cudaMemset(dev_data + n, 0, (N - n) * sizeof(int));
			cudaMemcpy(dev_data, idata, n * sizeof(int), ::cudaMemcpyHostToDevice);

			// 2) Upsweep
			timer().startGpuTimer();
			for (int d = 0; d <= ilog2ceil(N) - 1; d++) {
				kernWorkEffScanUpsweep<<<NUM_BLOCKS, NUM_THREADS>>>(n, d, dev_data, dev_data);
			}

			// 3) Downsweep
			cudaMemset(dev_data + (N-1), 0, 1*sizeof(int)); // Set last element to 0
			for (int d = ilog2ceil(N) - 1; d >= 0; d--) {
				kernWorkEffScanDownsweep<<<NUM_BLOCKS, NUM_THREADS>>>(n, d, dev_data, dev_data);
			}
			
			// 4) Cleanup
			timer().endGpuTimer();
			cudaMemcpy(odata, dev_data, N * sizeof(int), ::cudaMemcpyDeviceToHost);
			cudaFree(dev_data);

			return;
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
			const int NUM_THREADS = n;
			const int NUM_BLOCKS = 1;

			// Sim8ilar to CPU implementation, except we use CUDA kernels
			// instead of for-loops

			// 0) Correct length to be Power of 2
			const int N = nextPowerOfTwo(n); // Returns 'n' if input is already a power of 2.

			int* INSPECT = (int*)malloc(N * sizeof(int));

			// Prepare memory
			int* dev_odata;
			int* dev_idata;
			int* dev_map;
			int* dev_indicies;

			cudaMalloc(&dev_odata, N * sizeof(int));
			cudaMalloc(&dev_idata, N * sizeof(int));
			cudaMalloc(&dev_map, N * sizeof(int));
			cudaMalloc(&dev_indicies, N * sizeof(int));

			cudaMemcpy(dev_idata, idata, N * sizeof(int), ::cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // 1) Map
			Common::kernMapToBoolean << <NUM_BLOCKS, NUM_THREADS >> > (N, dev_map, dev_idata);
			cudaMemcpy(INSPECT, dev_idata, N * sizeof(int), ::cudaMemcpyDeviceToHost);
			cudaMemcpy(INSPECT, dev_map, N * sizeof(int), ::cudaMemcpyDeviceToHost);

			// 2) Scan
			// 2a) Upsweep
			cudaMemcpy(dev_indicies, dev_map, N * sizeof(int), ::cudaMemcpyDeviceToDevice);
			for (int d = 0; d <= ilog2ceil(N) - 1; d++) {
				kernWorkEffScanUpsweep << <NUM_BLOCKS, NUM_THREADS >> > (N, d, dev_indicies, dev_indicies);
			}
			// 2b) Downsweep
			cudaMemset(dev_indicies + (N - 1), 0, 1 * sizeof(int)); // Set last element to 0
			for (int d = ilog2ceil(N) - 1; d >= 0; d--) {
				kernWorkEffScanDownsweep << <NUM_BLOCKS, NUM_THREADS >> > (N, d, dev_indicies, dev_indicies);
			}
			cudaMemcpy(INSPECT, dev_indicies, N * sizeof(int), ::cudaMemcpyDeviceToHost);

			// 3) Scatter
			Common::kernScatter << <NUM_BLOCKS, NUM_THREADS >> > (N, dev_odata, dev_idata, dev_map, dev_indicies);
            timer().endGpuTimer();

			// Copy back to host
			cudaMemcpy(odata, dev_odata, N * sizeof(int), ::cudaMemcpyDeviceToHost);

			// Get number of elements from indicies
			int num_elements = 0;
			cudaMemcpy(&num_elements, dev_indicies + N - 1, sizeof(int), ::cudaMemcpyDeviceToHost);

			cudaFree(dev_odata);
			cudaFree(dev_idata);
			cudaFree(dev_map);
			cudaFree(dev_indicies);

            return num_elements;
        }

		__global__ void kernWorkEffScanUpsweep(const int N, const int D, int *out, const int* in) {
			int k = threadIdx.x + (blockIdx.x * blockDim.x);
			if (k >= N) {
				return;
			}

			if (k % (int)powf(2, D + 1) == 0) {
				out[k + (int)powf(2, D + 1) - 1] += in[k + (int)powf(2, D) - 1];
			}
		}

		__global__ void kernWorkEffScanDownsweep(const int N, const int D, int *out, const int* in) {
			int k = threadIdx.x + (blockIdx.x * blockDim.x);
			if (k >= N) {
				return;
			}

			if (k % (int)powf(2, D + 1) == 0) {
				int tmp = in[k + (int)powf(2, D) - 1];
				out[k + (int)powf(2, D) - 1] = out[k + (int)powf(2, D + 1) - 1];
				out[k + (int)powf(2, D + 1) - 1] += tmp;
			}
		}
    }
}
