#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();

			// Exclusive, naive, sequential scan.
			odata[0] = 0;
			for (int i = 1; i < n; i++) {
				odata[i] = odata[i - 1] + idata[i - 1];
			}

	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            
			// Simple stream compaction w/o scan.
			// Fills output array with non-null values.
			int oidx = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i]) {
					odata[oidx] = idata[i];
					oidx++;
				}
			}

	        timer().endCpuTimer();
            return oidx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
			// CPU Stream compation with scan() function.
			// Create intermediate buffer.
			int *tmpMap = (int*)malloc(n * sizeof(int));
			if (!tmpMap) {
				throw std::runtime_error("Failed to allocate memory for tmpMap buffer!");
			}
			int *tmpScan = (int*)malloc(n * sizeof(int));
			if (!tmpScan) {
				throw std::runtime_error("Failed to allocate memory for tmpScan buffer!");
			}

			// Exclude the above mallocs from timing, since they can block!
			// Assume everything is allocated already for us
			timer().startCpuTimer();

			// Step 1: Map
			for (int i = 0; i < n; i++) {
				tmpMap[i] = (idata[i] != 0);
			}

			// Step 2: Scan
			tmpScan[0] = 0;
			for (int i = 1; i < n; i++) {
				tmpScan[i] = tmpScan[i - 1] + tmpMap[i - 1];
			}

			// Step 3: Scatter
			int oidx = 0;
			for (int i = 0; i < n; i++) {
				if (tmpMap[i]) {
					odata[tmpScan[i]] = idata[i];
					oidx++;
				}
			}

			timer().endCpuTimer();

			free(tmpMap);
			free(tmpScan);
            return oidx;
        }
    }
}
