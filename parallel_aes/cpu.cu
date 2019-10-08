#include <cstdio>
#include "cpu.h"
#include "common.h"

#include "tiny-AES-c/aes.h" // Use tiny-aes for CPU implementation
                            // GPU implementation will be very different.

namespace PAES {
    namespace CPU {
        using PAES::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }
    }
}