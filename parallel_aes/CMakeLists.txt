set(SOURCE_FILES
    "common.h"
    "common.cu"
    "cpu.h"
    "cpu.cu"
    "gpu.h"
    "gpu.cu"
    )

cuda_add_library(parallel_aes
    ${SOURCE_FILES}
    OPTIONS -arch=sm_75
    )
