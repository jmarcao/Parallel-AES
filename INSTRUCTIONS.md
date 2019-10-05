# GPU Assignment 4 Proposal

Eric Micallef, Taylor Nelms, John Marcao

# Proposal

We propose a project to implement, in parallel, the AES encryption algorithm across the GPU using the CUDA framework. There are variants of encryption algorithms that can be implemented in a very data-parallel manner, and we hope to increase performance over CPU implementations by leveraging the parallel-processing power of a GPU engine.

For AES, there are two block-cipher modes that are potentially parallelizable: ECB and CTR. Other modes use data-dependent block cipher modes that would require serialization of execution, which would limit performance capabilities.
We hope to explore in-depth a variety of methods discussed in class for achieving peak GPU performance, with regards to memory access, block configuration, and scheduling strategies.

Note: this is an individual project.

# Features

## Implementation

* Baseline CPU implementation of AES algorithm for benchmarking purposes (see notes about External Libraries)
* Parallel implementation of AES ECB mode (encryption and decryption)
* Parallel implementation of AES CTR mode (encryption and decryption)
* Runtime configurability of AES strength (support for 128, 192, and 256-bit key sizes)
* Runtime configurability of performance parameters (required for Analysis)
* Any additional features to improve efficiency or performance

## Analysis

* Analysis of performance with regards to block size and thread configuration
* Analysis of performance with different approaches to data storage locations, for values consistent across threads (likely, used for expanded keys and various lookup tables), including:
* Passing data as kernel arguments
* Pulling data from global memory
* Pulling data from global constant memory
* Reading data into shared memory
* Resource usage analysis for all of the above
* Analysis of performance effects of different data structuring methods (ex. using uchar4 structs for storage/processing rather than byte arrays to make use of CUDA mixed-precision programming)
* support for command line argument file i/o


## External Libraries

We will allow ourselves use of an existing CPU library implementation of AES as a baseline (https://github.com/kokke/tiny-AES-c). Additionally, we would like to use Thrust functions, where sensible, when manipulating data on the GPU.
No other third-party support should be necessary; all other primitive operations should be achievable directly via CUDA and host functions.

Instructor Note (Ziad)
* “The provided CPU implementation seems to provide A LOT of code so we'd like you to limit the amount of straight copying from that repo. Please make this implementation as original as possible. We will be checking this programmatically.”

Instructor Note (Shehzan)
* “I agree with Ziad. I think the mixed precision would be useful.
* Additionally, the compile time options for strength and performance options are not a good idea and you should use runtime options.  Using compile time options kill the usability of the code you are writing. You don't need to make it production quality, but you can use a library to parse the arguments (I found https://github.com/jarro2783/cxxopts, more https://attractivechaos.wordpress.com/2018/08/31/a-survey-of-argument-parsing-libraries-in-c-c/).
* Limit the use of thrust to algorithms that either mundane or too complex, for example sort (which could be complicated), or stream compaction (because you already know it).
* If you still have reservations about mixed precision, let me know.”