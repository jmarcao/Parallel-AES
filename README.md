Parallelizing AES with CUDA
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* John Marcao
  * [LinkedIn](https://www.linkedin.com/in/jmarcao/)
  * [Personal Website](https://jmarcao.github.io)
* Tested on: Windows 10, i5-4690K @ 3.50GHz, 8GB DDR3, RTX 2080 TI 3071MB (Personal)

# Overview
In this project I implemented AES128, AES192, and AES256 in Electronic Codebook (ECB) and Counter (CTR) modes. ECB and CTR are great candidates for parallelization thanks to predictable or independent IVs. I began by implementing AES on the CPU. I followed examples directly from the AES Algorithm and the [tiny-aes-C](https://github.com/kokke/tiny-AES-c) implementation. After verifying that the algorithms were correct with NIST test vectors, I developed a naive GPU imeplementation. I designed the algorithm to operate on each block in parallel. This implementation was already significantly faster than the CPU. using NSight, I profiled the application to find bottlenecks and places to improve. Iterating over several different versions, I finally settled on an optimized implementation was 92% faster than the naive GPU implementation. I focused on utilizing shared memory, reducing calls to global memory, and changing 8bit operations to 32bit operations and unrolling loops. I also added lookup tables to reduce some repetitive operations.

``` C
// AES Algorithm Overview

// Initial Setup
KeyExpansion()
AddRoundKey()

// Rounds 1 to {9, 11, 13}
for round in rounds do:
    SubBytes()
    ShiftRows()
    MixColumns()
    AddRoundKey()

// Final Round
SubBytes()
ShiftRows()
AddRoundKey()
```

# AES Algorithm Choices
ECB was chosen because of its simple and parallelizable design. ECB encrypts every block with the same key, which is incredibly insecure, but it is simple to implement. Since each block can be encrypted independently, there is no issue.

CTR mode works by encrypting an IV joined with a unique counter value, and then XORing the encrypted value with a plaintext block to produce a ciphertext block. Since the counter can be calculated for each block ahead of time, it is easy to parallelize this. Each block receives a base IV and counter value and then increments the counter by the block index. Counter overflow is accounted for and each block receives a unique key.

# Design Choices
When implementing these algorithms in CUDA, I chose to parallelize each block. I did this to make the implementation simpler. This also means each encrypt/decrypt operations only has one CUDA kernel launch point, with one thread per block. Alternatively, I could have launched a CUDA kernel for each step in the AES algorithm. This would have allowed for even greater parallelization, especially during the SubBytes and InverseSubBytes steps. However, I decided against it for two reasons:
    1) The implementation would be more complicated.
    2) Reduced efficiency of shared memory.
By launching one thread per block, I can set up a phase where the kernels load lookup tables into shared memory. As we'll see in a bit, this provided a significant boost to performance.

# Measurements
My test application creates a buffer of test data and a test key and IV. I then call each implementation (CPU, GPU_NAIVE, GPU_OPTIMIZED) with each AES flavor (AES128, AES192, AES256). I encrypt each test buffer 5 times and then average the result. I noticed during testing that some runs would take significantly longer than others, so I wanted to smooth out that irregularity. I also added an option to pass the test buffer size in from the command line. If the test buffer is over 64MB, the tester will skip the CPU tests since those can take VERY long. This can be disabled by using a '-c' option to force the tester to always run CPU performance tests.

# Optimization Results

| Encrypting 64 MB in ECB AES256    |               |                          |                      |                     |                  |                     |                 |
|--------------------|---------------|--------------------------|----------------------|---------------------|------------------|---------------------|-----------------|
| Implementation     | Duration (ms) | Memory Throughput (GB/s) | No Warps Eligible (%) | MIO Throttle (inst) | MIO Throttle (%) | LG Throttle (inst.) | LG Throttle (%) |
| Unoptimized        | 42.23         | 4.11                     | 96.63                | 0.18                | 0                | 213                 | 93              |
| + Lookup Tables    | 31.17         | 5.27                     | 95.71                | 140.75              | 80               | 9.88                | 5               |
| + Shared Memory    | 28.95         | 5.35                     | 96.64                | 4.27                | 2                | 176.1               | 84              |
| + 32bit Operations | 20.12         | 7.44                     | 95.06                | 21.09               | 14               | 114.3               | 76              |
| + Block Copy       | 2.29          | 58.69                    | 69.39                | 18                  | 73               | 0.35                | 1               |

To collect a snapshot of what each optimization introduced to my design, I ran the same test with various optimizations enabled and disabled. The table above summarizes the changes and the effect each. For each implementation, the optimization above it is included, since each layer depends on the previous layer to be efficient.

For the AES MixColumns step, the block is multiplied with a set matrix. The matrix is designed to produce a unique output for each input to the matrix multiplication. (Sidenote: The mathematics behind it, Galois Fields, is super interesting and I spent a solid hour just learning how those work and how it applies to AES). Since the matrix is a constant and we are multiplying it with a byte, it is efficient to write a lookup table for each multiplication. For encryption, we need two tables (mul2 and mul3) and for decryption we need four (mul9, mulB, mulD, mulE). My first optimization was to store these in constant memory for each thread to access. This gave some small bumps in performance across the board, but greatly increased the MIO throttle. This is because accesses to the lookup table are mostly random. Each thread in a CUDA warp that wanted to read form the table had to do so serially.

``` C
// Table for mix_rows multiplication by 2
__constant__ static const uint8_t c_mul2[] = {
	0x00, 0x02, 0x04, 0x06, 0x08, 0x0a, 0x0c, 0x0e, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, 
	...
	0xfb, 0xf9, 0xff, 0xfd, 0xf3, 0xf1, 0xf7, 0xf5, 0xeb, 0xe9, 0xef, 0xed, 0xe3, 0xe1, 0xe7, 0xe5
};

// Table for mix_rows multiplication by 3
__constant__ static const uint8_t c_mul3[] = {
	0x00, 0x03, 0x06, 0x05, 0x0c, 0x0f, 0x0a, 0x09, 0x18, 0x1b, 0x1e, 0x1d, 0x14, 0x17, 0x12, 0x11, 
	...
	0x0b, 0x08, 0x0d, 0x0e, 0x07, 0x04, 0x01, 0x02, 0x13, 0x10, 0x15, 0x16, 0x1f, 0x1c, 0x19, 0x1a
};
```

To improve the performance of the lookup tables, I created regions in shared memory to store them. In shared memory the threads would be closer to the data and would not be limited by the cost of serializing a constant memory access. I implemented this by having a shared memory buffer allocated for each lookup table and then having each thread in the kernel launch read a portion of the data into shared memory. If not enough threads exist to read all the data, then thread 0 does all the reading as a backup. This gave a small improvement in performance again, but still not enough. The bottleneck was again the LG (Local Global Memory Access) throttle.

NSight suggested that LG Throttling is a sign of reading too much data at once. LG Throttling occurs when the memory request pipelines are full and the thread refuses to be scheduled until the pipeline has room. To reduce this, I changed each 8bit access to Local and Global memory to a 32bit access, when possible. This was an interesting change, since it introduced a lot of changes. I unrolled several loops to help bundle data reads together. This also led me to add a small optimization to the ShiftRows step. Before, I was calculating each row shift programmatically. This required reading 8 bits from each of 4 32bit values. I unrolled my loops and changed it to read 4 32bit words instead of 12 8bit words spread around. This also let me precompute the actual location of each byte. What previously involved several adds and multiplies to calculate proper index values turned into a series of simple shift and bitwise operations. To me, this was the most interesting optimization.

``` C
// ShiftRows Step Objective
// Transform the matrix on the left to the one on the right
/*
	| 0  1  2  3 |       | 0  5  A  F |
	| 4  5  6  7 |  -->  | 4  9  E  3 |
	| 8  9  A  B |  -->  | 8  D  2  7 |
	| C  D  E  F |       | C  1  6  B |
*/

// Unoptimized: 12 8bit reads, 12 8bit writes
...
uint8_t tmp[4];
for (int row = 1; row < N_ROWS; row++) {
	tmp[0] = data[row + N_COLS * 0];
	tmp[1] = data[row + N_COLS * 1];
	tmp[2] = data[row + N_COLS * 2];
	tmp[3] = data[row + N_COLS * 3];
	data[row + N_COLS * 0] = tmp[mod4(0 + row)];
	data[row + N_COLS * 1] = tmp[mod4(1 + row)];
	data[row + N_COLS * 2] = tmp[mod4(2 + row)];
	data[row + N_COLS * 3] = tmp[mod4(3 + row)];
}
...

// Optimized: 4 32bit reads, 4 32bit writes
...
uint32_t c0 = ((uint32_t*)data)[0];
uint32_t c1 = ((uint32_t*)data)[1];
uint32_t c2 = ((uint32_t*)data)[2];
uint32_t c3 = ((uint32_t*)data)[3];

// Don't calculate indices, we already know what the result should be
// #define BYTE0(x) (x & 0x000000FF), etc...
((uint32_t*)data)[0] = BYTE0(c0) | BYTE1(c1) | BYTE2(c2) | BYTE3(c3);
((uint32_t*)data)[1] = BYTE0(c1) | BYTE1(c2) | BYTE2(c3) | BYTE3(c0);
((uint32_t*)data)[2] = BYTE0(c2) | BYTE1(c3) | BYTE2(c0) | BYTE3(c1);
((uint32_t*)data)[3] = BYTE0(c3) | BYTE1(c0) | BYTE2(c1) | BYTE3(c2);
...
```

My last optimization came way too late. I realized that accesses to each threads block data was going all the way out to global memory. This was useless, since no other thread would read or write to that memory. I added a copy of the 16byte block from global to thread-local memory and then performance SOARED. This makes sense, obviously. After moving all my data to thread local or SM local memory, my access to Global space was nearly zero. According to NSight, i read a total of 64MB (the actual block data, required) from Device Memory and wrote all of it back (save the data read for lookup tables). As can be seen below, memory read from the device and global memory dropped significantly. This ended up being the largest optimization, increasing memory throughput +1,327% from the unoptimized implementation.

![](img\memory_access_opt_vs_unopt.JPG)

# Algorithm Results
I implemented a performance test mode in my application to easily compare each algorithm's throughput. The tester runs each algorithm 5 times and averages the result. The validity of the results are not checked in the performance tester, though that can be done through the '-t' option.

![](img\example_performance_output.JPG)

I ran the tester over several test buffer sizes to see how each stacks up against the other.

![](img\pchart_cpu.png)

Looking at the chart, the CPU implementations become far too slow after 5MB of data. Now, far too slow is relative to the GPU implementations, but from that point on the CPU is removed from the graph. Beyond 50MB CPU data is no longer collected due to the time ti takes to process the data, the next step being 500MB. We can start to see here that the three variants of AES do have some impact, but not significantly. The main difference between AES128, AES192, and AES256 in the implementation is how many rounds are needed (10, 12, and 14 respectively). If each round is optimized, then the difference between them is never really seen.

![](img\pchart_gpu_enc.png)

In this chart I have zoomed in closer to the GPU data. We can see here even more that the different variations of AES indeed slower, but not by a huge margin. We do see though that the unoptimized implementations perform worse than the optimized.

| Algo | Duration | Relative to AES128 |
|----------|--------------------|------| 
| AES128   | 111.328            | 1.00 | 
| AES192   | 125.025            | 1.12 | 
| AES256   | 145.177            | 1.30 | 

One thing to notice here is that I am only comparing the ECB implementations. One benefit of the CTR mode implementation is that, even unoptimized, it is very fast. Because each CUDA thread only needs to read the counter value from global memory, and every thread does this at the same time, the value is quickly cached and made available to every kernel. Once the counter is incremented and encrypted, it is XOR'd directly into the global memory space. Each thread is written to a separate block in global space, so there are no contentions to worry about. Because of this, even when unoptimized, the CTR mode performs very well. In fact, the optimized version performs worse! I am not sure why this is the case, but I imagine that, with all the operations being done locally in the thread, there are some compiler tricks the NVCC tool is performing that I am unaware of.

![](img\pchart_ctr.png)

# Summary
Doing this project really introduced me to the useful features of the NSight profiler, as well as how to identify and fix issues performance issues in CUDA code. I did find the profiler lacking compared to some legacy tools, but unfortunately my GPU is not compatible with legacy Nvidia tools. I found out how algorithms can be molded and adapted for parallel execution and what unique challenges need to be solved there. I found that having a deeper knowledge of AES helped greatly, since I was able to find shortcuts in the implementation.