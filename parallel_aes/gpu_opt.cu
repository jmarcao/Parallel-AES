#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <bitset>
#include <assert.h>
#include "common.h"
#include "gpu_opt.h"

#define N_COLS 4
#define N_ROWS 4
#define SUBKEY_SIZE 16

#define BYTE0(x) (x & 0x000000FF)
#define BYTE1(x) (x & 0x0000FF00)
#define BYTE2(x) (x & 0x00FF0000)
#define BYTE3(x) (x & 0xFF000000)

// If someone else has an optimized flag, get rid of it.
#ifdef OPTIMIZED
#undef OPTIMIZED
#endif
#define OPTIMIZED 1

namespace PAES {
    namespace GPU_OPT {
        using PAES::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		// A small helper for quick mod16, since it'll be used alot when accessing keys
		// in sharedmem. Works by abusing the fact that 16 and 4 are powers of 2.
		template<typename T>
		__host__ __device__ __forceinline__ T mod16(T n) {
			return n & 15;
		}

		template<typename T>
		__host__ __device__ __forceinline__ T mod4(T n) {
			return n & 3;
		}

		// Gets length of key in bytes for flavor
		__host__ int get_key_len(const AESType& a) {
			switch (a) {
				case AESType::AES128: return 128 / 8;
				case AESType::AES192: return 192 / 8;
				case AESType::AES256: return 256 / 8;
			}

			return -1;
		}

		// Returns length of expanded key in bytes.
		__host__ int get_exp_key_len(const AESType& a) {
			switch (a) {
				case AESType::AES128: return 176;
				case AESType::AES192: return 208;
				case AESType::AES256: return 240;
			}

			return -1;
		};

		// Gets number of rounds for flavor
		__host__ int get_num_rounds(const AESType& a) {
			switch (a) {
				case AESType::AES128: return 10;
				case AESType::AES192: return 12;
				case AESType::AES256: return 14;
			}

			return -1;
		}

		__host__ void encrypt_ecb(const AESType& flavor, uint8_t * key, uint8_t * data, uint32_t datalen)
		{
			// Straightforward, nothing fancy is done here.
			// ECB is inherently insecure because it does not diffuse the data
			assert(mod16(datalen) == 0);

			// Expand the key
			uint8_t* expkey = (uint8_t*)malloc(get_exp_key_len(flavor));
			expandKey(flavor, key, expkey, get_key_len(flavor), get_num_rounds(flavor));

			// Copy key to device memory
			uint8_t* d_expkey;
			cudaMalloc(&d_expkey, get_exp_key_len(flavor));
			cudaMemcpy(d_expkey, expkey, get_exp_key_len(flavor), cudaMemcpyHostToDevice);

			// Copy data to device memory
			uint8_t* d_data;
			cudaMalloc(&d_data, sizeof(uint8_t) * datalen);
			cudaMemcpy(d_data, data, sizeof(uint8_t) * datalen, cudaMemcpyHostToDevice);

			// Calculate number of kernels needed
			// We need one kernel per aes block. So we will calculate how many
			// cuda blocks of 1024 threads we need to satisfy that
			int threadsPerBlock = 1024;
			int aes_blocks = datalen / BLOCKSIZE;
			int cudaBlocks = (aes_blocks + 1023) / 1024;

			// Call the kernels to get to work!
			timer().startGpuTimer();
			core_encrypt_ecb << <cudaBlocks, threadsPerBlock >> > (aes_blocks, d_data, d_expkey, get_num_rounds(flavor));
			checkCUDAError("ECB Encrypt Failed!");
			cudaDeviceSynchronize();
			timer().endGpuTimer();

			// Retrieve the data from the device
			cudaMemcpy(data, d_data, sizeof(uint8_t) * datalen, cudaMemcpyDeviceToHost);

			// Free CUDA memory
			cudaFree(d_data);
			cudaFree(d_expkey);
		}

		__host__ void decrypt_ecb(const AESType& flavor, uint8_t * key, uint8_t * data, uint32_t datalen)
		{
			// Straightforward, nothing fancy is done here.
			// ECB is inherently insecure because it does not diffuse the data
			assert(mod16(datalen) == 0);

			// Expand the key
			uint8_t* expkey = (uint8_t*)malloc(get_exp_key_len(flavor));
			expandKey(flavor, key, expkey, get_key_len(flavor), get_num_rounds(flavor));

			// Copy key to device memory
			uint8_t* d_expkey;
			cudaMalloc(&d_expkey, get_exp_key_len(flavor));
			cudaMemcpy(d_expkey, expkey, get_exp_key_len(flavor), cudaMemcpyHostToDevice);

			// Copy data to device memory
			uint8_t* d_data;
			cudaMalloc(&d_data, sizeof(uint8_t) * datalen);
			cudaMemcpy(d_data, data, sizeof(uint8_t) * datalen, cudaMemcpyHostToDevice);

			// Calculate number of kernels needed
			// We need one kernel per aes block. So we will calculate how many
			// cuda blocks of 1024 threads we need to satisfy that
			int threadsPerBlock = 1024;
			int aes_blocks = datalen / BLOCKSIZE;
			int cudaBlocks = (aes_blocks + 1023) / 1024;

			// Call the kernels to get to work!
			timer().startGpuTimer();
			core_decrypt_ecb << <cudaBlocks, threadsPerBlock >> > (aes_blocks, d_data, d_expkey, get_num_rounds(flavor));
			checkCUDAError("ECB Decrypt Failed!");
			cudaDeviceSynchronize();
			timer().endGpuTimer();

			// Retrieve the data from the device
			cudaMemcpy(data, d_data, sizeof(uint8_t) * datalen, cudaMemcpyDeviceToHost);

			// Free CUDA memory
			cudaFree(d_data);
			cudaFree(d_expkey);
		}

		__host__ void encrypt_ctr(const AESType& flavor, uint8_t * key, uint8_t * ctr, uint8_t * data, uint32_t datalen)
		{
			// In counter mode, we don't actually encrypt the data.
			// Instead, we encrypt a counter value and then xor it with the data.
			// We use an IV to create our counter and then increment the counter for
			// each block encrypted.
			// CTR/IV is one block size.
			assert(mod16(datalen) == 0);

			// Expand the key
			uint8_t* expkey = (uint8_t*)malloc(get_exp_key_len(flavor));
			expandKey(flavor, key, expkey, get_key_len(flavor), get_num_rounds(flavor));

			// Copy key to device memory
			uint8_t* d_expkey;
			cudaMalloc(&d_expkey, get_exp_key_len(flavor));
			cudaMemcpy(d_expkey, expkey, get_exp_key_len(flavor), cudaMemcpyHostToDevice);

			// Copy data to device memory
			uint8_t* d_data;
			cudaMalloc(&d_data, sizeof(uint8_t) * datalen);
			cudaMemcpy(d_data, data, sizeof(uint8_t) * datalen, cudaMemcpyHostToDevice);

			// Copy counter to device memroy
			// OG ctr will be constant, each kernel reads it into its own memory
			// and performs increments
			uint8_t* d_ctr;
			cudaMalloc(&d_ctr, sizeof(uint8_t) * BLOCKSIZE);
			cudaMemcpy(d_ctr, ctr, sizeof(uint8_t) * BLOCKSIZE, cudaMemcpyHostToDevice);

			// Calculate number of kernels needed
			// We need one kernel per aes block. So we will calculate how many
			// cuda blocks of 1024 threads we need to satisfy that
			int threadsPerBlock = 1024;
			int aes_blocks = datalen / BLOCKSIZE;
			int cudaBlocks = (aes_blocks + 1023) / 1024;

			// Start the kernels. Each kernel will increment the counter
			// based on their index.
			timer().startGpuTimer();
			core_xcrypt_ctr << <cudaBlocks, threadsPerBlock>> > (aes_blocks, d_data, d_expkey, get_num_rounds(flavor), d_ctr);
			checkCUDAError("CTR Xcrypt Failed!");
			cudaDeviceSynchronize();
			timer().endGpuTimer();

			// Retrieve the data from the device
			cudaMemcpy(data, d_data, sizeof(uint8_t) * datalen, cudaMemcpyDeviceToHost);

			// Free CUDA memory
			cudaFree(d_data);
			cudaFree(d_expkey);
			cudaFree(d_ctr);
		}

		__host__ void decrypt_ctr(const AESType& flavor, uint8_t * key, uint8_t * ctr, uint8_t * data, uint32_t datalen)
		{
			// A convienent feature of CTR mode... decryption is the SAME operation! No inverse needed!
			encrypt_ctr(flavor, key, ctr, data, datalen);
		}

		__host__ void rot_word(uint8_t* n) {
			// This function shifts the 4 bytes in a word to the left once.
			// [a0,a1,a2,a3] becomes [a1,a2,a3,a0]
			uint8_t tmp = n[0];
			n[0] = n[1];
			n[1] = n[2];
			n[2] = n[3];
			n[3] = tmp;
		}
		
		__host__ void sub_word(uint8_t* n) {
			// SubWord() is a function that takes a four-byte input word and 
			// applies the S-box to each of the four bytes to produce an output word.
			n[0] = h_sbox[n[0]];
			n[1] = h_sbox[n[1]];
			n[2] = h_sbox[n[2]];
			n[3] = h_sbox[n[3]];
		}

		__host__ void expandKey(const AESType& flavor, uint8_t* ogkey, uint8_t* expkey, uint32_t keysize, uint32_t num_rounds) {
			// The AES key will either be 128, 192, or 256 bits.
			// The AES algorithm itself is not actually modified by the key size, but the number
			// of rounds is. In expandKey() we take the provided key and stretch it out to create enough
			// 128bit subkeys/roundkeys for each round. 
			// This is only done once, so we won't parallelize this.

			// The logic below follows the Rijndael Key Schedule (https://en.wikipedia.org/wiki/Rijndael_key_schedule)

			// Code adapted from tiny-aes since this is not parallelizable (each subkey depends on value of previous subkey)
			// and there is nothing inherintly unique about this.
			// Changes were made to make keysize a runtime option.

			unsigned i, j, k;
			uint8_t tmp[4]; // Used for the column/row operations
			uint32_t N = keysize / 4; // Length of key in 32bit words.

			// The first round key is the key itself.
			for (i = 0; i < N; ++i) {
				expkey[(i * 4) + 0] = ogkey[(i * 4) + 0];
				expkey[(i * 4) + 1] = ogkey[(i * 4) + 1];
				expkey[(i * 4) + 2] = ogkey[(i * 4) + 2];
				expkey[(i * 4) + 3] = ogkey[(i * 4) + 3];
			}

			// All other round keys are found from the previous round keys.
			for (i = N; i < N_COLS * (num_rounds + 1); ++i) {
				// Tuck away temporary values
				// These will be reused later.
				k = (i - 1) * 4;
				tmp[0] = expkey[k + 0];
				tmp[1] = expkey[k + 1];
				tmp[2] = expkey[k + 2];
				tmp[3] = expkey[k + 3];

				// If i % N is zero, xor the previous word with sbox(rotate(previousword)) and
				// xor that with the Round Constant. Round Constant depends on the
				// flavor of AES
				if (i % N == 0) {
					rot_word(tmp); // Rotate...
					sub_word(tmp); // Substitute...
					tmp[0] = tmp[0] ^ roundcon[i / N]; // Apply round coefficient
				}
				// Next step is only done if in AES256 mode
				else if (flavor == AESType::AES256 && (i % N == 4)) {
					sub_word(tmp); // Just subsitute
				}

				j = i * 4;
				k = (i - N) * 4;
				expkey[j + 0] = expkey[k + 0] ^ tmp[0];
				expkey[j + 1] = expkey[k + 1] ^ tmp[1];
				expkey[j + 2] = expkey[k + 2] ^ tmp[2];
				expkey[j + 3] = expkey[k + 3] ^ tmp[3];
			}
		}

		 __global__ void core_encrypt_ecb(int N, uint8_t* data, const uint8_t* key, const int num_rounds) {
			// Lenght of buffer is ALWAYS 128 bits == 16 bytes
			// This is defined by AES Algorithm
			uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= N) {
				return;
			}

			// Get shared mem buffers
			__shared__ uint8_t s_sbox[256];
			__shared__ uint8_t s_mul2[256];
			__shared__ uint8_t s_mul3[256];

			// If we have enough threads, let each one do one copy.
			if (N >= 256 && idx < 256) {
				s_sbox[idx] = c_sbox[idx];
				s_mul2[idx] = c_mul2[idx];
				s_mul3[idx] = c_mul3[idx];
			}
			// If we don't have enough blocks, just let thread 0 do it all.
			// If you only have 256 blocks you should be doing AES in CPU 
			// but thats none of my business *sips tea*.
			else if (idx == 0) {
				for (int i = 0; i < 256; i++) {
					s_sbox[i] = c_sbox[i];
					s_mul2[i] = c_mul2[i];
					s_mul3[i] = c_mul3[i];
				}
			}

			// Wait for shared memory to load
			__syncthreads();

			// Each thread running this function will act on ONE block in memory.
#if 0 // Get a pointer to global mem
			uint8_t* myData = data + idx * BLOCKSIZE;
#else // Copy into local mem
			uint8_t myData[BLOCKSIZE];
			((uint32_t*)myData)[0] = ((uint32_t*)data)[idx*(BLOCKSIZE / 4) + 0];
			((uint32_t*)myData)[1] = ((uint32_t*)data)[idx*(BLOCKSIZE / 4) + 1];
			((uint32_t*)myData)[2] = ((uint32_t*)data)[idx*(BLOCKSIZE / 4) + 2];
			((uint32_t*)myData)[3] = ((uint32_t*)data)[idx*(BLOCKSIZE / 4) + 3];
#endif

			// Lots of comments are pulled from https://en.wikipedia.org/wiki/Advanced_Encryption_Standard
			
			// Above article mentions that SubBytes, ShiftRows, and MixColumns can be combined
			// into 16 table lookups and 12 32bit XOR operations.
			// If doing each block with 1 byte per thread, then each thread needs to perform
			// one table lookup and at most one XOR.
			// What are the memory implications? Can the tables be stored in RO memory to speed
			// up the operations? Hoho! Use texture memory???

			// Initial Round Key Addition
			// Each byte of the state is combined with a block of the round key using bitwise xor
			add_round_key(idx, 0, myData, key);

			// We perform the next steps for a number of rounds
			// dependent on the flavor of AES.
			// Pass this in via a context? 
			for (int r = 1; r < num_rounds; r++) {
				sub_bytes(idx, myData, s_sbox);
				shift_rows(idx, myData);
				mix_columns(idx, myData, s_mul2, s_mul3);
				add_round_key(idx, r, myData, key);
			}

			// For the last step we do NOT perform the mix_columns step.
			sub_bytes(idx, myData, s_sbox);
			shift_rows(idx, myData);
			add_round_key(idx, num_rounds, myData, key);

			// Encryption on this block is done, encrypted data is stored inplace.
#if 1 // If copied into local mem above, need to write it back down
			((uint32_t*)data)[idx*(BLOCKSIZE / 4) + 0] = ((uint32_t*)myData)[0];
			((uint32_t*)data)[idx*(BLOCKSIZE / 4) + 1] = ((uint32_t*)myData)[1];
			((uint32_t*)data)[idx*(BLOCKSIZE / 4) + 2] = ((uint32_t*)myData)[2];
			((uint32_t*)data)[idx*(BLOCKSIZE / 4) + 3] = ((uint32_t*)myData)[3];
#endif
		}

		 __global__ void core_decrypt_ecb(int N, uint8_t* data, const uint8_t* key, const int num_rounds) {
			// This performs the same steps as the encryption, but uses inverted values
			// to recover the plaintext.
			uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= N) {
				return;
			}

			// Get shared mem buffers
			__shared__ uint8_t s_rsbox[256];
			__shared__ uint8_t s_mul9[256];
			__shared__ uint8_t s_mulB[256];
			__shared__ uint8_t s_mulD[256];
			__shared__ uint8_t s_mulE[256];
			// If we have enough threads, let each one do one copy.
			if (N >= 256 && idx < 256) {
				s_rsbox[idx] = c_rsbox[idx];
				s_mul9[idx] = c_mul9[idx];
				s_mulB[idx] = c_mulB[idx];
				s_mulD[idx] = c_mulD[idx];
				s_mulE[idx] = c_mulE[idx];
			}
			// If we don't have enough blocks, just let thread 0 do it all.
			// If you only have 256 blocks you should be doing AES in CPU 
			// but thats none of my business *sips tea*.
			else if (idx == 0) {
				for (int i = 0; i < 256; i++) {
					s_rsbox[i] = c_rsbox[i];
					s_mul9[i] = c_mul9[i];
					s_mulB[i] = c_mulB[i];
					s_mulD[i] = c_mulD[i];
					s_mulE[i] = c_mulE[i];
				}
			}

			// Wait for shared memory to load
			__syncthreads();

			// Each thread running this function will act on ONE block in memory.
#if OPTIMIZED == 0 // Get a pointer to global mem
			uint8_t* myData = data + idx * BLOCKSIZE;
#else // Copy into local mem
			uint8_t myData[BLOCKSIZE];
			((uint32_t*)myData)[0] = ((uint32_t*)data)[idx*(BLOCKSIZE / 4) + 0];
			((uint32_t*)myData)[1] = ((uint32_t*)data)[idx*(BLOCKSIZE / 4) + 1];
			((uint32_t*)myData)[2] = ((uint32_t*)data)[idx*(BLOCKSIZE / 4) + 2];
			((uint32_t*)myData)[3] = ((uint32_t*)data)[idx*(BLOCKSIZE / 4) + 3];
#endif

			// Initial Round Key Addition
			// Each byte of the state is combined with a block of the round key using bitwise xor
			add_round_key(idx, num_rounds, myData, key);

			// We perform the next steps for a number of rounds
			// dependent on the flavor of AES.
			// This is done in the inverse compared to encryption
			for (int r = num_rounds - 1; r > 0; r--)
			{
				inv_shift_rows(idx, myData);
				inv_sub_bytes(idx, myData, s_rsbox);
				add_round_key(idx, r, myData, key);
				inv_mix_columns(idx, myData, s_mul9, s_mulB, s_mulD, s_mulE);
			}

			// For the last step we do NOT perform the mix_columns step.
			inv_shift_rows(idx, myData);
			inv_sub_bytes(idx, myData, s_rsbox);
			add_round_key(idx, 0, myData, key);

			// Decryption on this block is done, decrypted data is stored inplace.
#if 1 // If copied into local mem above, need to write it back down
			((uint32_t*)data)[idx*(BLOCKSIZE / 4) + 0] = ((uint32_t*)myData)[0];
			((uint32_t*)data)[idx*(BLOCKSIZE / 4) + 1] = ((uint32_t*)myData)[1];
			((uint32_t*)data)[idx*(BLOCKSIZE / 4) + 2] = ((uint32_t*)myData)[2];
			((uint32_t*)data)[idx*(BLOCKSIZE / 4) + 3] = ((uint32_t*)myData)[3];
#endif
		}

		 __global__ void core_xcrypt_ctr(int N, uint8_t* data, const uint8_t* key, const int num_rounds, const uint8_t * ctr) {
			 // Lenght of buffer is ALWAYS 128 bits == 16 bytes
			 // This is defined by AES Algorithm
			 uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
			 if (idx >= N) {
				 return;
			 }

			 // Get shared mem buffers
			 __shared__ uint8_t s_sbox[256];
			 __shared__ uint8_t s_mul2[256];
			 __shared__ uint8_t s_mul3[256];

			 // If we have enough threads, let each one do one copy.
			 if (N >= 256 && idx < 256) {
				 s_sbox[idx]  = c_sbox[idx];
				 s_mul2[idx]  = c_mul2[idx];
				 s_mul3[idx]  = c_mul3[idx];
			 }
			 // If we don't have enough blocks, just let thread 0 do it all.
			 // If you only have 256 blocks you should be doing AES in CPU 
			 // but thats none of my business *sips tea*.
			 else if (idx == 0) {
				 for (int i = 0; i < 256; i++) {
					 s_sbox[i] = c_sbox[i];
					 s_mul2[i] = c_mul2[i];
					 s_mul3[i] = c_mul3[i];
				 }
			 }

			 // Wait for shared memory to load
			 __syncthreads();

			 // Each thread running this function will act on ONE block in memory.
			 // (This is at least true in ECB mode, CTR mode might need its own kern.)
			 uint8_t* myData = data + idx * BLOCKSIZE;

			 // Copy the counter to this kernel and increment it by our idx
			 uint8_t myCtr[BLOCKSIZE];
			 for (int i = 0; i < BLOCKSIZE; i++) {
				 myCtr[i] = ctr[i];
			 }
			 ctr_increment(myCtr, idx);

			 // Lots of comments are pulled from https://en.wikipedia.org/wiki/Advanced_Encryption_Standard

			 // Above article mentions that SubBytes, ShiftRows, and MixColumns can be combined
			 // into 16 table lookups and 12 32bit XOR operations.
			 // If doing each block with 1 byte per thread, then each thread needs to perform
			 // one table lookup and at most one XOR.
			 // What are the memory implications? Can the tables be stored in RO memory to speed
			 // up the operations? Hoho! Use texture memory???

			 // Initial Round Key Addition
			 // Each byte of the state is combined with a block of the round key using bitwise xor
			 add_round_key(idx, 0, myCtr, key);

			 // We perform the next steps for a number of rounds
			 // dependent on the flavor of AES.
			 // Pass this in via a context? 
			 for (int r = 1; r < num_rounds; r++) {
				 sub_bytes(idx, myCtr, s_sbox);
				 shift_rows(idx, myCtr);
				 mix_columns(idx, myCtr, s_mul2, s_mul3);
				 add_round_key(idx, r, myCtr, key);
			 }

			 // For the last step we do NOT perform the mix_columns step.
			 sub_bytes(idx, myCtr, s_sbox);
			 shift_rows(idx, myCtr);
			 add_round_key(idx, num_rounds, myCtr, key);

			 // myCtr is now encrypted, xor it with our data before we leave
			 for (int i = 0; i < BLOCKSIZE; i++) {
				 myData[i] ^= myCtr[i];
			 }
		 }


		 __device__ void ctr_increment(uint8_t * ctr, int val)
		 {
			 // We want to increment the counter by VAL while
			 // avoiding as much divergence as possible.
			 // There will be SOME divergence, but hopefully minimal.
			 int remaining = val;
			 for (int b = BLOCKSIZE - 1; b >= 0 && remaining > 0; b--)
			 {
				 if (ctr[b] == 255)
				 {
					 ctr[b] = 0;
					 continue;
				 }
				 int added = __min(remaining, 255 - ctr[b]);
				 ctr[b] += added;
				 remaining -= added;
			 }
		 }

		 __device__ void add_round_key(int idx, uint8_t round, uint8_t * data, const uint8_t * key)
		{
			// Treat everything as a 1d array.
			// Matrix representations will be helpful later on, 
			// but this is clearer to me. We can easily parallelize this
#if OPTIMIZED == 0 // loop on 8bits
			for (uint8_t i = 0; i < 4; i++)
			{
				data[i] ^= key[(round * SUBKEY_SIZE) + i];
			}
#else // unrolled loop, 32bits
			((uint32_t*)data)[0] ^= ((uint32_t*)key)[(round * (SUBKEY_SIZE / 4)) + 0];
			((uint32_t*)data)[1] ^= ((uint32_t*)key)[(round * (SUBKEY_SIZE / 4)) + 1];
			((uint32_t*)data)[2] ^= ((uint32_t*)key)[(round * (SUBKEY_SIZE / 4)) + 2];
			((uint32_t*)data)[3] ^= ((uint32_t*)key)[(round * (SUBKEY_SIZE / 4)) + 3];
#endif

			// Parallelized
			//             Cache issues accessing key? Store Key in SMEM/TMEM
			//data[idx] ^= key[(round * SUBKEY_SIZE) + mod16(idx)];
		}

		__device__ void sub_bytes(int idx, uint8_t * data, uint8_t* sbox)
		{
			uint8_t i;
			for (i = 0; i < 16; ++i)
			{
				data[i] = sbox[data[i]];
			}

			// Again, this is EASILY parallelizable
			// data[idx] = sbox[data[mod16(idx)]];
			// However, I decided to make 1 kernel per block. This would require breaking that promise.
		}

		__device__ void inv_sub_bytes(int idx, uint8_t * data, uint8_t* rsbox)
		{
			uint8_t i;
			for (i = 0; i < 16; ++i)
			{
				data[i] = rsbox[data[i]];
			}

			// Again, this is EASILY parallelizable
			// data[idx] = rsbox[data[mod16(idx)]];
			// However, I decided to make 1 kernel per block. This would require breaking that promise.
		}

		__device__ void shift_rows(int idx, uint8_t * data)
		{
			// This is not as simple as the previous steps. If we want to parallelize this,
			// it will need to be a read, followed by a syncthreads, and then a write.
			// Could the overhead of a syncthreads be more expnsive then just reducing the
			// parallelism???

			// row0 -- No Shift
			// rows 1 to 3, shift left by row ammount

#if OPTIMIZED == 0 
			
			// 12 8bit reads / 12 8bit writes
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


#else // Unrolled 4 32bit reads 4 32bit writes + some math
			// Read our whole block in...
			uint32_t c0 = ((uint32_t*)data)[0];
			uint32_t c1 = ((uint32_t*)data)[1];
			uint32_t c2 = ((uint32_t*)data)[2];
			uint32_t c3 = ((uint32_t*)data)[3];

			// This looks a bit cryptic, but trust me, this is the end configuration
			// once the math is done.
			/*
			  | 0  1  2  3 |       | 0  5  A  F |
			  | 4  5  6  7 |  -->  | 4  9  E  3 |
			  | 8  9  A  B |  -->  | 8  D  2  7 |
			  | C  D  E  F |       | C  1  6  B |
			*/
			((uint32_t*)data)[0] = BYTE0(c0) | BYTE1(c1) | BYTE2(c2) | BYTE3(c3);
			((uint32_t*)data)[1] = BYTE0(c1) | BYTE1(c2) | BYTE2(c3) | BYTE3(c0);
			((uint32_t*)data)[2] = BYTE0(c2) | BYTE1(c3) | BYTE2(c0) | BYTE3(c1);
			((uint32_t*)data)[3] = BYTE0(c3) | BYTE1(c0) | BYTE2(c1) | BYTE3(c2);
#endif
		}

		__device__ void inv_shift_rows(int idx, uint8_t * data)
		{
			// This is not as simple as the previous steps. If we want to parallelize this,
			// it will need to be a read, followed by a syncthreads, and then a write.
			// Could the overhead of a syncthreads be more expnsive then just reducing the
			// parallelism???

			// row0 -- No Shift
			// rows 1 to 3, shift right (since we inv) by row ammount
#if OPTIMIZED == 0 // 12 8bit reads / 12 8bit writes
			uint8_t tmp[4];
			for (int row = 1; row < N_ROWS; row++) {
				tmp[0] = data[row + N_COLS * 0];
				tmp[1] = data[row + N_COLS * 1];
				tmp[2] = data[row + N_COLS * 2];
				tmp[3] = data[row + N_COLS * 3];
				data[row + N_COLS * 0] = tmp[mod4(0 + 4 - row)];
				data[row + N_COLS * 1] = tmp[mod4(1 + 4 - row)];
				data[row + N_COLS * 2] = tmp[mod4(2 + 4 - row)];
				data[row + N_COLS * 3] = tmp[mod4(3 + 4 - row)];
			}
#else // Unrolled 4 32bit reads 4 32bit writes + some math
			// Read our whole block in...
			uint32_t c0 = ((uint32_t*)data)[0];
			uint32_t c1 = ((uint32_t*)data)[1];
			uint32_t c2 = ((uint32_t*)data)[2];
			uint32_t c3 = ((uint32_t*)data)[3];

			// This looks a bit cryptic, but trust me, this is the end configuration
			// once the math is done.
			((uint32_t*)data)[0] = BYTE0(c0) | BYTE1(c3) | BYTE2(c2) | BYTE3(c1);
			((uint32_t*)data)[1] = BYTE0(c1) | BYTE1(c0) | BYTE2(c3) | BYTE3(c2);
			((uint32_t*)data)[2] = BYTE0(c2) | BYTE1(c1) | BYTE2(c0) | BYTE3(c3);
			((uint32_t*)data)[3] = BYTE0(c3) | BYTE1(c2) | BYTE2(c1) | BYTE3(c0);
#endif
		}

		__device__ void mix_columns(int idx, uint8_t * data, uint8_t* mul2, uint8_t* mul3)
		{
			// This is the most complicated step, but can be improved using our lookup tables.
			// Problem is, this is going to cause all sorts of contention because we read and write across
			// 4 different banks.
			/* Matrix used for mixin'
			   | 2  3  1  1 |
			   | 1  2  3  1 |
			   | 1  1  2  3 |
			   | 3  1  1  2 |
			*/

			// logic adapted from https://www.youtube.com/watch?v=bERjYzLqAfw
			
			for (int i = 0; i < N_COLS; i++) {
#if OPTIMIZED == 0  // 8bit RW
				uint8_t idx0 = i * N_COLS + 0;
				uint8_t idx1 = i * N_COLS + 1;
				uint8_t idx2 = i * N_COLS + 2;
				uint8_t idx3 = i * N_COLS + 3;

				// Hmmm... will compiler vectorize this as one read32?
				uint8_t d0 = data[idx0];
				uint8_t d1 = data[idx1];
				uint8_t d2 = data[idx2];
				uint8_t d3 = data[idx3];

				data[idx0] = mul2[d0] ^ mul3[d1] ^ d2       ^ d3;
				data[idx1] = d0 ^ mul2[d1] ^ mul3[d2] ^ d3;
				data[idx2] = d0 ^ d1       ^ mul2[d2] ^ mul3[d3];
				data[idx3] = mul3[d0] ^ d1       ^ d2       ^ mul2[d3];
#else // 32bit RW
				// One 32bit read from global
				uint32_t quarterblock = ((uint32_t*)data)[i];

				uint8_t in0 = ((quarterblock & 0x000000FF) >> 0);
				uint8_t in1 = ((quarterblock & 0x0000FF00) >> 8);
				uint8_t in2 = ((quarterblock & 0x00FF0000) >> 16);
				uint8_t in3 = ((quarterblock & 0xFF000000) >> 24);
				
				// Use lookup table, with shared mem this is p good
				uint32_t out0 = mul2[in0] ^ mul3[in1] ^ in2       ^ in3;
				uint32_t out1 = in0       ^ mul2[in1] ^ mul3[in2] ^ in3;
				uint32_t out2 = in0       ^ in1       ^ mul2[in2] ^ mul3[in3];
				uint32_t out3 = mul3[in0] ^ in1       ^ in2       ^ mul2[in3];
				
				// One 32bit write to global
				((uint32_t*)data)[i] = out0 | (out1 << 8) | (out2 << 16) | (out3 << 24);
#endif

				// Compute at runtime - Ended up slower, higher instruction cost, still high LG Throttle
				//data[idx0] = rt_mul2(in0) ^ rt_mul3(in1) ^ in2          ^ in3;
				//data[idx1] = in0          ^ rt_mul2(in1) ^ rt_mul3(in2) ^ in3;
				//data[idx2] = in0          ^ in1          ^ rt_mul2(in2) ^ rt_mul3(in3);
				//data[idx3] = rt_mul3(in0) ^ in1          ^ in2          ^ rt_mul2(in3);
			}

			// This can be parallelized by block, but going deeper will incur
			// contention penalties. Not worth the trouble.
		}

		__device__ void inv_mix_columns(int idx, uint8_t * data, 
			uint8_t* mul9, uint8_t* mulB, uint8_t* mulD, uint8_t* mulE)
		{
			// Inverse mix columns -- This is the same procedure, but we use MORE lookup tables!
			// The inmverse operation defines a differnt table, which will increase out total
			// operations. Interesting to see how this compares against the forward case...
			/* Matrix used for mixin'
			   | E  B  D  9 |
			   | 9  E  B  D |
			   | D  9  E  B |
			   | B  D  9  E |
			*/

			for (int i = 0; i < N_COLS; i++) {
#if OPTIMIZED == 0 // 8bit RW
				uint8_t idx0 = i * N_COLS + 0;
				uint8_t idx1 = i * N_COLS + 1;
				uint8_t idx2 = i * N_COLS + 2;
				uint8_t idx3 = i * N_COLS + 3;

				// Hmmm... will compiler vectorize this as one read32?
				uint8_t d0 = data[idx0];
				uint8_t d1 = data[idx1];
				uint8_t d2 = data[idx2];
				uint8_t d3 = data[idx3];

				data[idx0] = mulE[d0] ^ mulB[d1] ^ mulD[d2] ^ mul9[d3];
				data[idx1] = mul9[d0] ^ mulE[d1] ^ mulB[d2] ^ mulD[d3];
				data[idx2] = mulD[d0] ^ mul9[d1] ^ mulE[d2] ^ mulB[d3];
				data[idx3] = mulB[d0] ^ mulD[d1] ^ mul9[d2] ^ mulE[d3];

				uint8_t base = i * N_COLS;
#else // 32bit RW
				// One 32bit read from global
				uint32_t quarterblock = ((uint32_t*)data)[i];

				uint8_t in0 = ((quarterblock & 0x000000FF) >> 0);
				uint8_t in1 = ((quarterblock & 0x0000FF00) >> 8);
				uint8_t in2 = ((quarterblock & 0x00FF0000) >> 16);
				uint8_t in3 = ((quarterblock & 0xFF000000) >> 24);

				// Use lookup table, with shared mem this is p good
				uint32_t out0 = mulE[in0] ^ mulB[in1] ^ mulD[in2] ^ mul9[in3];
				uint32_t out1 = mul9[in0] ^ mulE[in1] ^ mulB[in2] ^ mulD[in3];
				uint32_t out2 = mulD[in0] ^ mul9[in1] ^ mulE[in2] ^ mulB[in3];
				uint32_t out3 = mulB[in0] ^ mulD[in1] ^ mul9[in2] ^ mulE[in3];

				// One 32bit write to global
				((uint32_t*)data)[i] = out0 | (out1 << 8) | (out2 << 16) | (out3 << 24);
#endif
			}
		}
    }
}
