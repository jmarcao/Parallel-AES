#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <bitset>
#include <assert.h>
#include "common.h"
#include "gpu.h"

#define N_COLS 4
#define N_ROWS 4
#define SUBKEY_SIZE 16

namespace PAES {
    namespace GPU {
        using PAES::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		// A small helper for quick mod16, since it'll be used alot when accessing keys
		// in sharedmem/texmem. 
		template<typename T>
		__host__ __device__ __forceinline__ T mod16(T n) {
			return n & 15;
		}

		template<typename T>
		__host__ __device__ __forceinline__ T mod4(T n) {
			return n & 3;
		}

		// Gets lenght of key in bytes for flavor
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
			int blocks = datalen / BLOCKSIZE;
			int threads = blocks;

			// Call the kernels to get to work!
			core_encrypt_ecb << <1, threads >> > (d_data, d_expkey, get_num_rounds(flavor));
			cudaDeviceSynchronize();

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
			int blocks = datalen / BLOCKSIZE;
			int threads = blocks;

			// Call the kernels to get to work!
			core_decrypt_ecb << <1, threads >> > (d_data, d_expkey, get_num_rounds(flavor));
			cudaDeviceSynchronize();

			// Retrieve the data from the device
			cudaMemcpy(data, d_data, sizeof(uint8_t) * datalen, cudaMemcpyDeviceToHost);

			// Free CUDA memory
			cudaFree(d_data);
			cudaFree(d_expkey);
		}

		void encrypt_ctr(const AESType& flavor, uint8_t * key, uint8_t * ctr, uint8_t * data, uint32_t datalen)
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
			int blocks = datalen / BLOCKSIZE;
			int threads = blocks;

			// Start the kernels. Each kernel will increment the counter
			// based on their index.
			core_xcrypt_ctr << <1, threads >> > (d_data, d_expkey, get_num_rounds(flavor), d_ctr);
			cudaDeviceSynchronize();

			// Retrieve the data from the device
			cudaMemcpy(data, d_data, sizeof(uint8_t) * datalen, cudaMemcpyDeviceToHost);

			// Free CUDA memory
			cudaFree(d_data);
			cudaFree(d_expkey);
			cudaFree(d_ctr);
		}

		void decrypt_ctr(const AESType& flavor, uint8_t * key, uint8_t * ctr, uint8_t * data, uint32_t datalen)
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

			// Code adapted from tiny-aes since this is not parallelized
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

		 __global__ void core_encrypt_ecb(uint8_t* data, const uint8_t* key, const int num_rounds) {
			// Lenght of buffer is ALWAYS 128 bits == 16 bytes
			// This is defined by AES Algorithm
			uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

			// Each thread running this function will act on ONE block in memory.
			// (This is at least true in ECB mode, CTR mode might need its own kern.)
			uint8_t* myData = data + idx * BLOCKSIZE;

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
				sub_bytes(idx, myData);
				shift_rows(idx, myData);
				mix_columns(idx, myData);
				add_round_key(idx, r, myData, key);
			}

			// For the last step we do NOT perform the mix_columns step.
			sub_bytes(idx, myData);
			shift_rows(idx, myData);
			add_round_key(idx, num_rounds, myData, key);

			// Encryption on this block is done, encrypted data is stored inplace.
		}

		 __global__ void core_decrypt_ecb(uint8_t* data, const uint8_t* key, const int num_rounds) {
			// This performs the same steps as the encryption, but uses inverted values
			// to recover the plaintext.
			uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

			// Each thread running this function will act on ONE block in memory.
			// (This is at least true in ECB mode, CTR mode might need its own kern.)
			uint8_t* myData = data + idx * BLOCKSIZE;

			// Initial Round Key Addition
			// Each byte of the state is combined with a block of the round key using bitwise xor
			add_round_key(idx, num_rounds, myData, key);

			// We perform the next steps for a number of rounds
			// dependent on the flavor of AES.
			// This is done in the inverse compared to encryption
			for (int r = num_rounds - 1; r > 0; r--)
			{
				inv_shift_rows(idx, myData);
				inv_sub_bytes(idx, myData);
				add_round_key(idx, r, myData, key);
				inv_mix_columns(idx, myData);
			}

			// For the last step we do NOT perform the mix_columns step.
			inv_shift_rows(idx, myData);
			inv_sub_bytes(idx, myData);
			add_round_key(idx, 0, myData, key);

			// Decryption on this block is done, decrypted data is stored inplace.
		}

		 __global__ void core_xcrypt_ctr(uint8_t* data, const uint8_t* key, const int num_rounds, const uint8_t * ctr) {
			 // Lenght of buffer is ALWAYS 128 bits == 16 bytes
			 // This is defined by AES Algorithm
			 uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

			 // Each thread running this function will act on ONE block in memory.
			 // (This is at least true in ECB mode, CTR mode might need its own kern.)
			 uint8_t* myData = data + idx * BLOCKSIZE;

			 // Copy the counter to this kernel and increment it
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
				 sub_bytes(idx, myCtr);
				 shift_rows(idx, myCtr);
				 mix_columns(idx, myCtr);
				 add_round_key(idx, r, myCtr, key);
			 }

			 // For the last step we do NOT perform the mix_columns step.
			 sub_bytes(idx, myCtr);
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
			for (uint8_t i = 0; i < 16; i++)
			{
				data[i] ^= key[(round * SUBKEY_SIZE) + i];
			}

			// Parallelized
			//             Cache issues accessing key? Store Key in SMEM/TMEM
			//data[idx] ^= key[(round * SUBKEY_SIZE) + mod16(idx)];
		}

		__device__ void sub_bytes(int idx, uint8_t * data)
		{
			uint8_t i;
			for (i = 0; i < 16; ++i)
			{
				data[i] = sbox[data[i]];
			}

			// Again, this is EASILY parallelizable
			//data[idx] = sbox[data[mod16(idx)]];
		}

		__device__ void inv_sub_bytes(int idx, uint8_t * data)
		{
			uint8_t i;
			for (i = 0; i < 16; ++i)
			{
				data[i] = rsbox[data[i]];
			}

			// Again, this is EASILY parallelizable
			//data[idx] = rsbox[data[mod16(idx)]];
		}

		__device__ void shift_rows(int idx, uint8_t * data)
		{
			uint8_t tmp[4];
			// This is not as simple as the previous steps. If we want to parallelize this,
			// it will need to be a read, followed by a syncthreads, and then a write.
			// Could the overhead of a syncthreads be more expnsive then just reducing the
			// parallelism???

			// row0 -- No Shift
			// rows 1 to 3, shift left by row ammount
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

			// Parallelize
			// Launch N_ELEMENTS % 4 kernels
			//row = idx;
			//tmp[0] = data[row * N_COLS + 0];
			//tmp[1] = data[row * N_COLS + 1];
			//tmp[2] = data[row * N_COLS + 2];
			//tmp[3] = data[row * N_COLS + 3];
			//data[1 * N_COLS + 0] = tmp[mod4(0 + row)];
			//data[1 * N_COLS + 1] = tmp[mod4(1 + row)];
			//data[1 * N_COLS + 2] = tmp[mod4(2 + row)];
			//data[1 * N_COLS + 3] = tmp[mod4(3 + row)];
		}

		__device__ void inv_shift_rows(int idx, uint8_t * data)
		{
			uint8_t tmp[4];
			// This is not as simple as the previous steps. If we want to parallelize this,
			// it will need to be a read, followed by a syncthreads, and then a write.
			// Could the overhead of a syncthreads be more expnsive then just reducing the
			// parallelism???

			// row0 -- No Shift
			// rows 1 to 3, shift right (since we inv) by row ammount
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
		}

		__device__ void mix_columns(int idx, uint8_t * data)
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
				data[idx1] = d0       ^ mul2[d1] ^ mul3[d2] ^ d3;
				data[idx2] = d0       ^ d1       ^ mul2[d2] ^ mul3[d3];
				data[idx3] = mul3[d0] ^ d1       ^ d2       ^ mul2[d3];
			}

			// This can be parallelized by block, but going deeper will incur
			// contention penalties. Probably not worth the trouble.
		}

		__device__ void inv_mix_columns(int idx, uint8_t * data)
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
			}
		}
    }
}
