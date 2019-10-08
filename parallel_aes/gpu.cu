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
		__host__  __forceinline__ T mod16(T n) {
			return n & 15;
		}

		template<typename T>
		__host__  __forceinline__ T mod4(T n) {
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
			expandKey(key, expkey, get_key_len(flavor), get_num_rounds(flavor));

			// Do the actual encryption inplace.
			for (uint32_t i = 0; i < datalen; i += BLOCKSIZE) {
				core_encrypt(&data[i], expkey, get_num_rounds(flavor));
			}
		}

		__host__ void decrypt_ecb(const AESType& flavor, uint8_t * key, uint8_t * data, uint32_t datalen)
		{
			// Straightforward, nothing fancy is done here.
			// ECB is inherently insecure because it does not diffuse the data
			assert(mod16(datalen) == 0);

			// Expand the key
			uint8_t* expkey = (uint8_t*)malloc(get_exp_key_len(flavor));
			expandKey(key, expkey, get_key_len(flavor), get_num_rounds(flavor));

			// Do the actual decryption inplace.
			for (uint32_t i = 0; i < datalen; i += BLOCKSIZE) {
				core_decrypt(&data[i], expkey, get_num_rounds(flavor));
			}
		}

		void encrypt_ctr(const AESType& flavor, uint8_t * key, uint32_t ctr, uint8_t * data, uint32_t datalen)
		{
		}

		void decrypt_ctr(const AESType& flavor, uint8_t * key, uint32_t ctr, uint8_t * data, uint32_t datalen)
		{
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
			n[0] = sbox[n[0]];
			n[1] = sbox[n[1]];
			n[2] = sbox[n[2]];
			n[3] = sbox[n[3]];
		}

		__host__ void expandKey(uint8_t* ogkey, uint8_t* expkey, uint32_t keysize, uint32_t num_rounds) {
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
				else if (i % N == 4) {
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

		 __host__ void core_encrypt(uint8_t* data, const uint8_t* key, const int num_rounds) {
			// Lenght of buffer is ALWAYS 128 bits == 16 bytes
			// This is defined by AES Algorithm

			// Lots of comments are pulled from https://en.wikipedia.org/wiki/Advanced_Encryption_Standard
			
			// Above article mentions that SubBytes, ShiftRows, and MixColumns can be combined
			// into 16 table lookups and 12 32bit XOR operations.
			// If doing each block with 1 byte per thread, then each thread needs to perform
			// one table lookup and at most one XOR.
			// What are the memory implications? Can the tables be stored in RO memory to speed
			// up the operations? Hoho! Use texture memory???

			// Initial Round Key Addition
			// Each byte of the state is combined with a block of the round key using bitwise xor
			add_round_key(0, data, key);

			// We perform the next steps for a number of rounds
			// dependent on the flavor of AES.
			// Pass this in via a context? 
			for (int r = 1; r < num_rounds; r++) {
				sub_bytes(data);
				shift_rows(data);
				mix_columns(data);
				add_round_key(r, data, key);
			}

			// For the last step we do NOT perform the mix_columns step.
			sub_bytes(data);
			shift_rows(data);
			add_round_key(num_rounds, data, key);

			// Encryption on this block is done, encrypted data is stored inplace.
		}

		 __host__ void core_decrypt(uint8_t* data, const uint8_t* key, const int num_rounds) {
			// This performs the same steps as the encryption, but uses inverted values
			// to recover the plaintext.

			// Initial Round Key Addition
			// Each byte of the state is combined with a block of the round key using bitwise xor
			add_round_key(0, data, key);

			// We perform the next steps for a number of rounds
			// dependent on the flavor of AES.
			// This is done in the inverse compared to encryption
			for (int r = num_rounds - 1; r > 0; r--)
			{
				inv_shift_rows(data);
				inv_sub_bytes(data);
				add_round_key(r, data, key);
				inv_mix_columns(data);
			}

			// For the last step we do NOT perform the mix_columns step.
			inv_shift_rows(data);
			inv_sub_bytes(data);
			add_round_key(0, data, key);

			// Decryption on this block is done, decrypted data is stored inplace.
		}

		__host__  void add_round_key(uint8_t round, uint8_t * data, const uint8_t * key)
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

		__host__  void sub_bytes(uint8_t * data)
		{
			uint8_t i;
			for (i = 0; i < 16; ++i)
			{
				data[i] = sbox[data[i]];
			}

			// Again, this is EASILY parallelizable
			//data[idx] = sbox[data[mod16(idx)]];
		}

		__host__  void inv_sub_bytes(uint8_t * data)
		{
			uint8_t i;
			for (i = 0; i < 16; ++i)
			{
				data[i] = rsbox[data[i]];
			}

			// Again, this is EASILY parallelizable
			//data[idx] = rsbox[data[mod16(idx)]];
		}

		__host__  void shift_rows(uint8_t * data)
		{
			uint8_t tmp[4];
			// This is not as simple as the previous steps. If we want to parallelize this,
			// it will need to be a read, followed by a syncthreads, and then a write.
			// Could the overhead of a syncthreads be more expnsive then just reducing the
			// parallelism???

			// row0 -- No Shift
			// rows 1 to 3, shift left by row ammount
			for (int row = 1; row < N_ROWS; row++) {
				tmp[0] = data[row * N_COLS + 0];
				tmp[1] = data[row * N_COLS + 1];
				tmp[2] = data[row * N_COLS + 2];
				tmp[3] = data[row * N_COLS + 3];
				data[row * N_COLS + 0] = tmp[mod4(0 + row)];
				data[row * N_COLS + 1] = tmp[mod4(1 + row)];
				data[row * N_COLS + 2] = tmp[mod4(2 + row)];
				data[row * N_COLS + 3] = tmp[mod4(3 + row)];
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

		__host__  void inv_shift_rows(uint8_t * data)
		{
			uint8_t tmp[4];
			// This is not as simple as the previous steps. If we want to parallelize this,
			// it will need to be a read, followed by a syncthreads, and then a write.
			// Could the overhead of a syncthreads be more expnsive then just reducing the
			// parallelism???

			// row0 -- No Shift
			// rows 1 to 3, shift right (since we inv) by row ammount
			for (int row = 1; row < N_ROWS; row++) {
				tmp[0] = data[row * N_COLS + 0];
				tmp[1] = data[row * N_COLS + 1];
				tmp[2] = data[row * N_COLS + 2];
				tmp[3] = data[row * N_COLS + 3];
				data[row * N_COLS + 0] = tmp[mod4(0 + 4 - row)];
				data[row * N_COLS + 1] = tmp[mod4(1 + 4 - row)];
				data[row * N_COLS + 2] = tmp[mod4(2 + 4 - row)];
				data[row * N_COLS + 3] = tmp[mod4(3 + 4 - row)];
			}
		}

		__host__  void mix_columns(uint8_t * data)
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
				uint8_t idx0 = 0 + N_COLS * i;
				uint8_t idx1 = 1 + N_COLS * i;
				uint8_t idx2 = 2 + N_COLS * i;
				uint8_t idx3 = 3 + N_COLS * i;

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

		__host__  void inv_mix_columns(uint8_t * data)
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
				uint8_t idx0 = 0 + N_COLS * i;
				uint8_t idx1 = 1 + N_COLS * i;
				uint8_t idx2 = 2 + N_COLS * i;
				uint8_t idx3 = 3 + N_COLS * i;

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
