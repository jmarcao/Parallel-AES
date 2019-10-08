/**
 * Parallel AES for CIS565 Project 4
 * John Marcao
 */

#include <cstdio>
#include <parallel_aes/cpu.h>
#include <parallel_aes/gpu.h>
#include "testing_helpers.hpp"

const int SIZE = 1 << 10; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

int main(int argc, char* argv[]) {

	// TODO: Parse opts. For now, use variables

	// ECB Mode
	// CPU Test
	// CPU_AES_ECB_128
	// CPU_AES_ECB_192
	// CPU_AES_ECB_256

	// GPU-Naive
	// GPU_N_AES_ECB_128
	// GPU_N_AES_ECB_192
	// GPU_N_AES_ECB_256

	// GPU-Optimized
	// GPU_O_AES_ECB_128
	// GPU_O_AES_ECB_192
	// GPU_O_AES_ECB_256

	// CTR Mode
	// CPU Test
	// CPU_AES_CTR_128
	// CPU_AES_CTR_192
	// CPU_AES_CTR_256

	// GPU-Naive
	// GPU_N_AES_CTR_128
	// GPU_N_AES_CTR_192
	// GPU_N_AES_CTR_256

	// GPU-Optimized
	// GPU_O_AES_CTR_128
	// GPU_O_AES_CTR_192
	// GPU_O_AES_CTR_256


    system("pause"); // stop Win32 console from closing on exit
}
