#pragma once

#include "common.h"
//#include "tiny-AES-c/aes.h" // Use tiny-aes for CPU. 

namespace PAES {
    namespace CPU {
		PAES::Common::PerformanceTimer& timer();

		//// All encryption done inplace.
		//// Electronic Codebook Mode (ECB)
		//void CPU_AES_ECB_Encrypt(cis565::AES128_Ctx ctx, uint8_t* data, uint32_t len);
		//void CPU_AES_ECB_Encrypt(cis565::AES192_Ctx ctx, uint8_t* data, uint32_t len);
		//void CPU_AES_ECB_Encrypt(cis565::AES256_Ctx ctx, uint8_t* data, uint32_t len);

		//void CPU_AES_ECB_Decrypt(cis565::AES128_Ctx ctx, uint8_t* data, uint32_t len);
		//void CPU_AES_ECB_Decrypt(cis565::AES192_Ctx ctx, uint8_t* data, uint32_t len);
		//void CPU_AES_ECB_Decrypt(cis565::AES256_Ctx ctx, uint8_t* data, uint32_t len);

		//// Counter Mode (CTR)
		//void CPU_AES_CTR_Encrypt(cis565::AES128_Ctx ctx, uint8_t* data, uint32_t len);
		//void CPU_AES_CTR_Encrypt(cis565::AES192_Ctx ctx, uint8_t* data, uint32_t len);
		//void CPU_AES_CTR_Encrypt(cis565::AES256_Ctx ctx, uint8_t* data, uint32_t len);

		//void CPU_AES_CTR_Decrypt(cis565::AES128_Ctx ctx, uint8_t* data, uint32_t len);
		//void CPU_AES_CTR_Decrypt(cis565::AES192_Ctx ctx, uint8_t* data, uint32_t len);
		//void CPU_AES_CTR_Decrypt(cis565::AES256_Ctx ctx, uint8_t* data, uint32_t len);
    }
}
