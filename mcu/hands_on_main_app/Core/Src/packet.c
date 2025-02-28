/*
 * packet.c
 */
#include "aes_ref.h"
#include "aes.h"
#include "config.h"
#include "packet.h"
#include "main.h"
#include "utils.h"
#include <string.h>

const uint8_t AES_Key[16]  = {
                            0x00,0x00,0x00,0x00,
							0x00,0x00,0x00,0x00,
							0x00,0x00,0x00,0x00,
							0x00,0x00,0x00,0x00};

void tag_cbc_mac(uint8_t *tag, const uint8_t *msg, size_t msg_len) {
	// Allocate a buffer of the key size to store the input and result of AES
	// uint32_t[4] is 4*(32/8)= 16 bytes long

#if (AES_HW_ACCELERATION == 1)

	size_t size = msg_len+16-msg_len%16;
	uint8_t plain[size];
	uint8_t cipher[size];

	memcpy(plain, msg, msg_len);

	HAL_CRYP_AESCBC_Encrypt(&hcryp, plain, size, cipher, 1000);
	
    // Copy the result of CBC-MAC-AES to the tag.
	memcpy(tag, cipher + msg_len - 16, 16);

#else
	uint8_t state[16] = {0};

    // TO DO : Complete the CBC-MAC_AES, without allocating memory and its stack space usage must be constant

	// Parse x into blocks (x1, x2, . . . , xn) such that the length of each block is 16 bytes (except for xn)
	// If the length of xn is not 16, append as many zero bytes to xn to extend it to 16 bytes.
	// s ← 0**16 (Where 0**16 denotes the string made of 16 zero bytes.)
	// void AES128_encrypt ( unsigned char * block , const unsigned char * key ) ;

	// Process each 16-byte block
	for (size_t i = 0; i < msg_len; i += 16) {
		// XOR the current block with the state
		for (size_t j = 0; j < 16; j++) {
			if (i + j < msg_len) {
				state[j] ^= msg[i + j];// XOR the current block with the state
			} else {
				state[j] ^= 0x00; // Padding with zero if the block is less than 16 bytes
			}
		}
		// Encrypt the state with AES
		AES128_encrypt(state, AES_Key);
	}

    // Copy the result of CBC-MAC-AES to the tag.
	memcpy(tag, state, 16);

#endif
}
// Assumes payload is already in place in the packet
int make_packet(uint8_t *packet, size_t payload_len, uint8_t sender_id, uint32_t serial) {
    size_t packet_len = payload_len + PACKET_HEADER_LENGTH + PACKET_TAG_LENGTH;
    // Initially, the whole packet header is set to 0s
    //memset(packet, 0, PACKET_HEADER_LENGTH);
    // So is the tag
	//memset(packet + payload_len + PACKET_HEADER_LENGTH, 0, PACKET_TAG_LENGTH);

	// TO DO :  replace the two previous command by properly
	//			setting the packet header with the following structure :
	/***************************************************************************
	 *    Field       	Length (bytes)      Encoding        Description
	 ***************************************************************************
	 *  r 					1 								Reserved, set to 0.
	 * 	emitter_id 			1 					BE 			Unique id of the sensor node.
	 *	payload_length 		2 					BE 			Length of app_data (in bytes).
	 *	packet_serial 		4 					BE 			Unique and incrementing id of the packet.
	 *	app_data 			any 							The feature vectors.
	 *	tag 				16 								Message authentication code (MAC).
	 *
	 *	Note : BE refers to Big endian
	 *		 	Use the structure 	packet[x] = y; 	to set a byte of the packet buffer
	 *		 	To perform bit masking of the specific bytes you want to set, you can use
	 *		 		- bitshift operator (>>),
	 *		 		- and operator (&) with hex value, e.g.to perform 0xFF
	 *		 	This will be helpful when setting fields that are on multiple bytes.
	*/

	// Set the reserved field to 0
	packet[0] = 0x00;
	// Set the emitter_id field
	packet[1] = sender_id;
	// Set the payload_length field
	*(uint16_t *)(packet + 2) = __builtin_bswap16(payload_len);
	// Set the packet_serial field
	*(uint32_t *)(packet + 4) = __builtin_bswap32(serial);

	// app_data is already in place in the packet

	// For the tag field, you have to calculate the tag. The function call below is correct but
	// tag_cbc_mac function, calculating the tag, is not implemented.
    tag_cbc_mac(packet + payload_len + PACKET_HEADER_LENGTH, packet, payload_len + PACKET_HEADER_LENGTH);

    return packet_len;
}
