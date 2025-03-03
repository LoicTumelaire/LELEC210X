/*
 * packet.c
 */
#include "aes_ref.h"
#include "config.h"
#include "packet.h"
#include "main.h"
#include "utils.h"
#include <string.h>

#include "stm32l4xx_hal.h"
#include "stm32l4xx_hal_cryp.h"

extern CRYP_HandleTypeDef hcryp;

void tag_cbc_mac(uint8_t *tag, const uint8_t *msg, size_t msg_len) {
    size_t padded_len = (msg_len + 15) & ~15; // Round up to next multiple of 16
    uint8_t padded_msg[padded_len];

    // Copy original message
    memcpy(padded_msg, msg, msg_len);

    uint8_t pad_value = padded_len - msg_len;
    memset(padded_msg + msg_len, 0, pad_value);

    uint8_t cipher_output[padded_len];

    if (HAL_CRYP_AESCBC_Encrypt(&hcryp, padded_msg, padded_len, cipher_output, HAL_MAX_DELAY) != HAL_OK) {
        DEBUG_PRINT("Error in AES encryption\r\n");
    }

    // CBC-MAC is the last 16 bytes of the ciphertext
    memcpy(tag, &cipher_output[padded_len - 16], 16);
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

    tag_cbc_mac(packet + payload_len + PACKET_HEADER_LENGTH, packet, payload_len + PACKET_HEADER_LENGTH);
    return packet_len;
}
