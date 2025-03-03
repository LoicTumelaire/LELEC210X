/*
 * config.h
 */

#ifndef INC_CONFIG_H_
#define INC_CONFIG_H_

#include <stdio.h>

// Runtime parameters
#define MAIN_APP 0
#define EVAL_RADIO 1

#define RUN_CONFIG MAIN_APP

// Radio parameters
#define ENABLE_RADIO 1

// General UART enable/disable (disable for low-power operation)
#define ENABLE_UART 1

// In continuous mode, we start and stop continuous acquisition on button press.
// In non-continuous mode, we send a single packet on button press.
#define CONTINUOUS_ACQ 0

// Spectrogram parameters
#define SAMPLES_PER_MELVEC 512
#define MELVEC_LENGTH 24
#define N_MELVECS 20

// Enable performance measurements
#define PERF_COUNT 0

// Enable debug print
#define DEBUGP 1

#if (DEBUGP == 1)
#define DEBUG_PRINT(...) do{ printf(__VA_ARGS__ ); } while( 0 )
#else
#define DEBUG_PRINT(...) do{ } while ( 0 )
#endif

/////////////////////////////////CUSTOM CONFIG////////////////////////////////////////////

/////	1 = TRUE (activated) 0 = INITAL behavior (deactivated)

// Enable MCU sleep mode while waiting
#define MCU_SLEEPMODE 1

// Enable MCU low power run mode
#define MCU_LOW_POwER_MODE 1

// Enable hardware AES tag computation acceleration
#define AES_HW_ACCELERATION 1

// Enable sleep mode for the radio when no transmission occurs
#define RADIO_SLEEP_MODE 1

// Disable the ticking clock when the MCU is in sleep mode
#define MCU_TICK_STOP 0

// LED used for debugging
#define LED 1

#endif /* INC_CONFIG_H_ */
