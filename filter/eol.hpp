#pragma once

#include "../file/File.hpp"
#include "../Encoder.hpp"
#include "../CharacterNames.hpp"
#include "Filter.hpp"
#include <cstdint>

/**
 * End of line transform
 */
class EolFilter : public Filter {
public:
  void encode(File *in, File *out, uint64_t size, int /*info*/, int & /*headerSize*/) override {
    uint8_t b = 0;
    uint8_t pB = 0;
    for( uint64_t i = 0; i < size; i++ ) {
      b = in->getchar();
      if( pB == CARRIAGE_RETURN && b != NEW_LINE ) {
        out->putChar(pB);
      }
      if( b != CARRIAGE_RETURN ) {
        out->putChar(b);
      }
      pB = b;
    }
    if( b == CARRIAGE_RETURN ) {
      out->putChar(b);
    }
  }

  uint64_t decode(File * /*in*/, File *out, FMode fMode, uint64_t size, uint64_t &diffFound) override {
    uint8_t b = 0;
    uint64_t count = 0;
    for( uint64_t i = 0; i < size; i++, count++ ) {
      if((b = encoder->decompressByte(&encoder->predictorMain)) == NEW_LINE ) {
        if( fMode == FMode::FDECOMPRESS ) {
          out->putChar(CARRIAGE_RETURN);
        } else if( fMode == FMode::FCOMPARE ) {
          if( out->getchar() != CARRIAGE_RETURN && (diffFound == 0)) {
            diffFound = size - i;
            break;
          }
        }
        count++;
      }
      if( fMode == FMode::FDECOMPRESS ) {
        out->putChar(b);
      } else if( fMode == FMode::FCOMPARE ) {
        if( b != out->getchar() && (diffFound == 0)) {
          diffFound = size - i;
          break;
        }
      }
      if( fMode == FMode::FDECOMPRESS && ((i & 0xFFF) == 0)) {
        encoder->printStatus();
      }
    }
    return count;
  }
};
