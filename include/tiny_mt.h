/**********************************************************************
 * Copyright by https://github.com/ESultanik/mtwister.git
 *********************************************************************/
#ifndef TINY_MT_H 
#define TINY_MT_H

#define UPPER_MASK              0x80000000
#define LOWER_MASK              0x7fffffff
#define TEMPERING_MASK_B        0x9d2c5680
#define TEMPERING_MASK_C        0xefc60000
#define STATE_VECTOR_LENGTH 624
#define STATE_VECTOR_M      397

typedef struct tagMTRand {
  unsigned long mt[STATE_VECTOR_LENGTH];
  int index;
} MTRand;

MTRand seedRand(unsigned long seed);
unsigned long genRandLong(MTRand* rand);
double genRand(MTRand* rand);

#endif
