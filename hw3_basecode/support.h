#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

// Matrix Structure declaration
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    float* elements;
} Matrix;

#define FILTER_SIZE 5
#define TILE_SIZE 16
__constant__ float Mc[FILTER_SIZE][FILTER_SIZE];

Matrix allocateMatrix(unsigned height, unsigned width);
void initMatrix(Matrix mat);
Matrix allocateDeviceMatrix(unsigned height, unsigned width);
void copyToDeviceMatrix(Matrix dst, Matrix src);
void copyFromDeviceMatrix(Matrix dst, Matrix src);
void verify(Matrix M, Matrix  N, Matrix P);
void freeMatrix(Matrix mat);
void freeDeviceMatrix(Matrix mat);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif
