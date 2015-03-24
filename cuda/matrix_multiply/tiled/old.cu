// A tiled matrix multiplication program
//

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"
#include <iostream>
using std::endl;
using std::cout;

#include "cutil_inline.h"

// define size of matrix
#define SIZE 512
#define TILE_WIDTH 16

__global__ void MatrixMulKernel(float * Ad,float * Bd,float * Cd,int len,int tile_width)
{

int row = blockIdx.y*tile_width + threadIdx.y;
int col = blockIdx.x*tile_width + threadIdx.x;
float sum=0.0;
for (int k = 0;k < len;k++) {
sum += Ad[row*len+k] * Bd[k*len+col];
}

Cd[row*len+col] = sum;

}

int main() {

float Ah[SIZE*SIZE],Bh[SIZE*SIZE],Ch[SIZE*SIZE];
float *Ad,*Bd,*Cd;

dim3 dimGrid(SIZE/TILE_WIDTH,SIZE/TILE_WIDTH);
dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);

int devID;
cudaDeviceProp props;

// this can be set to 0 or 1 on degennes
// 0 = GeForce GTX 480
// 1 = Tesla C2050
cudaSetDevice(0);

// get number of SMs on this GPU
cudaGetDevice(&devID);
cudaGetDeviceProperties(&props, devID);

printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major,props.minor);

// alternatively, we could do this on the device
// then there would be no need to copy the host data
// onto the device
int i;
for (i=0;i<SIZE*SIZE;i++) {

Ah[i] = float(i);
Bh[i] = float(SIZE)*float(SIZE)-float(i)-float(1);
Ch[i] = float(0);

}

// allocate space on device
int size = SIZE*SIZE*sizeof(float);
cudaMalloc(&Ad,size);
cudaMalloc(&Bd,size);
cudaMalloc(&Cd,size);

//copy data to device
cudaMemcpy(Ad,Ah,size,cudaMemcpyHostToDevice);
cudaMemcpy(Bd,Bh,size,cudaMemcpyHostToDevice);
cudaMemcpy(Cd,Ch,size,cudaMemcpyHostToDevice);

// setup the timer
unsigned int timer = 0;
cutCreateTimer(&timer);
cutStartTimer(timer);

// invoke the kernel here
MatrixMulKernel<<<dimGrid,dimBlock>>>(Ad,Bd,Cd,SIZE,TILE_WIDTH);

// sync threads then stop and destroy timer
cudaThreadSynchronize();
cutStopTimer(timer);

// the cutGetTimerValue function returns time in ms
// divide by 1000 to get to seconds
double dSeconds = cutGetTimerValue(timer)/1000.0;
printf("kernel time (seconds) : %.5f\n",dSeconds);

// copy results back to host
cudaMemcpy(Ch,Cd,size,cudaMemcpyDeviceToHost);

// output results
for (i=0;i<SIZE*SIZE;i++) {
cout << "i= " << i << " Ch[i]= " << Ch[i] << endl;
}

cudaFree(Ad);
cudaFree(Bd);
cudaFree(Cd);

return 0;

}