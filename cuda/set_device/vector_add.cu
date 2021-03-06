/*  Vector addition on the GPU: C = A + B  */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 1024
#define BLOCKSIZE 32

// Device function (i.e. kernel)
__global__ void VecAdd(float * A, float * B, float * C, int N)
{

   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if ( i < N ) {
      C[i] = A[i] + B[i];
   }

}

// CPU version of the vector addition function
void vecAddCPU(float * A, float * B, float * C, int N)
{

   int i;
   for (i=0; i<N; i++)
   {
      C[i] = A[i] + B[i];
   }

}

// Function compares two 1d arrays
void compareVecs( float * vec1, float * vec2, int N )
{

   int i;
   int vecsEqual = 1;
   for (i=0; i<N; i++)
   {
      if ( abs (vec1[i] - vec2[i]) > 0.00001 )
      {
         printf("vectors not equal! i: %d  vec1[i]: %f  vec2[i]: %f\n",i,vec1[i],vec2[i]);
         vecsEqual = 0;
      }
   }
   if ( vecsEqual ) printf("GPU vector addition agrees with CPU version!\n");

}

/* Host function for filling vector (1d array) with 
   random numbers between -20.0 and 20.0 */
void fillOutVector( float * vec, int vec_length )
{

   time_t t;
   srand((unsigned) time(&t)); // initialize random number generator
   int i;
   for (i=0; i<vec_length; i++)
   {
      vec[i] = ( (float)rand() / (float)(RAND_MAX) ) * 40.0;
      vec[i] -= 20.0;
   }

}

// Host function for printing a vector (1d array)
void printVector( float * vec, int vec_length )
{
   int i;
   for (i=0; i<vec_length; i++) {
      printf("i: %d vec[i]: %f\n",i,vec[i]);
   }

}

// program execution begins here
int main( int argc, char ** argv )
{

   int cnt = 1;
   int gpu_index = 0;

   if ( argc == 1 ) {
      printf("Usage: ./vec_add --device <GPU_INDEX>\n");
      printf("where GPU_INDEX is the GPU you want to use.\n");
      printf("On the ACCRE GPU cluster valid values are 0, 1, 2, and 3\n");
      exit(-1);
   }

   while ( cnt < argc )
   {

      if ( !strcmp("--device",argv[cnt]) )
      {
         cnt++;
         gpu_index = atoi( argv[cnt] );
      }
      else
      {
         printf("Usage: ./vec_add --device <GPU_INDEX>\n");
         printf("where GPU_INDEX is the GPU you want to use.\n");
         printf("On the ACCRE GPU cluster valid values are 0, 1, 2, and 3\n\n");
         printf("Unrecognized option: %s\n",argv[cnt]);
         exit(-1);
      }
      cnt++;

   }

   // it's important to set the device before initializing events! Otherwise 
   // the events will assume gpu=0
   cudaSetDevice(gpu_index);
   // get number of SMs on this GPU
   int devID;
   cudaDeviceProp props;
   cudaGetDevice(&devID);
   cudaGetDeviceProperties(&props, devID);
   printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major,props.minor);

   // initialize timer events
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   size_t vec_bytes = SIZE * sizeof(float);

   // host arrays
   float * h_A = (float *)malloc( vec_bytes );
   float * h_B = (float *)malloc( vec_bytes );
   float * h_C = (float *)malloc( vec_bytes );

   // fill array with random floats
   fillOutVector( h_A, SIZE );
   fillOutVector( h_B, SIZE );

   // device arrays
   float * d_A, * d_B, * d_C;
   cudaError_t rc; // return code from cuda functions
   rc = cudaMalloc(&d_A, vec_bytes);
   if ( rc ) printf("Error from cudaMalloc: %s\n",cudaGetErrorString(rc));
   cudaMalloc(&d_B, vec_bytes);
   cudaMalloc(&d_C, vec_bytes);

   // copy A and B to the device
   cudaMemcpy(d_A, h_A, vec_bytes, cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, vec_bytes, cudaMemcpyHostToDevice);

   // dim3 is a 3-element struct with elements x, y, z (all ints)
   dim3 threadsPerBlock(BLOCKSIZE);
   dim3 blocksPerGrid( (SIZE + BLOCKSIZE - 1) / BLOCKSIZE );
   // launch vector addition kernel!
   cudaEventRecord(start);
   VecAdd<<< blocksPerGrid, threadsPerBlock >>>(d_A, d_B, d_C, SIZE);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("kernel time (ms) : %7.5f\n",milliseconds);

   // copy results to host
   cudaMemcpy(h_C, d_C, vec_bytes, cudaMemcpyDeviceToHost);
   printVector( h_C, SIZE );

   // verify that we got correct results
   float * gold_C = (float *)malloc( vec_bytes );
   cudaEventRecord(start);
   vecAddCPU( h_A, h_B, gold_C, SIZE );
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("cpu function time (ms) : %7.5f\n",milliseconds);
   compareVecs( gold_C, h_C, SIZE );

   // clean up timer variables
   cudaEventDestroy(start);
   cudaEventDestroy(stop);

   // free memory on device
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

   // free memory on host
   free(h_A);
   free(h_B);
   free(h_C);
   free(gold_C);

   return 0;
}