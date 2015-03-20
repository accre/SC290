#include <stdio.h>
#include <stdlib.h>

#define SIZE 1024

// Device function (i.e. kernel)
__global__ void myKernel(float * A)
{

   int i = threadIdx.x;
   A[i] *= (float)i;

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

// Wrapper for cudaMalloc that shows error message if needed
void myCudaMalloc( float ** array, size_t size )
{
   // The GTX 480s have ~1536 MB = 1610285056 bytes of RAM
   // As you approach this ballpark you should get an error
   cudaError_t rc = cudaMalloc(array, size);
   if ( rc )
   {
      printf("cudaMalloc error: %s\n",cudaGetErrorString(rc));
      exit(-1);
   }
}

// program execution begins here
int main( int argc, char ** argv )
{
   size_t vec_bytes = SIZE * sizeof(float);

   //printf("size of float: %d bytes\n",(int)sizeof(float));

   // host arrays
   float * h_A = (float *)malloc( vec_bytes );

   // fill array with random floats
   fillOutVector( h_A, SIZE );

   // device arrays
   float * d_A;

   /* Calling cuda wrapper for automatically getting error code info
      and halting execution */ 
   myCudaMalloc(&d_A, vec_bytes);

   // copy A and B to the device
   cudaMemcpy(d_A, h_A, vec_bytes, cudaMemcpyHostToDevice);

   // dim3 is a 3-element struct with elements x, y, z (all ints)
   dim3 threadsPerBlock(SIZE);
   dim3 blocksPerGrid(1);
   myKernel<<< blocksPerGrid, threadsPerBlock >>>(d_A);

   // free memory on device
   cudaFree(d_A);

   // free memory on host
   free(h_A);

   return 0;
}