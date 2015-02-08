#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
   
   // check command line arguments
   if ( argc != 3 ) {
      printf("This program computes the product of n x n matrix with itself\n");
      printf("Usage: ./matrix_multiply filename n\n");
      exit(0);
   }

   // TODO: parse input arguments

   int * matrix_in, matrix_out;  // declare input and output matrices
   // TODO: dynamically allocate space for matrix_in and matrix_out

   // TODO: call function to read data from file and copy into matrix_A

   // TODO: call function to perform matrix multiplication ( matrix_B = matrix_A * matrix_A )
   
   // TODO: call function to write results (matrix_out) to stdout

   return 0;
}