#include <stdio.h>
#include <stdlib.h>

void allocateIntArray( int ** array , int array_size )
{

   // the input type for array must be a pointer to a pointer,
   // otherwise we would just be modifying a local copy of a pointer,
   // just like any other type! This way we are actually modifying
   // what the pointer in main is pointing to.
   *array = malloc( array_size * sizeof(int) );
   if ( array == NULL ) {
      printf("malloc failed!\n");
      exit(0);
   }

}

void readInMatrix(char * file, int * mat, int mat_len)
{

   FILE *fp_in;
   fp_in = fopen(file,"r");
   if ( fp_in == NULL ) {
      printf("%s not found, exiting...\n",file);
      exit(0);
   }

   int row,col,tot_elements=0;
   for ( row=0; row<mat_len ; row++ ) {
      for ( col=0; col<mat_len ; col++ ) {
         fscanf(fp_in,"%d ",&mat[tot_elements]);
         //printf("row: %d col: %d matrix: %d\n",row,col,mat[tot_elements]);
         tot_elements++;
      }
   }

   fclose(fp_in);

} 

void matrixMultiply( int * matrix_A, int * matrix_B, int mat_len )
{

   int row,col,tot_elements=0;
   for ( row=0; row<mat_len ; row++ ) {
      for ( col=0; col<mat_len ; col++ ) {
         matrix_B[tot_elements++] = 0; // initialize array first
      }
   }

   tot_elements = 0;
   for ( row=0; row<mat_len ; row++ ) {
      for ( col=0; col<mat_len ; col++ ) {
         int position;
         for ( position=0; position<mat_len ; position++ ) {
            int row_index = row * mat_len + position;
            int col_index = col + position * mat_len;
            matrix_B[tot_elements] += matrix_A[row_index] * matrix_A[col_index]; 
         }
         tot_elements++;
      }
   }

}

void printResult( int * mat, int mat_len ) 
{

   int row,col,tot_elements=0;
   for ( row=0; row<mat_len ; row++ ) {
      for ( col=0; col<mat_len ; col++ ) {
         printf("%d ",mat[tot_elements]);
         tot_elements++;
      }
      printf("\n");
   }

}   

int main(int argc, char **argv)
{
   
   // check command line arguments
   if ( argc != 3 ) {
      printf("This program computes the product of n x n matrix with itself\n");
      printf("Usage: ./matrix_multiply filename n\n");
      exit(0);
   }

   int n = atoi(argv[2]);
   printf("n: %d\n",n);

   // allocate space for matrix in 1d array
   int * matrix_A;  
   allocateIntArray( &matrix_A, n*n );

   // read in matrix
   char * filename = argv[1];
   readInMatrix(filename, matrix_A, n);

   // allocate space for product matrix in 1d array
   int * matrix_B;
   allocateIntArray( &matrix_B, n*n );

   // perform matrix multiplication
   matrixMultiply( matrix_A, matrix_B, n );
   
   // write results to stdout
   printResult( matrix_B, n );   

   return 0;
}