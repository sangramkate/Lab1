#include <stdlib.h>

// define the matrix dimensions A is MxP, B is PxN, and C is MxN
#define M 8192
#define N 8192
#define P 8192


// calculate C = AxB
void matmul(float **A, float **B, float **C, int Block) {
  int   i;
  int   j;
  int   k;
  int   ib, jb, kb;

 for (ib = 0; ib < M ; ib = ib + Block) {
	for (jb = 0; jb < N; jb = jb + Block) {
		for (kb = 0; kb < P ; kb = kb + Block) {
    			 for (i=0; i<Block; i++) {
    			   // for each row of C
    			   for (j=0; j<Block; j++) {
    			     // for each column of C
    			     for (k=0; k<Block; k++) {
    			       // dot product of row from A and column from B
    			       
    			       C[ib+i][jb+j] += A[ib+i][kb+k]*B[kb+k][jb+j];
	
    			     }
    			   }
    			 }
		}
  	}
  }
}

// function to allocate a matrix on the heap
// creates an mXn matrix and returns the pointer.
//
// the matrices are in row-major order.
void create_matrix(float*** A, int m, int n) {
  float **T = 0;
  int i;

  T = (float**)malloc( m*sizeof(float*));
  for ( i=0; i<m; i++ ) {
     T[i] = (float*)malloc(n*sizeof(float));
  }
  *A = T;
}

int main(int argc, char **argv) {

  if (argc < 2) {
	return(1); 
  }
  float** A;
  float** B;
  float** C;

  create_matrix(&A, M, P);
  create_matrix(&B, P, N);
  create_matrix(&C, M, N);

  // assume some initialization of A and B
  // think of this as a library where A and B are
  // inputs in row-major format, and C is an output
  // in row-major.
  int Block = atoi(argv[1]);
  matmul(A, B, C, Block);

  return (0);
}
