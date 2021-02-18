#include <stdlib.h>

// define the matrix dimensions A is MxP, B is PxN, and C is MxN
#define M 4096
#define N 4096
#define P 4096

// calculate C = AxB
void matmul(float **A, float **B, float **C, int Block) {
  int   i;
  int   j;
  int   k;
  int   ib, jb, kb;
//going for ikj as it has best locality - both spatial and temporal

 for (ib = 0; ib < M ; ib = ib + 32) {
	for (kb = 0; kb < P ; kb = kb + 64) {
		for (jb = 0; jb < N; jb = jb + 32) {
    			 for (i=0; i<32; i++) {
    			    // for each row of C
    			    for (k=0; k<64; k++) {
    			       // dot product of row from A and column from B
    			       for (j=0; j<32; j++) {
    			         // for each column of C
    			       
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
