#include <stdlib.h>
#include <stdio.h>

// define the matrix dimensions A is MxP, B is PxN, and C is MxN
#define TILE_SIZE 4096
#define M TILE_SIZE
#define N TILE_SIZE
#define P TILE_SIZE

// calculate C = AxB
void matmul(float **A, float **B, float **C, int Block) {
  int   i;
  int   j;
  int   k;
  int   ib2, jb2, kb2;
  int   ib, jb, kb;
//going for ikj as it has best locality - both spatial and temporal

int block2_ij = 128;
int block2_k = 256;
int block1_ij = 8;
int block1_k = 16;

//printf("%d %d %d", M, N, P);
for (ib2 = 0; ib2 < M ; ib2 = ib2 + block2_ij) {
  for (kb2 = 0; kb2 < P; kb2 = kb2 + block2_k) {
    for (jb2 = 0; jb2 < N; jb2 = jb2 + block2_ij) {
 	for (ib = 0; ib < block2_ij ; ib = ib + block1_ij) {
	     for (kb = 0; kb < block2_k ; kb = kb + block1_k) {
		for (jb = 0; jb < block2_ij; jb = jb + block1_ij) {
    			 for (i=0; i<block1_ij; i++) {
    			    // for each row of C
    			    for (k=0; k<block1_k; k++) {
    			       // dot product of row from A and column from B
    			       for (j=0; j<block1_ij; j++) {
    			         // for each column of C
    			       
    			       C[ib2+ib+i][jb2+jb+j] += A[ib2+ib+i][kb2+kb+k]*B[kb2+kb+k][jb2+jb+j];
	
    			     }
    			   }
    			 }
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

  if (argc < 1) {
	printf("ERROR");
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
  int Block = 32; //not used //atoi(argv[1]);
  matmul(A, B, C, Block);

  return (0);
}
