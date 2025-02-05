#include <stdlib.h>
#include <stdio.h>
#define MAX(a,b) (((a)>(b))?(a):(b))

#define MATRIX_SIZE 4096
// define the matrix dimensions A is MxP, B is PxN, and C is MxN

// calculate C = AxB
void matmul(float **A, float **B, float **C, int M_start, int N_start, int P_start, int M, int N, int P) {
 //base case
 //printf("Running with M=%d, N=%d and P=%d\n", M, N, P);
  if((M==16)&&(P==16)&&(N==16)) {
	  float sum;
	  int   i;
	  int   j;
	  int   k;
	int count = 0;
	  for (i=M_start; i<M_start + 16; i++) {
	    // for each row of C
	    for (j=N_start; j<N_start + 16; j++) {
	      // for each column of C
	      sum = 0.0f; // temporary value
	      for (k=P_start; k<P_start + 16; k++) {
		// dot product of row from A and column from B
		sum += A[i][k]*B[k][j];
		count ++;
	      }
	      C[i][j] = sum;
	    }
	  }
	//printf("COUNT : %d\n", count);
  }

 //Splitting along the largest dimension
  else if(M >= MAX(N,P)) {
    //printf("M : Max is %d\n", MAX(N,P));
    matmul(A, B, C, M_start, N_start, P_start, M/2, N, P); 
    matmul(A, B, C, M_start + (M/2), N_start, P_start, M/2, N, P); 
  } 

  else if(N >= MAX(M,P)) {
    //printf("N : Max is %d\n", MAX(M,P));
    matmul(A, B, C, M_start, N_start, P_start, M, N/2, P); 
    matmul(A, B, C, M_start, N_start + (N/2), P_start, M, N/2, P); 
  } 

  else if(P >= MAX(N,M)) {
    //printf("P : Max is %d\n", MAX(N,M));
    matmul(A, B, C, M_start, N_start, P_start, M, N, P/2); 
    matmul(A, B, C, M_start, N_start, P_start + (P/2), M, N, P/2); 
  } 

  else {
     printf("ERROR: Unrecognized condition\n");
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

void create_zero_matrix(float*** C, int m, int n) {
  float **T = 0;
  int i;

  T = (float**)calloc( m, sizeof(float*));
  for ( i=0; i<m; i++ ) {
     T[i] = (float*)calloc(n, sizeof(float));
  }
  *C = T;
}

int main() {
  float** A;
  float** B;
  float** C;
 // float** C_golden;
  int M = MATRIX_SIZE, N=MATRIX_SIZE, P=MATRIX_SIZE;

  create_matrix(&A, M, P);
  create_matrix(&B, P, N);
//  create_matrix(&C_golden, M, N);
  create_zero_matrix(&C, M, N);

  // assume some initialization of A and B
  // think of this as a library where A and B are
  // inputs in row-major format, and C is an output
  // in row-major.

  matmul(A, B, C, 0, 0, 0, M, N, P);
//  matmul_golden(A, B, C_golden);
//  compare_golden(C, C_golden);
  return (0);
}


