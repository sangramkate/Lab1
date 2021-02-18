#include <stdlib.h>
#include <immintrin.h>

// define the matrix dimensions A is MxP, B is PxN, and C is MxN
 #define M 512
 #define N 512
 #define P 512


 // calculate C = AxB
 void matmul(float **A, float **B, float **C) {
 int   i;
 int   j;
 int   k;
 int   ib3, jb3, kb3;
 int   ib2, jb2, kb2;
 int   ib1, jb1, kb1;

 int Block1 = 32;
 int Block2 = 256;
 int Block3 = 1024;

  for(ib3 =0; ib3 < M; ib3 = ib3 + Block3) {
    for(kb3 = 0; kb3 < N ; kb3 = kb3 + 2048) {
      for (jb3 = 0; jb3 < P; jb3 = jb3 + Block3) {
        for (ib2 = 0; ib2 < Block3 ; ib2 = ib2 + Block2) {
          for (kb2 = 0; kb2 < 2048; kb2 = kb2 + Block2) {
            for (jb2 = 0; jb2 < Block3 ; jb2 = jb2 + Block2) {
              for (ib1=0; ib1<Block2; ib1 = ib1 + Block1) {
              // for each row of C
                for (kb1=0; kb1<Block2; kb1 = kb1 + 64) {
                // for each column of C
                  for (jb1=0; jb1<Block2; jb1 = jb1+Block1) {
                    for (i = 0; i < Block1; i+=1) {
                      for(k = 0; k < 64; k+=1) {
                        __m512 m1 = _mm512_set1_ps(A[ib3+ib2+ib1+i][kb3+kb2+kb1+k]);

                        for (j = 0; j < Block1; j = j+16) {
			  __m512 m2 = _mm512_loadu_ps(&B[kb3+kb2+kb1+k][jb3+jb2+jb1+j]);
			  __m512 m3 = _mm512_loadu_ps(&C[ib3+ib2+ib1+i][jb3+jb2+jb1+j]);
			  __m512 m4 = _mm512_fmadd_ps(m1,m2,m3);
		     	  _mm512_storeu_ps(&C[ib3+ib2+ib1+i][jb3+jb2+jb1+j], m4);
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
    



 int main() {
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
   matmul(A, B, C);

   return (0);
 }
