#include <stdio.h>
#include <stdlib.h>
#include <string.h>
# include <xmmintrin.h>
#include <x86intrin.h>
#include <sys/time.h>
#include <pthread.h>
#include "myblockmm.h"

struct thread_info
{
    int tid;
    double **a, **b, **c;
    int array_size;
    int number_of_threads;
    int n;
};
void *mythreaded_vector_blockmm(void *t);

#define CHECK_X 4
#define CHECK_Y 4
#define BIT_ORGANIZE_VECTOR 0b11100100

#define BLOCK_SIZE 64

char name[128];
char SID[128];
#define VECTOR_WIDTH 4
void my_threaded_vector_blockmm(double **a, double **b, double **c, int n, int ARRAY_SIZE, int number_of_threads)
{
  int i=0;
  int j=0;

  // double **a_t;
  double **b_t;

  pthread_t *thread;
  struct thread_info *tinfo;
  strcpy(name,"Tyler Woods");
  strcpy(SID,"861299081");
  thread = (pthread_t *)malloc(sizeof(pthread_t)*number_of_threads);
  tinfo = (struct thread_info *)malloc(sizeof(struct thread_info)*number_of_threads);

  __m256d vb_t;

  printf("Array size: %d\n", ARRAY_SIZE);

/*
  b_t = (double **)malloc(ARRAY_SIZE*sizeof(double *));
  for(i = 0; i < ARRAY_SIZE; i++)
  {
    //printf("allocating b[%d]\n", i);
    b_t[i] = (double *)_mm_malloc(ARRAY_SIZE*sizeof(double),256);
    for(j = 0; j < ARRAY_SIZE; j++) {
        // printf("transposing i: %d, j: %d", i, j);
        // printf("copying %f from B\n", b[j][i]);
        b_t[i][j] = b[j][i];
        // printf("storing %f into B_t\n", b_t[i][j]);
        // printf("..done\n");
      }
  }
*/
/*
  a_t = (double **)malloc(ARRAY_SIZE*sizeof(double *));
  for(i = 0; i < ARRAY_SIZE; i++)
  {
    a_t[i] = (double *)_mm_malloc(ARRAY_SIZE*sizeof(double),256);
    for(j = 0; j < ARRAY_SIZE; j++) {
        a_t[i][j] = a[j][i];
    }
  }
*/

  for(i = 0 ; i < number_of_threads ; i++)
  {
    tinfo[i].a = a;
    // tinfo[i].a = a_t;
    tinfo[i].b = b;
    // tinfo[i].b = b_t;
    tinfo[i].c = c;
    tinfo[i].tid = i;
    tinfo[i].number_of_threads = number_of_threads;
    tinfo[i].array_size = ARRAY_SIZE;
    tinfo[i].n = n;
    pthread_create(&thread[i], NULL, mythreaded_vector_blockmm, &tinfo[i]);
  }  
  for(i = 0 ; i < number_of_threads ; i++)
    pthread_join(thread[i], NULL);

  return;
}

#define VECTOR_WIDTH 4
void *mythreaded_vector_blockmm(void *t)
{
  int i,j,k, ii, jj, kk, x;
  __m256d va;
  __m256d vb;
  __m256d vc;
  // __m256d vc_t;
  // __m256d va, vb, vc;
  // __m256d va_0, va_1, va_2, va_3;
  // __m256d vb_0, vb_1, vb_2, vb_3; 
  // __m256d vc_0, vc_1, vc_2, vc_3;
  struct thread_info tinfo = *(struct thread_info *)t;
  int number_of_threads = tinfo.number_of_threads;
  int tid =  tinfo.tid;
  double **a = tinfo.a;
  double **b = tinfo.b;
  double **c = tinfo.c;
  int ARRAY_SIZE = tinfo.array_size;
  int n = tinfo.n;

  for(i = (ARRAY_SIZE/number_of_threads)*(tid); i < (ARRAY_SIZE/number_of_threads)*(tid+1); i+=ARRAY_SIZE/n)
  {
    // printf("i = %d; i < %d; i+=%d\n", (ARRAY_SIZE/number_of_threads)*(tid), (ARRAY_SIZE/number_of_threads)*(tid+1), ARRAY_SIZE/n);
    for(k = 0; k < ARRAY_SIZE; k+=(ARRAY_SIZE/n))
    {
      // printf("j = 0; j < %d; j+=%d\n",  ARRAY_SIZE, (ARRAY_SIZE/n));
      for(j = 0; j < ARRAY_SIZE; j+=(ARRAY_SIZE/n))
      {        
         for(ii = i; ii < i+(ARRAY_SIZE/n); ii++)
         {
            for(jj = j; jj < j+(ARRAY_SIZE/n); jj+=VECTOR_WIDTH)
            {
                vc = _mm256_load_pd(&c[ii][jj]);    
                for(kk = k; kk < k+(ARRAY_SIZE/n); kk++)
		// for(kk = k; kk < k+(ARRAY_SIZE/n); kk+=VECTOR_WIDTH)
                {
			                  va = _mm256_broadcast_sd(&a[ii][kk]);
                        vb = _mm256_load_pd(&b[kk][jj]);
                        vc = _mm256_fmadd_pd(va,vb, vc);
                }
                _mm256_store_pd(&c[ii][jj],vc);
            }
          }
      }
    }
  }

}

