// CPP Program to multiply two matrix using pthreads
#include <bits/stdc++.h>
using namespace std;

// maximum size of matrix
#define N 100

int *matA, *matB, *matC;;
int step_i = 0;


void* multi(void* arg)
{
    int core = step_i++;
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k)
            matC[i*N + core] += matA[i*N + k] * matB[k*N + core];
}

void test() {

}

void check(int *d_z) {
    for (int i = 0; i < N*N; i++) {
        if (d_z[i] != N) {
            cout << "error " << d_z[i] << endl;
            return;
        }
    }
    cout << "right" << endl;
}

int main()
{
    int M = sizeof(int) * N * N;
    matA = (int*)malloc(M);
    matB = (int*)malloc(M);
    matC = (int*)malloc(M);

    for (int i = 0; i < N * N; i++) {
        matA[i] = 1;
        matB[i] = 1;
    }

    // declaring four threads
    pthread_t threads[N];

    // Creating four threads, each evaluating its own part
    for (int i = 0; i < N; i++) {
        int* p;
        pthread_create(&threads[i], NULL, multi, (void*)(p));
    }

    for (int i = 0; i < N; i++)
        pthread_join(threads[i], NULL);

//    for (int i = 0; i < N; i++) {
//        for (int j = 0; j < N; j++)
//            cout << matC[i*N + j] << " ";
//        cout << endl;
//    }
    check(matC);

    return 0;
}
