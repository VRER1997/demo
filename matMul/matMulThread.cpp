#include <bits/stdc++.h>
using namespace std;

#define N 128

int *matA, *matB, *matC, *matCC;
int step_i = 0;
mutex data_mutex;

void* task2(void* arg) {
    unique_lock<std::mutex> lock(data_mutex);
    int core = step_i++;
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k)
            matC[i*N + core] += matA[i*N + k] * matB[k*N + core];
    return 0;
}

void matMul_cpu2() {
    pthread_t threads[N];
    for (int i = 0; i < N; i++) {
        int* p;
        pthread_create(&threads[i], NULL, task2, (void*)(p));
    }
    for (int i = 0; i < N; i++) {
        pthread_join(threads[i], NULL);
    }
}

int step_ij = 0;
mutex data_mutex3;
void* task3(void* arg) {
    unique_lock<std::mutex> lock(data_mutex3);
    int core = step_ij++;
    int i = core / N;
    int j = core % N;
    for (int k = 0; k < N; ++k) {
        matCC[i*N + j] += matA[i*N + k] * matB[k*N + j];
    }
    return 0;
}

void matMul_cpu3() {
    pthread_t threads[N * N];
    for (int i = 0; i < N * N; i++) {
        int* p;
        pthread_create(&threads[i], NULL, task3, (void*)(p));
    }
    for (int i = 0; i < N * N; i++) {
        pthread_join(threads[i], NULL);
    }
}

bool check(int *d_z) {
    for (int i = 0; i < N*N; i++) {
        if (d_z[i] != N) {
            cout << "error " << d_z[i] << endl;
            return false;
        }
    }
    return true;
}

int main() {
    int M = sizeof(int) * N * N;
    matA = (int*)malloc(M);
    matB = (int*)malloc(M);
    matC = (int*)malloc(M);
    matCC = (int*)malloc(M);

    for (int i = 0; i < N * N; i++) {
        matA[i] = 1;
        matB[i] = 1;
    }
    auto start = chrono::steady_clock::now();
    matMul_cpu2();
    auto end = chrono::steady_clock::now();
    auto cputime2 = chrono::duration_cast<chrono::milliseconds> (end-start).count();
    bool ret1 = check(matC);

    start = chrono::steady_clock::now();
    if(N < 129) matMul_cpu3();
    end = chrono::steady_clock::now();
    auto cputime3 = chrono::duration_cast<chrono::milliseconds> (end-start).count();
    bool ret2 = true;
    if(N < 129) ret2 = check(matCC);

    printf("Num: %d   Reuslt: %s   CPUTime2: %ld ms   CPUTime3: %ld ms \n",
            N, ret1 && ret2 ? "Right!!!" : "Wrong...", cputime2, cputime3);

    free(matA);
    free(matB);
    free(matC);
    free(matCC);

    return 0;
}
