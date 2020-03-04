//
// Created by c1over on 3/4/20.
//

#include <bits/stdc++.h>
using namespace std;

const int N = 100;

void matMul_cpu(const int *h_x, const int *h_y, int *h_z, int N) {

    int sum = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            sum = 0;
            for (int z = 0; z < N; z++) {
                sum += h_x[i*N+z] * h_y[j+z*N];
            }
            h_z[i*N+j] = sum;
        }
    }
}

bool check(const int *h_z, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (h_z[i*N+j] != N)
                return false;
        }
    }
    return true;
}

int main() {
    int M = sizeof(int) * N * N;
    int *h_x = (int*)malloc(M);
    int *h_y = (int*)malloc(M);
    int *h_z = (int*)malloc(M);

    for (int i = 0; i < N*N; ++i) {
        h_x[i] = 1;
        h_y[i] = 1;
    }

    auto start = chrono::steady_clock::now();
    matMul_cpu(h_x, h_y, h_z, N);
    auto end = chrono::steady_clock::now();
    auto time = chrono::duration_cast<chrono::microseconds>(end-start).count();
    bool isRight = check(h_z, N);

    printf("Result: %s  Time: %ldms\n", isRight ? "Right!!!" : "Wrong...", time);

    return 0;
}
