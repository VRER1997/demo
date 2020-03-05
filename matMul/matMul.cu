//
// Created by c1over on 3/4/20.
//

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <bits/stdc++.h>

#include "cuUtils.h"

using namespace std;

const int N = 100;
const int EPS = 1e-6;
const int BLOCKSIZE = 16;

void matMul_cpu(const int *h_x, const int *h_y, int *h_g, int N) {
    int sum = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            sum = 0;
            for (int z = 0; z < N; z++) {
                sum += h_x[i*N+z] * h_y[j+z*N];
            }
            h_g[i*N+j] = sum;
        }
    }
}

__global__ void matMul_gpu(const int *d_x, const int *d_y, int *d_z, int N) {

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int y = by * BLOCKSIZE + ty, x = bx * BLOCKSIZE + tx;
    if (y < N && x < N) {
        int sum = 0;
        for (int i = 0; i < N; i++) {
            sum += d_x[y*N + i] * d_y[i*N + x];
        }
        d_z[y*N + x] = sum;
    }

}

__global__ void matMul_gpu2(const int* d_x,const int *d_y,int* d_z, int N) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int x = bx * BLOCKSIZE + tx, y = by * BLOCKSIZE + ty;

    int aBegin = N *(by*BLOCKSIZE);
    int aEnd = aBegin + N - 1;
    int aStep = BLOCKSIZE;

    int bBegin = BLOCKSIZE * bx;
    int bStep = BLOCKSIZE * N;

    int sum = 0;
    for (int a = aBegin,b = bBegin; a <= aEnd; a += aStep,b += bStep) {
        __shared__ int X[BLOCKSIZE][BLOCKSIZE];
        __shared__ int Y[BLOCKSIZE][BLOCKSIZE];

        X[ty][tx] = d_x[a + N*ty + tx];
        Y[ty][tx] = d_y[b + N*ty + tx];
        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; ++k) {
            sum += X[ty][k]*Y[k][tx];
        }
        __syncthreads();
    }

    int id = y * N + x;
    d_z[id] = sum;
}

bool check(const int *h_g, const int *h_gz) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (fabs(h_g[i*N+j] - h_gz[i*N+j]) > EPS){
                cout << "error: " << h_g[i*N + j] << " NOT EQUAL " << h_gz[i*N+j] << endl;
                return false;
            }
        }
    }
    return true;
}

void test(int N) {

    int M = sizeof(int) * N * N;
    int *h_x = (int*)malloc(M);
    int *h_y = (int*)malloc(M);

    int *h_z = (int*)malloc(M); // for cpu1
    int *h_g = (int*)malloc(M); // for gpu2
    int *h_gg = (int*)malloc(M); // for gpu2

    for (int i = 0; i < N*N; i++) {
        h_x[i] = 1;
        h_y[i] = 1;
    }

    //CPU1
    auto start = chrono::steady_clock::now();
    if(N < 1025) matMul_cpu(h_x, h_y, h_z, N);
    //cout << "hzzz: " << h_gzz[0] << endl;
    auto end = chrono::steady_clock::now();
    auto cpuTime = chrono::duration_cast<chrono::milliseconds>(end-start).count();

    int *d_x, *d_y, *d_z, *d_zz;
    cudaMalloc(&d_x, M);
    cudaMalloc(&d_y, M);
    cudaMalloc(&d_z, M);
    cudaMalloc(&d_zz, M);

    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    dim3 block(BLOCKSIZE, BLOCKSIZE);
    const int SIZE = (N + BLOCKSIZE - 1) / BLOCKSIZE;
    dim3 grid(SIZE, SIZE);

    //GPU1
    TimerGPU timer(0);
    matMul_gpu<<<grid, block>>> (d_x, d_y, d_z, N);
    double gpuTime = timer.read();
    cudaMemcpy(h_g, d_z, M, cudaMemcpyDeviceToHost);
    //cout << "hz: " << h_g[0] << endl;

    //GPU2
    TimerGPU timer2(0);
    matMul_gpu2<<<grid, block>>> (d_x, d_y, d_zz, N);
    double gpuTime2 = timer2.read();
    cudaMemcpy(h_gg, d_zz, M, cudaMemcpyDeviceToHost);
    cout << "hzz: " << h_gg[0] << endl;

    bool isRight = check(h_g, h_gg);

    printf("Num: %d   Result: %s  CPUTime: %ld ms   GPUTime: %f ms  GPUTime2: %f ms \n", N, isRight ? "Same!!!" : "Wrong...",
            cpuTime, gpuTime, gpuTime2);

    free(h_x);
    free(h_y);
    free(h_z);

    cudaFree(h_x);
    cudaFree(h_y);
    cudaFree(h_z);
    cudaFree(h_g);
    cudaFree(h_gg);
}

int main () {
    test(128);
    test(1024);
    test(10240);
}
