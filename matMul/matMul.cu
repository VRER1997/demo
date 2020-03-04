//
// Created by c1over on 3/4/20.
//

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <bits/stdc++.h>

#include "cuUtils.h"

using namespace std;

const int N = 10000;
const int EPS = 1e-6;
const int BLOCKSIZE = 16;

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
        __shared__ int X[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ int Y[BLOCK_SIZE][BLOCK_SIZE];

        X[ty][tx] = d_x[a + N*ty + tx];
        Y[ty][tx] = d_y[b + N*ty + tx];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += X[ty][k]*Y[k][tx];
        }
        __syncthreads();
    }

    int id = y * N + x;
    d_z[id] = sum;
}

bool check(const int *h_z, const int *h_zz) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (fabs(h_z[i*N+j] - h_zz[i*N+j]) > EPS){
                cout << "error: " << h_z[i*N + j] << " NOT EQUAL " << h_zz[i*N+j] << endl;
                return false;
            }
        }
    }
    return true;
}

int main() {
    int M = sizeof(int) * N * N;
    int *h_x = (int*)malloc(M);
    int *h_y = (int*)malloc(M);
    int *h_z = (int*)malloc(M);
    int *h_zz = (int*)malloc(M);
    int *h_zzz = (int*)malloc(M);

    for (int i = 0; i < N*N; ++i) {
        h_x[i] = 1;
        h_y[i] = 1;
    }

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
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    cout << "hz: " << h_z[0] << endl;

    //GPU2
    TimerGPU timer2(0);
    matMul_gpu2<<<grid, block>>> (d_x, d_y, d_zz, N);
    double gpuTime2 = timer2.read();
    cudaMemcpy(h_zz, d_zz, M, cudaMemcpyDeviceToHost);
    cout << "hzz: " << h_zz[0] << endl;
    //CPU
    auto start = chrono::steady_clock::now();
    if(N < 1000) matMul_cpu(h_x, h_y, h_zzz, N);
    cout << "hzzz: " << h_zzz[0] << endl;
    auto end = chrono::steady_clock::now();
    auto cpuTime = chrono::duration_cast<chrono::microseconds>(end-start).count();

    bool isRight = check(h_z, h_zz);

    printf("Result: %s  CPUTime: %ld ms   GPUTime: %f ms  GPUTime2: %f ms \n", isRight ? "Same!!!" : "Wrong...",
            cpuTime, gpuTime, gpuTime2);

    return 0;
}
