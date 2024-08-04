#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8
#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS ((N * N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)

__constant__ int d_xmoves[8];
__constant__ int d_ymoves[8];

__device__ bool can_move(int x, int y, const int* board) {
    return (0 <= x && x < N) && (0 <= y && y < N) && (board[x * N + y] == -1);
}

__global__ void knight_tour_kernel(int* board, bool* found_solution) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * N) return;

    int start_x = tid / N;
    int start_y = tid % N;

    int local_board[N * N];
    for (int i = 0; i < N * N; i++) {
        local_board[i] = -1;
    }
    local_board[start_x * N + start_y] = 0;

    int x = start_x, y = start_y;
    for (int move = 1; move < N * N; move++) {
        int next_x = -1, next_y = -1;
        int min_degree = 9;

        for (int i = 0; i < 8; i++) {
            int nx = x + d_xmoves[i];
            int ny = y + d_ymoves[i];
            if (can_move(nx, ny, local_board)) {
                int degree = 0;
                for (int j = 0; j < 8; j++) {
                    int nnx = nx + d_xmoves[j];
                    int nny = ny + d_ymoves[j];
                    if (can_move(nnx, nny, local_board)) {
                        degree++;
                    }
                }
                if (degree < min_degree) {
                    min_degree = degree;
                    next_x = nx;
                    next_y = ny;
                }
            }
        }

        if (next_x == -1) {
            return;
        }

        x = next_x;
        y = next_y;
        local_board[x * N + y] = move;
    }

    *found_solution = true;
    for (int i = 0; i < N * N; i++) {
        board[i] = local_board[i];
    }
}

void print_board(const int* board) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if ((i + j) % 2 == 0) {
                printf("\033[47m"); 
                printf("\033[30m"); 
            } else {
                printf("\033[40m"); 
                printf("\033[37m"); 
            }
            printf("%3d ", board[i * N + j]);
        }
        printf("\n");
        printf("\033[0m"); 
    }
}

int main() {
    int h_board[N * N];
    bool h_found_solution = false;
    int* d_board;
    bool* d_found_solution;

    int h_xmoves[8] = {2, 1, -1, -2, -2, -1, 1, 2};
    int h_ymoves[8] = {1, 2, 2, 1, -1, -2, -2, -1};

    cudaMemcpyToSymbol(d_xmoves, h_xmoves, 8 * sizeof(int));
    cudaMemcpyToSymbol(d_ymoves, h_ymoves, 8 * sizeof(int));

    cudaMalloc(&d_board, N * N * sizeof(int));
    cudaMalloc(&d_found_solution, sizeof(bool));

    cudaMemset(d_board, -1, N * N * sizeof(int));
    cudaMemcpy(d_found_solution, &h_found_solution, sizeof(bool), cudaMemcpyHostToDevice);

    knight_tour_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_board, d_found_solution);

    cudaMemcpy(&h_found_solution, d_found_solution, sizeof(bool), cudaMemcpyDeviceToHost);
    if (h_found_solution) {
        cudaMemcpy(h_board, d_board, N * N * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Solution found:\n");
        print_board(h_board);
    } else {
        printf("No solution found.\n");
    }

    cudaFree(d_board);
    cudaFree(d_found_solution);

    return 0;
}