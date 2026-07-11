#include <stdlib.h>
#include <string.h>
#include <time.h> // random
#include <stdio.h>

// sum into array C using pointers
void sumArrays(float *A, float *B, float *C, const int N) {
    for(int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// fill given array with random data
void initializeData(float *arr, int N) {

    time_t t;
    srand((unsigned int) time(&t));
    for(int i = 0; i < N; i++) {
        arr[i] = (float) ( rand() & 0xFF) / 10.0f;
        // Generates a random integer using rand(), applies a bitwise AND with 0xFF to limit the value to the range [0, 255],
        // then casts the result to float and divides by 10.0f to produce a floating-point number in the range [0.0, 25.5].
        // The resulting value is assigned to arr[i].
    }
    
    // for(int i = 0; i < N; i++) {
    //     arr[i] = 1.1; // just set to 1.1 so very obvious that summing occurs
    // }
}

void printArr(float *arr, int N) {
    for(int i = 0; i <  N; i++) {
        printf("%f  ", arr[i]);
    }
    printf("\n\n");
}



int main(int argc, char* argv[])  {
    int num_elements = 10;
    // set size for alloc operation
    size_t num_bytes = num_elements * sizeof(float);

    // init pointers
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(num_bytes);
    h_B = (float *)malloc(num_bytes);
    h_C = (float *)malloc(num_bytes);

    initializeData(h_A, num_elements);
    initializeData(h_B, num_elements);

    printArr(h_A, num_elements);
    printArr(h_B, num_elements);

    sumArrays(h_A, h_B, h_C, num_elements);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}