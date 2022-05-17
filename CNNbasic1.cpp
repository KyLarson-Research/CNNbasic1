// CNNbasic1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <omp.h>
#include <iostream>
using namespace std;
#define N_INPUT_NEURONS 2
#define N_WEIGHTS_1 4
#define N_WEIGHTS_PER_NEURON (N_WEIGHTS_1/N_INPUT_NEURONS)
#define N_HIDDEN 2
#define N_WEIGHTS_2 2
#define N_OUTPUT_NEURONS 1
#define N_INPUTS  (N_INPUT_NEURONS * N_INPUT_NEURONS)
struct NN {
    float input_neurons[N_INPUT_NEURONS];
    float weights_1[N_WEIGHTS_1];
    float hidden[N_HIDDEN];
    float weights_2[N_WEIGHTS_2];
    float output;
};
//Constructor Destructor
void init(float* nt, int layer) {
    if (layer == 0) {//input layer
        for (int i = 0; i < N_INPUT_NEURONS; i++) {
            nt[i] = 0;
        }
    }
    else if (layer == 1) {//input layer weights
        for (int i = 0; i < N_WEIGHTS_1; i++) {
            nt[i] = 0;
        }
    }
    else if (layer == 2) {
        for (int i = 0; i < N_HIDDEN; i++) {
            nt[i] = 0;
        }
    }
    else if (layer == 3) {
        for (int i = 0; i < N_WEIGHTS_2; i++) {
            nt[i] = 0;
        }
    }

}
NN *new_NN() {
    NN* NNet = (NN*)malloc(sizeof(NN));
    init(NNet->input_neurons, 0);
    init(NNet->weights_1, 1);
    init(NNet->hidden, 2);
    init(NNet->weights_2, 3);
    NNet->output = -1;
    return NNet;
}

void disolve(NN *NNet) {
    free(NNet);
    return;
}
//Multiplication of edge weights by inputs assuming even distribution
//of weights to input neurons
float *L1_multiply(float A[N_INPUTS], float B[N_WEIGHTS_1] ) {
    float C[N_INPUTS];
    //initialize C
    for (int k = 0; k < N_INPUTS; k++) {
        C[k] = 0;
    }
    //Do Parallelized 
    #pragma omp parallel for private(i,j) shared(A,B,C)
    for (int i = 0; i < N_INPUTS; ++i)
        for (int j = 0; j < N_WEIGHTS_PER_NEURON; ++j)
        {
            C[i] += A[i]*B[j];
        }

    return C;
}

void feed_forward(NN *NNet) {
    float *Second_Layer = L1_multiply(NNet->input_neurons, NNet->weights_1);
    for (int i = 0; i < N_HIDDEN; i++) {
        NNet->output += Second_Layer[i];
    }
}

//Primary Function
int main(void)
{
    NN *NNx = new_NN();
    float training_data[N_INPUTS][N_INPUT_NEURONS + N_OUTPUT_NEURONS] = {
        {0,0,0},
        {0,1,1},
        {1,0,1},
        {1,1,0}
    };
    //Produce outputs by Feed Forward
    for (int d = 0; d < N_INPUTS; d++) {
        //feed the input
        for (int e = 0; e < N_INPUTS; e++) {
            NNx->input_neurons[e] = training_data[d][e];
        }
        //Propigate to layer 2 and output
        feed_forward(NNx);
    //TODO: Produce improvements by feedback "Learning Rule"
        cout << " Output: " << NNx->output;
    }
    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
