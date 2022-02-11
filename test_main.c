#include <stdio.h>
#include "include/neural_network.h"

/******************************************************************
 * Main Function
 ******************************************************************/
int main(){

    printf("********************************************************\n");
    printf("** MAIN Function for Testing implementation ************\n");
    printf("********************************************************\n");
    printf("NUM_INPUTS %i\n",NUM_INPUTS);
    printf("NUM_HIDDEN_NODES %i\n",NUM_HIDDEN_NODES);
    printf("NUM_OUTPUTS %i\n",NUM_OUTPUTS);
    printf("NUM_TRAINING_SETS %i\n",NUM_TRAINING_SETS);
    printf("EPOCHS %i\n",EPOCHS);
    printf("PARALLEL_TASKS %i\n",PARALLEL_TASKS);
    printf("SCALE_LIMIT %i\n",SCALE_LIMIT);
    printf("SCALE_DEFAULT %f\n",SCALE_DEFAULT);

    printf("********************************************************\n");
    printf("** Compile and Train NN ********************************\n");
    printf("********************************************************\n");

    nn_init();

    nn_train(0,EPOCHS);

    nn_save("nn_trained.nn");

    printf("********************************************************\n");
    printf("** XOR Test with Save NN *******************************\n");
    printf("********************************************************\n");

    double ua[NUM_INPUTS] = {0,1,0,1,0,1,0,1,0,1,0,1};
    printf("0,1 = \t%lf\n",*(nn_run(ua,NUM_INPUTS)));
    double ua2[NUM_INPUTS] = {1,0,1,0,1,0,1,0,1,0,1,0};
    printf("1,0 = \t%lf\n",*(nn_run(ua2,NUM_INPUTS)));
    double ua3[NUM_INPUTS] = {0,0,0,0,0,0,0,0,0,0,0,0};
    printf("0,0 = \t%lf\n",*(nn_run(ua3,NUM_INPUTS)));
    double ua4[NUM_INPUTS] = {1,1,1,1,1,1,1,1,1,1,1,1};
    printf("1,1 = \t%lf\n",*(nn_run(ua4,NUM_INPUTS)));

    printf("********************************************************\n");
    printf("** XOR Test Random init ********************************\n");
    printf("********************************************************\n");

    nn_init();
   
    double ua5[NUM_INPUTS] = {0,1,0,1,0,1,0,1,0,1,0,1};
    printf("0,1 = \t%lf\n",*(nn_run(ua5,NUM_INPUTS)));
    double ua6[NUM_INPUTS] = {1,0,1,0,1,0,1,0,1,0,1,0};
    printf("1,0 = \t%lf\n",*(nn_run(ua6,NUM_INPUTS)));
    double ua7[NUM_INPUTS] = {0,0,0,0,0,0,0,0,0,0,0,0};
    printf("0,0 = \t%lf\n",*(nn_run(ua7,NUM_INPUTS)));
    double ua8[NUM_INPUTS] = {1,1,1,1,1,1,1,1,1,1,1,1};
    printf("1,1 = \t%lf\n",*(nn_run(ua8,NUM_INPUTS)));

    printf("********************************************************\n");
    printf("** XOR Test read from NN File **************************\n");
    printf("********************************************************\n");

    nn_load("nn_trained.nn");
   
    double ua9[NUM_INPUTS] = {0,1,0,1,0,1,0,1,0,1,0,1};
    printf("0,1 = \t%lf\n",*(nn_run(ua9,NUM_INPUTS)));
    double ua10[NUM_INPUTS] = {1,0,1,0,1,0,1,0,1,0,1,0};
    printf("1,0 = \t%lf\n",*(nn_run(ua10,NUM_INPUTS)));
    double ua11[NUM_INPUTS] = {0,0,0,0,0,0,0,0,0,0,0,0};
    printf("0,0 = \t%lf\n",*(nn_run(ua11,NUM_INPUTS)));
    double ua12[NUM_INPUTS] = {1,1,1,1,1,1,1,1,1,1,1,1};
    printf("1,1 = \t%lf\n",*(nn_run(ua12,NUM_INPUTS)));

    return 0;
}
