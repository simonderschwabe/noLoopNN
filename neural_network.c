/* Copyright 2022 Simon Sommer
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 */
#include "stdio.h"
#include <stdlib.h>
#include "math.h"
#include "stddef.h"
#include "time.h"
#include <dlfcn.h>
#include "include/neural_network.h"

signed int scale_min = 0;
signed int scale_max = 0;

/******************************************************************
 * Variable declation for neural network
 ******************************************************************/
double *hW = NULL;
double *oW = NULL;

double *hL = NULL;
double *oL = NULL;

double *hLB = NULL;
double *oLB = NULL;

double *ti = NULL;
double *to = NULL;

double hiddenWeights[NUM_INPUTS][NUM_HIDDEN_NODES];
double outputWeights[NUM_HIDDEN_NODES][NUM_OUTPUTS];

double hiddenLayer[NUM_HIDDEN_NODES];
double outputLayer[NUM_OUTPUTS];

double hiddenLayerBias[NUM_HIDDEN_NODES];
double outputLayerBias[NUM_OUTPUTS];

double t_input[NUM_TRAINING_SETS][NUM_INPUTS] = { 
		{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
		{1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f},
		{0.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f},
		{1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f} 
		};
double t_output[NUM_TRAINING_SETS][NUM_OUTPUTS] = { {0.0f},{1.0f},{1.0f},{0.0f} };

/******************************************************************
 * Function for Activation and the Derivative of it 
 ******************************************************************/
double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dSigmoid(double x) { return x * (1 - x); }

/******************************************************************
 * Function for init between 0 and 1 
 ******************************************************************/
double randx = 0.00001978f;
double pseudo_number() { randx += 0.00000439f; if(randx>1){ randx=0; } return randx; }

/******************************************************************
 * Scale any Integer to range between 0 and 1
 ******************************************************************/
double scale_to_zero_one(int input){
	if(input > SCALE_LIMIT){ return SCALE_DEFAULT; }
	else if(input < (-1*SCALE_LIMIT)){ return SCALE_DEFAULT; }
	if(input>scale_max && input > 0){ scale_max = input; }
	if(input<scale_min && input < 0){ scale_min = input; }
	return (((double)input-(double)scale_min)/((double)scale_max-(double)scale_min));
}

/******************************************************************
 * getTimeStamp for Measurement
 ******************************************************************/
long getTimeStamp(){

        struct timespec _t;
        clock_gettime(CLOCK_REALTIME, &_t);
        return  _t.tv_sec*1000 + lround(_t.tv_nsec/1.0e6);
}

/******************************************************************
 * Header Lines in generated Code outside function
 ******************************************************************/
int nn_generate_training_code_header(FILE *src_file){

	fprintf(src_file,"#include <stddef.h>\n");
	fprintf(src_file,"#include <stdio.h>\n");
	fprintf(src_file,"#include <math.h>\n");
	fprintf(src_file,"#include \"include/neural_network.h\"\n");

	fprintf(src_file,"extern double hiddenWeights[NUM_INPUTS][NUM_HIDDEN_NODES];\n");
	fprintf(src_file,"extern double outputWeights[NUM_HIDDEN_NODES][NUM_OUTPUTS];\n");

	fprintf(src_file,"extern double hiddenLayer[NUM_HIDDEN_NODES];\n");
	fprintf(src_file,"extern double outputLayer[NUM_OUTPUTS];\n");

	fprintf(src_file,"extern double hiddenLayerBias[NUM_HIDDEN_NODES];\n");
	fprintf(src_file,"extern double outputLayerBias[NUM_OUTPUTS];\n");

	fprintf(src_file,"extern double t_input[NUM_TRAINING_SETS][NUM_INPUTS];\n");
	fprintf(src_file,"extern double t_output[NUM_TRAINING_SETS][NUM_OUTPUTS];\n");

	fprintf(src_file,"extern double sigmoid(double);\n");
	fprintf(src_file,"extern double dSigmoid(double);\n");

	return 14;
}

/******************************************************************
 * Loops to flat formula Code gen
 ******************************************************************/
int nn_generate_training_code(char *filename){

	  FILE *src_file = fopen(filename,"w");
	  int code_lines_generated = 0;

	  code_lines_generated += nn_generate_training_code_header(src_file);

	  fprintf(src_file,"void nn_train_generated()\n");
	  code_lines_generated++;
	  fprintf(src_file,"{\n");
	  code_lines_generated++;
	  fprintf(src_file,"double deltaOutput[NUM_OUTPUTS];\n");
          fprintf(src_file,"double deltaHidden[NUM_HIDDEN_NODES];\n");
	  code_lines_generated++;
	  code_lines_generated++;
	  fprintf(src_file,"for (int n=0; n < %i; n++) {\n",EPOCHS);
	  code_lines_generated++;
          fprintf(src_file,"for (int x=0; x<NUM_TRAINING_SETS; x++) {\n");
	  code_lines_generated++;

	  for (int j=0; j<NUM_HIDDEN_NODES; j++) {
		fprintf(src_file,"hiddenLayer[%i] = sigmoid(hiddenLayerBias[%i]",j,j);
                for (int k=0; k<NUM_INPUTS; k++) {
		    	fprintf(src_file,"+(t_input[x][%i]*hiddenWeights[%i][%i])\n",k,k,j);
	  		code_lines_generated++;
                }
		fprintf(src_file,");\n");
	  	code_lines_generated++;
          }

	  for (int j=0; j<NUM_OUTPUTS; j++) {
		fprintf(src_file,"outputLayer[%i] = sigmoid(outputLayerBias[%i]",j,j);
           	for (int k=0; k<NUM_HIDDEN_NODES; k++) {
			fprintf(src_file,"+(hiddenLayer[%i]*outputWeights[%i][%i])\n",k,k,j);
			code_lines_generated++;
           	}
		fprintf(src_file,");\n");
		code_lines_generated++;
	  }

            for (int j=0; j<NUM_OUTPUTS; j++) {
		fprintf(src_file,"deltaOutput[%i] = (t_output[x][%i]-outputLayer[%i])*dSigmoid(outputLayer[%i]);\n",j,j,j,j); 
		code_lines_generated++;
            }

	    
            for (int j=0; j<NUM_HIDDEN_NODES; j++) {
		fprintf(src_file,"deltaHidden[%i] = (",j);
                for(int k=0; k<NUM_OUTPUTS; k++) {
		    fprintf(src_file,"(deltaOutput[%i]*outputWeights[%i][%i])\n",k,j,k);
		    code_lines_generated++;
		    if(k+1 < NUM_OUTPUTS){
			    fprintf(src_file,"+");
		    }
                }
		fprintf(src_file,")*dSigmoid(hiddenLayer[%i]);\n",j);
		code_lines_generated++;
            }
            
            for (int j=0; j<NUM_OUTPUTS; j++) {
		fprintf(src_file,"outputLayerBias[%i] += deltaOutput[%i]*0.1f;\n",j,j);
		code_lines_generated++;
                for (int k=0; k<NUM_HIDDEN_NODES; k++) {
		    fprintf(src_file,"outputWeights[%i][%i] += hiddenLayer[%i]*deltaOutput[%i]*0.1f;\n",k,j,k,j);
		    code_lines_generated++;
                }
            }
            
            for (int j=0; j<NUM_HIDDEN_NODES; j++) {
		fprintf(src_file,"hiddenLayerBias[%i] += deltaHidden[%i]*0.1f;\n",j,j);
		code_lines_generated++;
                for(int k=0; k<NUM_INPUTS; k++) {
		    fprintf(src_file,"hiddenWeights[%i][%i] += t_input[x][%i]*deltaHidden[%i]*0.1f;\n",k,j,k,j);
		    code_lines_generated++;
                }
            }

	  fprintf(src_file,"}}\n");
	  fprintf(src_file,"}\n");
	  code_lines_generated++;

	  fclose(src_file);

	  return code_lines_generated;
}

/******************************************************************
 * sync nn training data to shared lib structure from local 
 ******************************************************************/
int nn_sync_to_shared_lib(){

	if(ti != NULL && to != NULL && 
	   hW != NULL && oW != NULL && 
	   hL != NULL && hLB != NULL && 
	   oL != NULL && oLB != NULL){
	int ptr_offseti = 0;
	int ptr_offseto = 0;
	int ptr_offset = 0;
	for(int x=0; x < NUM_TRAINING_SETS; x++){
		for(int n=0; n < NUM_INPUTS; n++){
			*(ti+ptr_offseti) = t_input[x][n];
			ptr_offseti += 1;
		}
		for(int n=0; n < NUM_OUTPUTS; n++){
			*(to+ptr_offseto) = t_output[x][n];
			ptr_offseto += 1;
		}
	}
	for(int x=0; x < NUM_INPUTS; x++){
		for(int n=0; n < NUM_HIDDEN_NODES; n++){
			*(hW+ptr_offset) = hiddenWeights[x][n];
			ptr_offset += 1;
		}
	}
	ptr_offset = 0;
	for(int x=0; x < NUM_HIDDEN_NODES; x++){
		for(int n=0; n < NUM_OUTPUTS; n++){
			*(oW+ptr_offset) = outputWeights[x][n];
			ptr_offset += 1;
		}
		*(hL+x) = hiddenLayer[x];
		*(hLB+x) = hiddenLayerBias[x];
	}
	for(int x=0; x < NUM_OUTPUTS; x++){
		*(oL+x) = outputLayer[x];
		*(oLB+x) = outputLayerBias[x];
	}
	}
	return 0;
}

/******************************************************************
 * sync trained neurons from shared lib structure to local 
 ******************************************************************/
int nn_sync_from_shared_lib(){

	int ptr_offset = 0;
	for(int x=0; x < NUM_INPUTS; x++){
		for(int n=0; n < NUM_HIDDEN_NODES; n++){
			hiddenWeights[x][n]=*(hW+ptr_offset);
			ptr_offset += 1;
		}
	}
	ptr_offset = 0;
	for(int x=0; x < NUM_HIDDEN_NODES; x++){
		for(int n=0; n < NUM_OUTPUTS; n++){
			outputWeights[x][n] = *(oW+ptr_offset);
			ptr_offset += 1;
		}
		hiddenLayer[x] = *(hL+x);
		hiddenLayerBias[x] = *(hLB+x);
	}
	for(int x=0; x < NUM_OUTPUTS; x++){
		outputLayer[x] = *(oL+x);
		outputLayerBias[x] = *(oLB+x);
	}

	return 0;
}

/******************************************************************
 * compile gerated code in library
 ******************************************************************/
int nn_gen_code_compile(){

	int retcode = 0;
	nn_generate_training_code(NN_GEN_FILENAME);

	retcode = system("gcc -shared -Ofast -o nn_generated.so -fPIC nn_generated_train.c neural_network.c -lm");

	return retcode;
}

/******************************************************************
 * compile gerated code in library
 ******************************************************************/
int nn_gen_code_train(){

	void *handle;
    	void (*nn_train_generated)(void);
    	char *error;

    	handle = dlopen("./nn_generated.so", RTLD_LAZY);
    	if (!handle) {
        	fprintf(stderr, "%s\n", dlerror());
        	exit(EXIT_FAILURE);
    	}

    	dlerror();

    	*(void **) (&nn_train_generated) = dlsym(handle, "nn_train_generated");
	hW = (double *) dlsym(handle, "hiddenWeights");
	oW = (double *) dlsym(handle, "outputWeights");
	hL = (double *) dlsym(handle, "hiddenLayer");
	oL = (double *) dlsym(handle, "outputLayer");
	hLB = (double *) dlsym(handle, "hiddenLayerBias");
	oLB = (double *) dlsym(handle, "outputLayerBias");
	ti = (double *) dlsym(handle, "t_input");
	to = (double *) dlsym(handle, "t_output");

    	if ((error = dlerror()) != NULL)  {
        	fprintf(stderr, "%s\n", error);
        	exit(EXIT_FAILURE);
    	}

	nn_sync_to_shared_lib();

    	(*nn_train_generated)();

	nn_sync_from_shared_lib();

    	dlclose(handle);

	return 0;
}

/******************************************************************
 * Save NN to file
 ******************************************************************/
void nn_save(const char* file_name){

    FILE* file = fopen (file_name, "w");

    for (int j=0; j<NUM_HIDDEN_NODES; j++) {
        for(int k=0; k<NUM_INPUTS; k++) {
	    fprintf(file, "%lf\n", hiddenWeights[k][j]);
        }
    }
    for (int j=0; j<NUM_HIDDEN_NODES; j++) {
	    fprintf(file, "%lf\n", hiddenLayerBias[j]);
    }
    for (int j=0; j<NUM_OUTPUTS; j++) {
        for (int k=0; k<NUM_HIDDEN_NODES; k++) {
	    fprintf(file, "%lf\n", outputWeights[k][j]);
        }
    }
    for (int j=0; j<NUM_OUTPUTS; j++) {
	    fprintf(file, "%lf\n", outputLayerBias[j]);
    }

    fclose(file);
}

/******************************************************************
 * Load NN from file
 ******************************************************************/
void nn_load(const char* file_name){

    FILE* file = fopen (file_name, "r");
    int ret = 0;

    for (int j=0; j<NUM_HIDDEN_NODES; j++) {
        for(int k=0; k<NUM_INPUTS; k++) {
	    ret = fscanf(file,"%lf\n",&hiddenWeights[k][j]);
        }
    }
    for (int j=0; j<NUM_HIDDEN_NODES; j++) {
	    ret = fscanf(file,"%lf\n",&hiddenLayerBias[j]);
    }
    for (int j=0; j<NUM_OUTPUTS; j++) {
        for (int k=0; k<NUM_HIDDEN_NODES; k++) {
	    ret = fscanf(file,"%lf\n",&outputWeights[k][j]);
        }
    }
    for (int j=0; j<NUM_OUTPUTS; j++) {
	    ret = fscanf(file,"%lf\n",&outputLayerBias[j]);
    }
    if(ret < 1){ printf("Error read NN from File\n"); }
    
    fclose(file);
}

/******************************************************************
 * Read training data file
 ******************************************************************/
void nn_read_train_file(const char* file_name)
{
  	FILE* file = fopen (file_name, "r");
  	int i = 0;
	int cnt = 0;
	unsigned int line_counter = 0;
	int ret;

  	while (!feof (file) && line_counter < NUM_TRAINING_SETS)
    	{
  		ret = fscanf (file, "%d", &i);
		if(cnt < NUM_INPUTS){
			t_input[line_counter][cnt]=scale_to_zero_one(i);
		} else if ( cnt == NUM_INPUTS ){
			if(i>0){ t_output[line_counter][0] = 1.0f; }
			else { t_output[line_counter][0] = 0.0f; }
		} else {
			line_counter++;
			cnt = 0;
			t_input[line_counter][cnt]=scale_to_zero_one(i);
		}	
		cnt++;
    	}
  	fclose (file);
	printf("Read %i lines and %i I->O combinations from %s TrainingData file -> ReturnCode %i\n",line_counter*2,line_counter,file_name,ret);
}

/******************************************************************
 * Evaluate Input against Trained Network
 ******************************************************************/
double *nn_run(double *input, size_t n){

	double *tempHiddenLayer = malloc(sizeof(double)*NUM_HIDDEN_NODES);
	double *tempOutputLayer = malloc(sizeof(double)*NUM_OUTPUTS);

	if(n>1){

            for (int j=0; j<NUM_HIDDEN_NODES; j++) {
             	double activation=hiddenLayerBias[j];
                for (int k=0; k<NUM_INPUTS; k++) {
                    	activation+=input[k]*hiddenWeights[k][j];
                }
                tempHiddenLayer[j] = sigmoid(activation);
            }
            
            for (int j=0; j<NUM_OUTPUTS; j++) {
                double activation=outputLayerBias[j];
                for (int k=0; k<NUM_HIDDEN_NODES; k++) {
                    	activation+=tempHiddenLayer[k]*outputWeights[k][j];
                }
                tempOutputLayer[j] = sigmoid(activation);
            }
	
	}

	free(tempHiddenLayer);
	return tempOutputLayer;

}

/******************************************************************
 * Init neural network with random values before Training 
 ******************************************************************/
void nn_init(){
   
   for (int i=0; i<NUM_INPUTS; i++) {
        for (int j=0; j<NUM_HIDDEN_NODES; j++) {
            hiddenWeights[i][j] = pseudo_number();
        }
    }
    for (int i=0; i<NUM_HIDDEN_NODES; i++) {
        hiddenLayerBias[i] = pseudo_number();
        for (int j=0; j<NUM_OUTPUTS; j++) {
            outputWeights[i][j] = pseudo_number();
        }
    }
    for (int i=0; i<NUM_OUTPUTS; i++) {
        outputLayerBias[i] = pseudo_number();
    }

}

/******************************************************************
 * print Neurons Values
 ******************************************************************/
void nn_print_debug(){
    printf("Hidden Weights [");
    for (int j=0; j<NUM_HIDDEN_NODES; j++) {
        for(int k=0; k<NUM_INPUTS; k++) {
	    printf("%lf|",hiddenWeights[k][j]);
        }
    }
    printf("]\nHidden Bias [");
    for (int j=0; j<NUM_HIDDEN_NODES; j++) {
    	printf("%lf|",hiddenLayerBias[j]);
    }
    printf("]\nOutput Weights [");
    for (int j=0; j<NUM_OUTPUTS; j++) {
        for (int k=0; k<NUM_HIDDEN_NODES; k++) {
    		printf("%lf|",outputWeights[k][j]);
        }
    }
    printf("]\nOutput Biases [");
    for (int j=0; j<NUM_OUTPUTS; j++) {
    		printf("%lf|",outputLayerBias[j]);
    }
    printf("]\n");
}

/******************************************************************
 * run training
 ******************************************************************/
void nn_train(int train_index_start, int train_index_end){

    long c_start = getTimeStamp();
    nn_gen_code_compile();
    long c_done = getTimeStamp();
    nn_gen_code_train();
    long t_done = getTimeStamp();

    printf("Train compile time:\t\t%lus\n",((c_done-c_start)/1000));
    printf("Train loops done:\t\t%i\n",(EPOCHS*NUM_TRAINING_SETS));
    printf("Train run time:\t\t\t%lus\n",((t_done-c_done)/1000));
}
