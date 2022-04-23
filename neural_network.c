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
#include "stddef.h"
#include "string.h"
#include "time.h"
#include <dlfcn.h>
#include "include/neural_network.h"
#include "include/tiny_mt.h"

signed int scale_min = 0;
signed int scale_max = 0;

/******************************************************************
 * Variable declation for neural network
 ******************************************************************/
double *hW = NULL;
double *oW = NULL;

double *hLW = NULL;
double *hL = NULL;
double *oL = NULL;

double *hLB = NULL;
double *oLB = NULL;

double *ti = NULL;
double *to = NULL;

double hiddenWeights[NUM_INPUTS][NUM_HIDDEN_NODES];
double outputWeights[NUM_HIDDEN_NODES][NUM_OUTPUTS];

double hiddenLayerWeights[NUM_LAYER][NUM_HIDDEN_NODES][NUM_HIDDEN_NODES];
double hiddenLayer[NUM_LAYER][NUM_HIDDEN_NODES];
double outputLayer[NUM_OUTPUTS];

double hiddenLayerBias[NUM_LAYER][NUM_HIDDEN_NODES];
double outputLayerBias[NUM_OUTPUTS];

double t_input[NUM_TRAINING_SETS][NUM_INPUTS];
double t_output[NUM_TRAINING_SETS][NUM_OUTPUTS];

/******************************************************************
 * Function for Activation and the Derivative of it 
 * maybe switch to ReLU later
 ******************************************************************/
double sigmoid(double x) { 
	if(x> 8){return 0.999999;} 
	if(x<-8){return 0.000001;} 
	if(x> 6){return 0.998+(x*0.0001);}
	if(x> 3.25){return 0.975+(x*0.0028);}
	if(x>=0){return 0.5+((x*0.7777)-(x/1.5713));}
	if(x< 0){return ((8+x)*(8+x))*0.0071;}
	return 0.5;
}
double dSigmoid(double x) { return x * (1 - x); }

long getTimeStamp();
/******************************************************************
 * Function for init between 0 and 1 
 ******************************************************************/
double pseudo_number(MTRand *seed){

        double rN = 0.234f;
	double pos_neg = 1.0f;

newNumber:
	rN = genRand(seed);

	if((unsigned long)rN & 1) pos_neg = -1.0f;
	else pos_neg = 1.0f;

        if((rN < 0.03f || rN > 0.97f) || (rN > 0.46 && rN < 0.54)) { goto newNumber; };

        return rN*pos_neg;
}

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
 * Scale Int(char) to range between 0 and 1
 ******************************************************************/
double scale_text_to_zero_one(int input){
	if(input >= 200){ return 0; }
	else if(input <= 0){ return 0; }
	return (((double)input)/(200));
}

/******************************************************************
 * convert between 0 and 1 to char
 ******************************************************************/
char scale_zero_one_to_text(double input){
	if(input >= 200) { return (char)32; }
	if(input <= 0) { return (char)32; }
	return (char)((int)((input*200)+0.1));
}

/******************************************************************
 * getTimeStamp for Measurement
 ******************************************************************/
long getTimeStamp(){

        struct timespec _t;
        clock_gettime(CLOCK_REALTIME, &_t);
        return  _t.tv_sec*1000 + (_t.tv_nsec/1.0e6);
}

/******************************************************************
 * read text into training Input/Output from File 
 ******************************************************************/
void nn_read_train_text_file(const char* file_name)
{
	FILE* file = fopen (file_name, "r");
	char c;
	int cnt = 0;
	unsigned int line_counter = 0;
	unsigned int i_o_flag = 1;
	int ret;

	while (!feof (file) && line_counter < NUM_TRAINING_SETS)
	{
		ret = fscanf (file, "%c", &c);
		if((int)c != 10){
			if(i_o_flag == 1 && cnt < NUM_INPUTS){
				t_input[line_counter][cnt]=scale_text_to_zero_one((int)c);
			}
			else if(i_o_flag == 0 && cnt < NUM_OUTPUTS){
				t_output[line_counter][cnt]=scale_text_to_zero_one((int)c);
			}
			cnt++;
		}
		else if((int)c == 10) {
			cnt = 0;
			if(i_o_flag == 1) { i_o_flag = 0; }
			else { i_o_flag = 1; line_counter++; }
		}
	}
	fclose (file);
	printf("Read %i lines and %i I->O combinations from %s TrainingData file -> ReturnCode %i\n",line_counter*2,line_counter,file_name,ret);
}

/******************************************************************
 * Convert Text into Double Array for NN Input
 ******************************************************************/
double *nn_text_to_input(const char *text){

	size_t length = strlen(text);
	double *input_array = malloc(sizeof(double)*length);

	for(int i=0; i<length; i++){
		*(input_array+i) = scale_text_to_zero_one((int)*(text+i));
	}

	return input_array;
}

/******************************************************************
 * Convert NN Output to readable Text
 ******************************************************************/
char *nn_output_to_text(const double *output){

	char *text_output = malloc(sizeof(char)*NUM_OUTPUTS);

	for(int i = 0; i<NUM_OUTPUTS; i++){
		*(text_output+i) = scale_zero_one_to_text(*(output+i));
	}

	return text_output;
}

/******************************************************************
 * Run NN on Text
 ******************************************************************/
char *nn_run_text(char *text){

	printf("nn_r: %s\n",text);
	double *input = nn_text_to_input(text);
	size_t input_legth = strlen(text);
	double *output = nn_run(input,input_legth);
	char *str = nn_output_to_text(output);

	free(input);
	free(output);

	return str;
}

/******************************************************************
 * Header Lines in generated Code outside function
 ******************************************************************/
int nn_generate_training_code_header(FILE *src_file){

	fprintf(src_file,"#include <stddef.h>\n");
	fprintf(src_file,"#include <stdio.h>\n");
	fprintf(src_file,"#include \"include/neural_network.h\"\n");

	fprintf(src_file,"extern double hiddenWeights[NUM_INPUTS][NUM_HIDDEN_NODES];\n");
	fprintf(src_file,"extern double outputWeights[NUM_HIDDEN_NODES][NUM_OUTPUTS];\n");

	fprintf(src_file,"extern double hiddenLayerWeights[NUM_LAYER][NUM_HIDDEN_NODES][NUM_HIDDEN_NODES];\n");
	fprintf(src_file,"extern double hiddenLayer[NUM_LAYER][NUM_HIDDEN_NODES];\n");
	fprintf(src_file,"extern double outputLayer[NUM_OUTPUTS];\n");

	fprintf(src_file,"extern double hiddenLayerBias[NUM_LAYER][NUM_HIDDEN_NODES];\n");
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
          fprintf(src_file,"double deltaHidden[NUM_LAYER][NUM_HIDDEN_NODES];\n");
	  code_lines_generated++;
	  code_lines_generated++;
	  fprintf(src_file,"for (int n=0; n < %i; n++) {\n",EPOCHS);
	  code_lines_generated++;
          fprintf(src_file,"for (int x=0; x<NUM_TRAINING_SETS; x++) {\n");
	  code_lines_generated++;

	  for (int j=0; j<NUM_HIDDEN_NODES; j++) {
		fprintf(src_file,"hiddenLayer[0][%i] = sigmoid(hiddenLayerBias[0][%i]",j,j);
                for (int k=0; k<NUM_INPUTS; k++) {
		    	fprintf(src_file,"+(t_input[x][%i]*hiddenWeights[%i][%i])\n",k,k,j);
	  		code_lines_generated++;
                }
		fprintf(src_file,");\n");
	  	code_lines_generated++;
          }

	  for (int p=1; p<NUM_LAYER; p++){
		 for (int j=0; j<NUM_HIDDEN_NODES; j++){
			fprintf(src_file,"hiddenLayer[%i][%i] = sigmoid(hiddenLayerBias[%i][%i]",p,j,p,j);
			for (int k=0; k<NUM_HIDDEN_NODES; k++){
				fprintf(src_file,"+(hiddenLayer[%i-1][%i]*hiddenLayerWeights[%i][%i][%i])\n",p,j,p,j,k);
				code_lines_generated++;
			}
			fprintf(src_file,");\n");
			code_lines_generated++; 
		 }
	  }

	  for (int j=0; j<NUM_OUTPUTS; j++) {
		fprintf(src_file,"outputLayer[%i] = sigmoid(outputLayerBias[%i]",j,j);
           	for (int k=0; k<NUM_HIDDEN_NODES; k++) {
			fprintf(src_file,"+(hiddenLayer[NUM_LAYER-1][%i]*outputWeights[%i][%i])\n",k,k,j);
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
		fprintf(src_file,"deltaHidden[NUM_LAYER-1][%i] = (",j);
                for(int k=0; k<NUM_OUTPUTS; k++) {
		    fprintf(src_file,"(deltaOutput[%i]*outputWeights[%i][%i])\n",k,j,k);
		    code_lines_generated++;
		    if(k+1 < NUM_OUTPUTS){
			    fprintf(src_file,"+");
		    }
                }
		fprintf(src_file,")*dSigmoid(hiddenLayer[NUM_LAYER-1][%i]);\n",j);
		code_lines_generated++;
          }
            
          for (int j=0; j<NUM_OUTPUTS; j++) {
		fprintf(src_file,"outputLayerBias[%i] += deltaOutput[%i]*ADJ_RATE;\n",j,j);
		code_lines_generated++;
                for (int k=0; k<NUM_HIDDEN_NODES; k++) {
		    fprintf(src_file,"outputWeights[%i][%i] += hiddenLayer[NUM_LAYER-1][%i]*deltaOutput[%i]*ADJ_RATE;\n",k,j,k,j);
		    code_lines_generated++;
                }
          }
	  
	  for (int j=0; j<NUM_HIDDEN_NODES; j++) {
		fprintf(src_file,"hiddenLayerBias[NUM_LAYER-1][%i] += deltaHidden[NUM_LAYER-1][%i]*ADJ_RATE;\n",j,j);
		code_lines_generated++;
                for(int k=0; k<NUM_HIDDEN_NODES; k++) {
		for(int p=0; p<NUM_OUTPUTS; p++){
		    fprintf(src_file,"hiddenLayerWeights[NUM_LAYER-1][%i][%i] += t_output[x][%i]*deltaHidden[NUM_LAYER-1][%i]*ADJ_RATE;\n",j,k,p,k);
		    code_lines_generated++;
                }
		}
          }

	  for (int p=NUM_LAYER-2; p>=0; p--){
		  for(int j=0; j<NUM_HIDDEN_NODES; j++){
			  fprintf(src_file,"deltaHidden[%i][%i] = deltaHidden[%i+1][%i]*sigmoid(hiddenLayer[%i][%i]);\n",p,j,p,j,p,j);
		  }
		  for(int j=0; j<NUM_HIDDEN_NODES; j++){
			fprintf(src_file,"hiddenLayerBias[%i][%i] += deltaHidden[%i][%i]*(ADJ_RATE/1.72f);\n",p,j,p,j);
			if(p>0){
                	for(int k=0; k<NUM_HIDDEN_NODES; k++) {
		    		for(int i=0; i<NUM_INPUTS; i++){
		    		fprintf(src_file,"hiddenLayerWeights[%i][%i][%i] += t_input[x][%i]*deltaHidden[%i][%i]*(ADJ_RATE/1.72f);\n",p,j,k,i,p,k);
		    		}
			}
			}
		  }
	  }

	  for (int j=0; j<NUM_HIDDEN_NODES; j++) {
                for(int k=0; k<NUM_INPUTS; k++) {
		    fprintf(src_file,"hiddenWeights[%i][%i] += t_input[x][%i]*deltaHidden[NUM_LAYER-1][%i]*ADJ_RATE;\n",k,j,k,j);
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
	}
	ptr_offset = 0;
	for(int x=0; x < NUM_LAYER; x++){
		for(int n=0; n < NUM_HIDDEN_NODES; n++){
			*(hL+ptr_offset) = hiddenLayer[x][n];
			*(hLB+ptr_offset) = hiddenLayerBias[x][n];
			ptr_offset += 1;
		}
	}
	ptr_offset = 0;
	for(int x=0; x < NUM_LAYER; x++){
		for(int y=0; y < NUM_HIDDEN_NODES; y++){
			for(int z = 0; z < NUM_HIDDEN_NODES; z++){
				*(hLW+ptr_offset) = hiddenLayerWeights[x][y][z];
				ptr_offset += 1;
			}
		}
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
	}
	ptr_offset = 0;
	for(int x=0; x < NUM_LAYER; x++){
		for(int n=0; n < NUM_HIDDEN_NODES; n++){
			hiddenLayer[x][n] = *(hL+ptr_offset);
			hiddenLayerBias[x][n] = *(hLB+ptr_offset);
			ptr_offset += 1;
		}
	}
	ptr_offset = 0;
	for(int x=0; x < NUM_LAYER; x++){
		for(int y=0; y < NUM_HIDDEN_NODES; y++){
			for(int z = 0; z < NUM_HIDDEN_NODES; z++){
				hiddenLayerWeights[x][y][z] = *(hLW+ptr_offset);
				ptr_offset += 1;
			}
		}
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

	retcode = system("clang -s -shared -Ofast -o nn_generated.so -fPIC nn_generated_train.c neural_network.c tiny_mt.c");

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
	hLW = (double *) dlsym(handle, "hiddenLayerWeights");
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
    for (int u=0; u<NUM_LAYER; u++){
    	for (int j=0; j<NUM_HIDDEN_NODES; j++) {
	    for(int v=0; v < NUM_HIDDEN_NODES; v++){
	    	fprintf(file, "%lf\n", hiddenLayerWeights[u][j][v]);
	    }
	    fprintf(file, "%lf\n", hiddenLayerBias[u][j]);
    	}
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
    for (int u=0; u<NUM_LAYER; u++){
    	for (int j=0; j<NUM_HIDDEN_NODES; j++) {
	    for(int v=0; v<NUM_HIDDEN_NODES; v++){
	    	ret = fscanf(file,"%lf\n",&hiddenLayerWeights[u][j][v]);
	    }
	    ret = fscanf(file,"%lf\n",&hiddenLayerBias[u][j]);
    	}
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

	double *tempOutputLayer = malloc(sizeof(double)*NUM_OUTPUTS);

	if(n>1){

            for (int j=0; j<NUM_HIDDEN_NODES; j++) {
             	double activation=hiddenLayerBias[0][j];
                for (int k=0; k<NUM_INPUTS; k++) {
                    	activation+=input[k]*hiddenWeights[k][j];
                }
                hiddenLayer[0][j] = sigmoid(activation);
            }

	    for (int j=1; j<NUM_LAYER; j++){
		    for(int k=0; k<NUM_HIDDEN_NODES; k++){
			    double activation=hiddenLayerBias[j][k];
			    for(int u=0; u<NUM_HIDDEN_NODES; u++){
				    activation+=hiddenLayer[j-1][k]*hiddenLayerWeights[j][k][u];
			    }
			    hiddenLayer[j][k]=sigmoid(activation);
		    }
	    }

            for (int j=0; j<NUM_OUTPUTS; j++) {
                double activation=outputLayerBias[j];
                for (int k=0; k<NUM_HIDDEN_NODES; k++) {
                    	activation+=hiddenLayer[NUM_LAYER-1][k]*outputWeights[k][j];
                }
                tempOutputLayer[j] = sigmoid(activation);
            }
	
	}

	return tempOutputLayer;

}

/******************************************************************
 * Init neural network with random values before Training 
 ******************************************************************/
void nn_init(){
	
	MTRand r = seedRand(1337);

   	for (int i=0; i<NUM_INPUTS; i++) {
        	for (int j=0; j<NUM_HIDDEN_NODES; j++) {
            		hiddenWeights[i][j] = pseudo_number(&r);
        	}
   	}
	for (int i=0; i<NUM_HIDDEN_NODES; i++){
		for (int j=0; j<NUM_OUTPUTS; j++){
			outputWeights[i][j] = pseudo_number(&r);
		}
	}
	for (int i=0; i<NUM_LAYER; i++){
        	for (int j=0; j<NUM_HIDDEN_NODES; j++) {
			for(int u=0; u<NUM_HIDDEN_NODES; u++){
				hiddenLayerWeights[i][j][u] = pseudo_number(&r);
			}
		}
	}
	for (int i=0; i<NUM_LAYER; i++){
        	for (int j=0; j<NUM_HIDDEN_NODES; j++) {
			hiddenLayerBias[i][j] = pseudo_number(&r);
		}
	}
	for (int i=0; i<NUM_OUTPUTS; i++){
		outputLayerBias[i] = pseudo_number(&r);
	}

}

/******************************************************************
 * print Neurons Values
 ******************************************************************/
void nn_print_debug(){

    printf("********************************************************\n");
    for (int j=0; j<NUM_HIDDEN_NODES; j++) {
	printf("Hidden Weights Node \t[%i]\t[",j+1);
        for(int k=0; k<NUM_INPUTS; k++) {
	    printf("%.2lf|",hiddenWeights[k][j]);
        }
	printf("]\n");
    }
    printf("********************************************************\n");
    for(int i=0; i < NUM_LAYER; i++){
    	for (int j=0; j<NUM_HIDDEN_NODES; j++) {
		if(i>0){
    			printf("Hidden Layer Weights \t[%i][%i]\t[",i,j);
			for(int v=0; v<NUM_HIDDEN_NODES; v++){
				printf("%.2lf|",hiddenLayerWeights[i][j][v]);
			}
			printf("]\n");
		}
    	}
    	printf("Hidden Layer \t\t[%i]\t[",i);
    	for (int j=0; j<NUM_HIDDEN_NODES; j++) {
		printf("%.2lf|",hiddenLayer[i][j]);
    	}
    	printf("]\nHidden Bias \t\t[%i]\t[",i);
    	for (int j=0; j<NUM_HIDDEN_NODES; j++) {
    		printf("%.2lf|",hiddenLayerBias[i][j]);
    	}
    	printf("]\n");
    }
    printf("********************************************************\n");
    for (int j=0; j<NUM_OUTPUTS; j++) {
	printf("Output Weights %i Output \t[",j+1);
        for (int k=0; k<NUM_HIDDEN_NODES; k++) {
    		printf("%.2lf|",outputWeights[k][j]);
        }
	printf("]\n");
    }
    printf("Output Layer \t\t\t[");
    for (int j=0; j<NUM_OUTPUTS; j++) {
	printf("%.2lf|",outputLayer[j]);
    }
    printf("]\nOutput Biases \t\t\t[");
    for (int j=0; j<NUM_OUTPUTS; j++) {
	printf("%.2lf|",outputLayerBias[j]);
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
