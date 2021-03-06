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
#ifndef NEURAL_NETWORK_H 
#define NEURAL_NETWORK_H

/* Definition */
#define NN_GEN_FILENAME "nn_generated_train.c"
#define NN_GEN_LIBNAME "nn_generated.so"

#ifndef EPOCHS
	#define EPOCHS 250000
#endif
#ifndef NUM_INPUTS
	#define NUM_INPUTS 16
#endif
#ifndef NUM_HIDDEN_NODES
	#define NUM_HIDDEN_NODES 17
#endif
#ifndef NUM_LAYER
	#define NUM_LAYER 6
#endif
#ifndef NUM_OUTPUTS
	#define NUM_OUTPUTS 16
#endif
#ifndef ADJ_RATE
	#define ADJ_RATE 0.1f
#endif
#ifndef NUM_TRAINING_SETS
	#define NUM_TRAINING_SETS 12
#endif
#ifndef SCALE_LIMIT
        #define SCALE_LIMIT 5
#endif
#ifndef SCALE_DEFAULT
        #define SCALE_DEFAULT 0.5f
#endif


/* Functions */
void nn_save(const char*);
void nn_load(const char*);
void nn_read_train_file(const char*);
void nn_read_train_text_file(const char *);
double *nn_run(double *, size_t);
char   *nn_run_text(char *);
void nn_init();
void nn_print_debug();
void nn_train(int, int);
double scale_to_zero_one(int);
double scale_text_to_zero_one(int);
char   scale_zero_one_to_text(double);
double *nn_text_to_input(const char *);
char   *nn_output_to_text(const double *);
double sigmoid(double);
double dSigmoid(double);

#endif /* NEURAL_NETWORK_H */
