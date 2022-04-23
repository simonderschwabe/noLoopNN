# Title
noLoopNN - c high performance Neural Network (loop free)

# Description

Neural Networks are requiring more and more GPU power for training. 
To break that loop of death, i implemented a Neural Network which removes
the expensive foward and backword propagation loops in training and only 
relies on formulas. As the formula can get realy big for each neuron, i
use a code generator and compile/load the (no Loop) generated training code
at runtime.

Internaly the Network scales the Training Data into 0-1 Ranges. This is why
we give SCALE_LIMIT to filter out the max jumps in Training Data in non Text Mode
to get a nice even range from min to max values.

Please be aware that as more neurons you configure as more RAM is needed for the
compiling of the training with clang. This is due to the large generated code files.

Most Network use random numbers to initialize there weights. In this case i use
mersenne twister as it gets the same good Training results as random number initialized
weights. Also due to this fact, if network ist trained with same training data and 
input parameters you also get the same training result. Which means repeatable training
results.

# Language

C

# Performance Stats

On i7-8700:

```
12 	(Input Nuerons)
27 	(Hidden Neurons)
1  	(Output Neuron)
120000  (TrainingPairs a 12 Input 1 Output)
50000	(Epochs)

120000x50000=6b(6 Billion Training loops)
50.3 Mins

6b/3018=1988071/s
1988071/(12+27+1)=49701(Trainings per Second per Neuron)
```

# Config Options

can be set in the neural_network.h
```
EPOCHS			- Number of Training Loops
NUM_INPUTS		- Number of Input Neurons
NUM_HIDDEN_NODES	- Number of Hidden Neurons
NUM_LAYER		- Number of Hidden Layers
NUM_OUTPUTS		- Number of Output Neurons
NUM_TRAINING_SETS	- Number of Training Pairs
ADJ_RATE		- Adjustment Rate of the Network during Training
SCALE_LIMIT		- Big Jumps between Training Inputs (aka 1 2 9 10) which should be defaulted 
SCALE_DEFAULT		- Default value for the above
```

# How it works in code

Include the header file include/neural_network.h into you code

```
nn_read_train_file("nn_training_data.csv");
nn_init();
nn_train(0,EPOCHS-1);
nn_save("trading_nn_train.net");
nn_load("trading_nn_train.net");
*nn_result = nn_run(inputs,NUM_INPUTS);
```

in text mode use this:
```
nn_read_train_text_file("data/training.txt");
*nn_result = nn_run_text("Ask your Network something");
```

Please free() the return Pointer from nn_run* after usage to avoid memory getting full

# Examples

The main() function in test_main.c contains an simple XOR example 
which should explain how to use the Neural Network in practice.
See build section.

# Example Training Data file

12 Input Neurons/1 Output Neuron Pairs

```
-1 2 5 6 -6 -4 -2 1 2 3 4 5
1
2 3 4 5 6 7 8 3 4 6 7 8
-1
```

in text mode it looks like this:
```
Whats your Name?
SimonsAI
Whats your Task?
Answer Questions
```

# Status

initial Version

# Supported OS and Hardware

Linux x86_64
others may work, but have not been tested

# Requirements

```
clang
ninja build
(std lib)dlfcn.h
(std lib)math.h
(std lib)time.h
```

# Build & Show

run on commandline

```
ninja
./test
```

# License

Apache License, Version 2.0

# External Code

Copyright of Mersenne Twister is at https://github.com/ESultanik/mtwister
