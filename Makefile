default:
	clang -O3 -o test test_main.c neural_network.c -ldl
	./test

clean:
	rm nn_generated_train.c
	rm *.so
	rm test
	rm nn_trained.nn
