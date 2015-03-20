CC=nvcc
CFLAGS=-I/usr/local/cuda-7.0/samples/common/inc

vec_add : vector_add.cu
	$(CC) -o $@ $< $(CFLAGS)

.PHONY: clean

clean:
	rm vec_add