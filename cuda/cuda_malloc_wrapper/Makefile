CC=nvcc
CFLAGS=

cuda_malloc_wrapper : cuda_malloc_wrapper.cu
	$(CC) -o $@ $< $(CFLAGS)

.PHONY: clean

clean:
	rm cuda_malloc_wrapper