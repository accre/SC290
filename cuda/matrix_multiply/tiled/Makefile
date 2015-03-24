CC=nvcc
CFLAGS=

mat_multiply : mat_multiply.cu
	$(CC) -o $@ $< $(CFLAGS)

.PHONY: clean

clean:
	rm mat_multiply