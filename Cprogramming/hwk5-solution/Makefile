CC=/usr/local/gcc/latest/bin/gcc
CFLAGS=-Wall -O3

matrix_multiply : matrix_multiply.c
	$(CC) -o $@ $< $(CFLAGS)

generate_rand_matrix : generate_rand_matrix.c
	$(CC) -o $@ $< $(CFLAGS)

.PHONY: clean

clean :
	rm matrix_multiply