A = gpuArray(rand(2^16,1)); % gpuArray data types are for GPU processing
B = fft(A); % take the fast Fourier transform of A on the GPU
class(B) % print the data type of B
C = gather(B); % convert B to data type that can be processed on host