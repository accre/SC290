To run these you need to be in a Matlab group on ACCRE, then type:

	setpkgs -a matlab
	matlab -r "isGpuAvailable()"
	matlab < simpleFFT.m

Make sure you are logged into either a GPU node or gateway (vmps81).