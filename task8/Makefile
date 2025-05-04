gpu:
	nvcc -lboost_program_options -o task task.cu
	./task --size=1024 --accuracy=0.000001 --max_iterations=1000000
profile:
	nsys profile --trace=nvtx,cuda,openacc --stats=true ./task --size=128 --accuracy=0.0001 --max_iterations=50