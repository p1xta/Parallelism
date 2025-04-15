cpu_sequential:
	pgc++ -o task_cpu -lboost_program_options -acc=host -Minfo=all -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include task_cpu.cpp
	./task_cpu --size=128 --accuracy=0.000001 --max_iterations=1000000

cpu_multicore:
	pgc++ -o task_multicore -lboost_program_options -acc=multicore -Minfo=all -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include task_cpu.cpp
	./task_multicore --size=128 --accuracy=0.000001 --max_iterations=1000000

gpu:
	pgc++ -o task_gpu -lboost_program_options -acc=gpu -Minfo=all -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include task_gpu.cpp
	./task_gpu --size=1024 --accuracy=0.000001 --max_iterations=1000000
profile:
	nsys profile --trace=nvtx,cuda,openacc --stats=true ./task_gpu --size=256 --accuracy=0.0001 --max_iterations=50