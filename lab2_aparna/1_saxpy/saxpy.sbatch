#!/bin/bash
#SBATCH -J saxpy           # job name
#SBATCH -o outdir/saxpy.o%j       # output and error file name (%j expands to jobID)
#SBATCH -N 1               # total number of node
#SBATCH -n 1               # total number of mpi tasks requested
#SBATCH -p gtx             # queue (partition) -- normal, development, etc.
#SBATCH -t 00:30:00        # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=aparnakumari@utexas.edu
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH -A EE382N-20-Parallelis

module load gcc
module load cuda
module list
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/8.0/lib64
#echo "SAXPY BASIC "
#./cudaSaxpy -n 500000000 -p 1 -i 4
#nvprof ./cudaSaxpy -n 500000000 -p 1 -i 4
#echo "SAXPY UVM "
#./cudaSaxpyUvm -n 500000000 -p 1 -i 4
#echo "SAXPY STREAM SERIAL "
#./cudaSaxpyStream -n 500000000 -p 1 -i 4
echo "SAXPY STREAM PARALLEL"
./cudaSaxpyStream -n 500000000 -p 4 -i 4
nvprof ./cudaSaxpyStream -n 500000000 -p 4 -i 4

#nvprof --unified-memory-profiling ./cudaSaxpyUvm -n 500000000 -p 1 -i 4
