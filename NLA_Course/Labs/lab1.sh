#! /bin/bash

#iterative solvers with lis

#compile test1
mpicc -DUSE_MPI -I${mkLisInc} -L${mkLisLib} -llis test1.c -o test1

#run test code 1 b=1 1 1 ...
./test1 testmat0.mtx 1 sol.txt hist.txt

./test1 testmat0.mtx testvec0.mtx sol.txt hist.txt

#run with multiprocessor
mpirun -n 4 ./test1 testmat0.mtx 1 sol.txt hist.txt

#download and run matricies
wget https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/laplace/gr_30_30.mtx.gz
gzip gr_30_30.mtx.gz

mpirun -n 4 ./test1 gr_30_30.mtx 2 sol.txt hist.txt

#use different solvers and options
mpirun -n 4 ./test1 gr_30_30.mtx 2 sol.txt hist.txt -i cg 

mpirun -n 4 ./test1 gr_30_30.mtx 2 sol.txt hist.txt -i jacobi -maxiter 2000

mpirun -n 4 ./test1 gr_30_30.mtx 2 sol.txt hist.txt -i gs -tol 1.0e-10

#playng with other options
mpirun -n 4 ./test1 gr_30_30.mtx 2 sol.txt hist.txt -tol 1.0e-14

mpirun -n 4 ./test1 gr_30_30.mtx 2 sol.txt hist.txt -i bicgstab -maxiter 100

mpirun -n 4 ./test1 gr_30_30.mtx 2 sol.txt hist.txt -i gmres -restart 20

mpirun -n 4 ./test1 gr_30_30.mtx 2 sol.txt hist.txt -i bicg -p jacobi

#test 5 define matrix with gamma

./test5 100 0.1 -i jacobi

mpirun -n 2 ./test5 100 -1.0 -i gs 

mpirun -n 2 ./test5 100 12.0 -i gmres -restart 30  

mpirun -n 4 ./test5 100 9.9 -i bicgstab -p sainv

