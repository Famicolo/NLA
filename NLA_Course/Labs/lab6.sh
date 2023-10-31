#! /bin/bash

#compile the test for eigen in lis
mpicc -DUSE_MPI -I${mkLisInc} -L${mkLisLib} -llis etest1.c -o eigen1
mpirun -n 4 ./eigen1 testmat0.mtx eigvec.txt hist.txt

#useful options for eigensolvers
#pi power met, ii inverse pw, rqi, cr gives the small eigenvalue
mpirun -n 4 ./eigen1 testmat0.mtx eigvec.txt hist.txt -e pi
mpirun -n 4 ./eigen1 testmat0.mtx eigvec.txt hist.txt -e ii
mpirun -n 4 ./eigen1 testmat0.mtx eigvec.txt hist.txt -e cg
mpirun -n 4 ./eigen1 testmat2.mtx eigvec.txt hist.txt -e cr
mpirun -n 4 ./eigen1 testmat2.mtx eigvec.txt hist.txt -e rqi

#for the inverse we can spacify the iterative solver and preconditioner
mpirun -n 4 ./eigen1 testmat0.mtx eigvec.txt hist.txt -e ii -i cg
mpirun -n 4 ./eigen1 testmat0.mtx eigvec.txt hist.txt -e ii -i gs -p ilu 
mpirun -n 4 ./eigen1 testmat2.mtx eigvec.txt hist.txt -e ii -i gmres
mpirun -n 4 ./eigen1 testmat2.mtx eigvec.txt hist.txt -e rqi -i bicgstab -p ssor

#sepecify also maxi  and tolerance
mpirun -n 4 ./eigen1 testmat0.mtx eigvec.txt hist.txt -e pi -emaxiter 100 -etol 1.e-6
mpirun -n 4 ./eigen1 testmat0.mtx eigvec.txt hist.txt -e ii -emaxiter 200 -etol 1.e-15
mpirun -n 4 ./eigen1 testmat2.mtx eigvec.txt hist.txt -e ii -i gmres -etol 1.e-14
mpirun -n 4 ./eigen1 testmat2.mtx eigvec.txt hist.txt -e ii -i gs -etol 1.e-14

#shift can acelerate the convergence or eigenpairs different from the ones chosen

mpirun -n 4 ./eigen1 testmat0.mtx eigvec.txt hist.txt -e pi -shift 4.0
mpirun -n 4 ./eigen1 testmat0.mtx eigvec.txt hist.txt -e ii -shift 2.0
mpirun -n 4 ./eigen1 testmat0.mtx eigvec.txt hist.txt -e ii -i bicgstab -shift 8.0
mpirun -n 4 ./eigen1 testmat2.mtx eigvec.txt hist.txt -e ii -shift 1.0


#computing multiple eigenpairs -ss dimension of the subspace
mpirun -n 4 ./eigen1 testmat0.mtx eigvec.txt hist.txt -e si -ss 4 
mpirun -n 4 ./eigen1 testmat0.mtx eigvec.txt hist.txt -e li -ss 4 -ie cg
mpirun -n 4 ./eigen1 testmat0.mtx eigvec.txt hist.txt -e li -ss 4 -ie ii -i bicgstab -p jacobi
mpirun -n 4 ./eigen1 testmat2.mtx eigvec.txt hist.txt -e ai -ss 2 -ie rqi


#e test5 gives another way to compute multiple eigenpairs
mpicc -DUSE_MPI -I${mkLisInc} -L${mkLisLib} -llis etest5.c -o eigen2

mpirun -n 4 ./eigen2 testmat0.mtx  evals.mtx eigvecs.mtx res.txt iters.txt -ss 4 -e li 

mpirun -n 4 ./eigen2 testmat0.mtx  evals.mtx eigvecs.mtx res.txt iters.txt -ss 4 -e li -i cg -p jacobi -etol 1.0e-10 

mpirun -n 4 ./eigen2 testmat0.mtx evals.mtx eigvecs.mtx r.txt iters.txt -e si -ie ii -ss 4 -i cg -p ssor -etol 1.0e-8

mpirun -n 4 ./eigen2 testmat2.mtx evals.mtx eigvecs.mtx res.txt iters.txt -e ai -si ii -i gmres -ss 4



#using different etest
mpicc -DUSE_MPI -I${mkLisInc} -L${mkLisLib} -llis etest2.c -o etest2
mpicc -DUSE_MPI -I${mkLisInc} -L${mkLisLib} -llis etest4.c -o etest4

mpirun -n 4 ./etest2 20 20 1 eigvec.mtx hist.txt
mpirun -n 4 ./etest2 20 20 1 eigvec.mtx hist.txt -e pi
mpirun -n 4 ./etest2 20 20 1 eigvec.mtx hist.txt -e rqi -i gmres
mpirun -n 4 ./etest2 20 20 1 eigvec.mtx hist.txt -e ii -i gs
mpirun -n 4 ./etest2 20 20 1 eigvec.mtx hist.txt -e ii -i cg -p ssor

mpirun -n 4 ./etest4 100
mpirun -n 4 ./etest4 100 -e pi -etol 1.0e-8
mpirun -n 4 ./etest4 50 -e pi -etol 1.0e-8 -emaxiter 2000
mpirun -n 4 ./etest4 50 -e ii -etol 1.0e-10 -i cg
mpirun -n 4 ./etest4 50 -e ii -etol 1.0e-10 -i bicgstab -p jacobi


#solution of the eigenpairs with eigen

#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv)
{
    int n = 50;	
    SparseMatrix<double> mat(n,n);            // define matrix
    for (int i=0; i<n; i++) {
        mat.coeffRef(i, i) = 2.0;
	      if(i>0) mat.coeffRef(i, i-1) = -1.0;
        if(i<n-1) mat.coeffRef(i, i+1) = -1.0;	
    }

   MatrixXd A;
   A = MatrixXd(mat);  //muovo ad una matrice densa altrimenti eigen non funziona
   SelfAdjointEigenSolver<MatrixXd> eigensolver(A);  //funziona per le simmetriche
   if (eigensolver.info() != Eigen::Success) abort();
   std::cout << "The eigenvalues of A are:\n" << eigensolver.eigenvalues() << std::endl;
   // std::cout << "Here's a matrix whose columns are eigenvectors of A \n"
   //           << eigensolver.eigenvectors() << std::endl;
    return 0;    
}




#to use additive swarz preconditioner
mpirun -n 4 ./test1 testmat0.mtx 2 sol.mtx hist.txt -i cg -adds true -p jacobi

#play with additive shwarz
mpirun -n 2 ./test1 testmat2.mtx 2 sol.mtx hist.txt -i gmres
mpirun -n 2 ./test1 testmat2.mtx 2 sol.mtx hist.txt -i gmres -adds true -p ssor
#great it reduction

mpirun -n 2 ./test1 testmat2.mtx 2 sol.mtx hist.txt -i bicgstab
mpirun -n 2 ./test1 testmat2.mtx 2 sol.mtx hist.txt -i bicgstab -adds true -p ilu -ilu_fill 2
#grat it reduction
