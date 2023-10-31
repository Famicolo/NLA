#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/SparseExtra>
#include "bcgstab.hpp"

using namespace std;

// Some useful alias
using SpMat=Eigen::SparseMatrix<double>;
using SpVec=Eigen::VectorXd;

int main(int argc, char** argv){
    using namespace LinearAlgebra;

    SpMat A1, A2, A;
    Eigen::loadMarket(A1, "A1.mtx");
    Eigen::loadMarket(A2, "A2.mtx");
    A=A1*A2;

    std::cout << "Matrix size:"<< A.rows() << "X" << A.cols() << endl;
    std::cout << "Non zero entries:" << A.nonZeros() << endl;
    std::cout << "Euclidean norm of A: " << A.norm() << endl;
    

    SpVec b = SpVec::Ones(A.rows());
    SpVec x(A.rows());

    x=0*x;
  Eigen::DiagonalPreconditioner<double> D(A);
  int maxit = 100000; 
  double tol = 1.e-10;
  int result = BiCGSTAB(A, x, b, D, maxit, tol);
  cout << "BiCGSTAB flag = " << result << endl;
  cout << "iterations performed: " << maxit << endl;
  cout << "tolerance achived: " << tol <<endl;
  //cout << "effective error: " << (x-b).norm() << endl;
  double residual = (b-A*x).norm()/b.norm();
  cout << "Relative residual: " << residual << endl;
  cout << endl;




}