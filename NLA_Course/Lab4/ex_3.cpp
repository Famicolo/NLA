#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>

using std::endl;
using std::cout;

#include "cg.hpp"
#include "jacobi.hpp"

int main(int argc, char** argv)
{
  using namespace LinearAlgebra;
  // Some useful alias
  using SpMat=Eigen::SparseMatrix<double>;
  using SpVec=Eigen::VectorXd;
  //using namespace Eigen;

  int n = 100;
  SpMat A(n,n);                      // define matrix
  A.reserve(298);
  for (int i=0; i<n; i++) {
      A.coeffRef(i, i) = 2.0;
      if(i>0) A.coeffRef(i, i-1) = -1.0;
      if(i<n-1) A.coeffRef(i, i+1) = -1.0;
  }

  double tol = 1.e-6;                // Convergence tolerance
  int result, maxit = 100000;          // Maximum iterations

  std::cout<<"Matrix size:"<<A.rows()<<"X"<<A.cols()<<std::endl;
  std::cout<<"Non zero entries:"<<A.nonZeros()<<std::endl;
  //fai attenzione alla definizione di SpMat ogni volta che generi una nuova matrice altrimenti collassa tutto
  SpMat B = SpMat(A.transpose()) - A;
  std::cout << "Check A norm for simmetry: " << B.norm() << endl;
  SpVec e = SpVec::Ones(A.rows());
  SpVec b = A*e;
  SpVec x(A.rows());

  // Solve with CG method
  x=0*x;
  Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper> cg;
  cg.setMaxIterations(maxit);
  cg.setTolerance(tol);
  cg.compute(A);
  x = cg.solve(b);
  std::cout <<" Eigen native CG"<< endl;
  std::cout << "#iterations:     " << cg.iterations() << endl;
  std::cout << "estimated error: " << cg.error()      << endl;
  std::cout << "effective error: "<<(x-e).norm()<< endl; 

  // Solve with Jacobi method
  x=0*x;
  Eigen::DiagonalPreconditioner<double> D(A);
  result = Jacobi(A,x,b,D,maxit,tol);

  std::cout <<" hand-made Jacobi "<< endl;
  cout << "Jacobi flag = " << result << endl;
  cout << "iterations performed: " << maxit << endl;
  cout << "tolerance achived: " << tol <<endl;
  cout << "effective error: " << (x-e).norm() << endl;

  return 0;
}