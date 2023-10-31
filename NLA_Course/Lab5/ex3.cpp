#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include "cg.hpp"

int main(int argc, char** argv)
{
  using namespace LinearAlgebra;
  using namespace std;
  // Some useful alias
  using SpMat=Eigen::SparseMatrix<double>;
  using SpVec=Eigen::VectorXd;

  int n = 1000;
  SpMat A(n,n), H(n,n), B(n,n);                       // define matrix
  for (int i=0; i<20; i++) {
      for (int j=0; j<20; j++){
          H.coeffRef(i, j) = 1.0/(i+j+1);
      }
  }
  for (int i = 0; i < n; i++)
  {
    B.coeffRef(i,i) = 2.0;
    if(i<n-1) B.coeffRef(i,i+1) = -1.0;
    if(i>0) B.coeffRef(i,i-1) = -1.0;
    
  }

  A = H+B;
  //cout << "plot matrix A: " <<endl; 
  //cout << A.nonZeros() << endl; 

  SpMat C = SpMat(A.transpose())-A;
  cout << "norma di A: "<< C.norm() << endl;

  
  double tol = 1.e-8;                  // Convergence tolerance
  int result, maxit = 1000;            // Maximum iterations

  // Create Rhs b
  SpVec x_exact(n),b(n),x(n);
  for (int i = 0; i < n; i++)
  {
    x_exact(i)=1;
  }
  b = A*x_exact;
  //cout << b <<endl;
  
  


  // First with Eigen Choleski direct solver
  SpVec e = SpVec::Ones(A.rows());
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver;  // LDLT factorization
  solver.compute(A);
  if(solver.info()!=Eigen::Success) {                          // sanity check
      std::cout << "cannot factorize the matrix" << std::endl; // decomposition failed
      return 0;
  }
  x = solver.solve(b);                                         // solving
  std::cout << "Solution with Eigen Choleski:" << std::endl;
  std::cout << "effective error: "<<(x-e).norm()<< std::endl;

  // Then with Eigen SparseLU solver
  x=x*0;
  Eigen::SparseLU<Eigen::SparseMatrix<double> > solvelu;   // LU
  solvelu.compute(A);
  if(solvelu.info()!=Eigen::Success) {                          // sanity check
      std::cout << "cannot factorize the matrix" << std::endl;  // decomposition failed
      return 0;
  }

  x = solvelu.solve(b);                    
  std::cout << "Solution with Eigen LU:" << std::endl;
  std::cout << "effective error: "<<(x-e).norm()<< std::endl;

  // Compare with hand-made CG
  x=0*x;
  Eigen::DiagonalPreconditioner<double> D(A);
  result = CG(A,x,b,D,maxit,tol);
  std::cout << "Solution with Conjugate Gradient:" << std::endl;
  std::cout << "iterations performed: " << maxit << std::endl;
  std::cout << "tolerance achieved  : " << tol << std::endl;
  std::cout << "Error norm: "<<(x-e).norm()<< std::endl;

  return result;
}