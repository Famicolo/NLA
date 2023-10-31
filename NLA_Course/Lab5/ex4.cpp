#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <iostream>
#include <string>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/SparseExtra>

#include "cg.hpp"
#include "cgs.hpp"

int main(int argc, char *argv[]){
  using namespace Eigen;
  using namespace LinearAlgebra;

  if(argc != 2)
    {
      std::cerr << " Usage: provide matrix filename" << std::endl;
      return 1;
    }
  std::string matrixFile(argv[1]);
  // Some useful alias
  using SpMat = Eigen::SparseMatrix<double>;
  using SpVec = Eigen::VectorXd;

  // Read matrix
  SpMat A;
  Eigen::loadMarket(A, matrixFile);
  std::cout<<"Matrix size:"<<A.rows()<<"X"<<A.cols()<<std::endl;
  std::cout<<"Non zero entries:"<<A.nonZeros()<<std::endl;

  double tol = 1.e-6;                  // Convergence tolerance
  int result, maxit = 2000;            // Maximum iterations
  SpMat B = SpMat(A.transpose()) - A;  // Check symmetry
  std::cout<<"Norm of A-A.t: "<<B.norm()<<std::endl;

  // Create Rhs b
  SpVec e = SpVec::Ones(A.rows()), b(A.rows()), x(A.rows());
  b=A*e;


  // First with Eigen Choleski and Eigen SparseLU solvers
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver;  // LDLT factorization
  solver.compute(A);
  if(solver.info()!=Eigen::Success) {                          // sanity check
      std::cout << "cannot factorize the matrix" << std::endl; // decomposition failed
      return 0;
  }
  x = solver.solve(b);                                         // solving
  std::cout << "Solution with Eigen Choleski:" << std::endl;
  std::cout << "effective error: "<<(x-e).norm()<< std::endl;

  x=0*x;

   Eigen::SparseLU<Eigen::SparseMatrix<double> > solvelu;   // LU
  solvelu.compute(A);
  if(solvelu.info()!=Eigen::Success) {                          // sanity check
      std::cout << "cannot factorize the matrix" << std::endl;  // decomposition failed
      return 0;
  }

  x = solvelu.solve(b);                    
  std::cout << "Solution with Eigen LU:" << std::endl;
  std::cout << "effective error: "<<(x-e).norm()<< std::endl;

  // Then with hand-made CG or CGS method
 /** x=0*x;
  Eigen::DiagonalPreconditioner<double> D(A);
  result = CG(A,x,b,D,maxit,tol);
  std::cout << "Solution with Conjugate Gradient:" << std::endl;
  std::cout << "iterations performed: " << maxit << std::endl;
  std::cout << "tolerance achieved  : " << tol << std::endl;
  std::cout << "Error norm: "<<(x-e).norm()<< std::endl;
**/
  x=0*x;
  maxit = 2000;
  tol = 1.e-6;                  // Convergence tolerance
  Eigen::DiagonalPreconditioner<double> D(A);
  result = CGS(A,x,b,D,maxit,tol);
  std::cout << "Solution with Conjugate Gradient Stabilized:" << std::endl;
  std::cout << "iterations performed: " << maxit << std::endl;
  std::cout << "tolerance achieved  : " << tol << std::endl;
  std::cout << "Error norm: "<<(x-e).norm()<< std::endl;

  return result;
}