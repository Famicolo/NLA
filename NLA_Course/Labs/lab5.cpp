//direct methods vs iterative methods

//the native solver in eigen for cholesky/lu

Eigen::SparseLU<Eigen::SparseMatrix<double> > solver;    // define solver
  solvelu.compute(A);
  if(solvelu.info()!=Eigen::Success) {                     // sanity check
      std::cout << "cannot factorize the matrix" << std::endl; 
      return 0;
  }
  x = solvelu.solve(b);   


  //hand made gradient method

  namespace LinearAlgebra
{
template <class Matrix, class Vector, class Preconditioner>
int GRAD(const Matrix &A, Vector &x, const Vector &b, const Preconditioner &M,
   int &max_iter, typename Vector::Scalar &tol)
{
  using Real = typename Matrix::Scalar;
  Real   resid;
  Vector q(b.size());
  Vector z(b.size());
  Real   alpha, rho;

  Real   normb = b.norm();
  Vector r = b - A * x;
  if(normb == 0.0) normb = 1;

  if((resid = r.norm() / normb) <= tol)
    {
      tol = resid;
      max_iter = 0;
      return 0;
    }

  for(int i = 1; i <= max_iter; i++)
    {
      z = M.solve(r);
      rho = r.dot(z);
      q = A * z;
      alpha = rho / z.dot(q);

      x += alpha * z;
      r -= alpha * q;

      if((resid = r.norm() / normb) <= tol)
        {
          tol = resid;
          max_iter = i;
          return 0;
        }
    }

  tol = resid;
  return 1;
}
} // namespace LinearAlgebra     


//test preconditioned gradient method
#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>

using std::endl;
using std::cout;

#include "grad.hpp"
#include "gmres.hpp"

int main(int argc, char** argv)
{
  using namespace LinearAlgebra;
  // Some useful alias
  using SpMat=Eigen::SparseMatrix<double>;
  using SpVec=Eigen::VectorXd;

  int n = 400;
  SpMat A(n,n);                       // define matrix
  for (int i=0; i<n; i++) {
      A.coeffRef(i, i) = 2.0*(i+1);
      if(i>0) A.coeffRef(i, i-1) -= i;
      if(i<n-1) A.coeffRef(i, i+1) -= (i+1);
  }

  double tol = 1.e-8;                  // Convergence tolerance
  int result, maxit = 10000;           // Maximum iterations

  std::cout<<"Matrix size:"<<A.rows()<<"X"<<A.cols()<<endl;
  std::cout<<"Non zero entries:"<<A.nonZeros()<<endl;

  // Create Rhs b
  SpVec e = SpVec::Ones(A.rows());
  SpVec b = A*e;
  SpVec x(A.rows());
  Eigen::DiagonalPreconditioner<double> D(A);

  // First with Eigen CG
  Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper> cg;
  cg.setMaxIterations(maxit);
  cg.setTolerance(tol);
  cg.compute(A);
  x = cg.solve(b);
  std::cout <<" Eigen CG" << endl;
  std::cout << "#iterations:     " << cg.iterations() << endl;
  std::cout << "relative residual: " << cg.error()      << endl;
  std::cout << "absolute error: "<<(x-e).norm()<< endl;

  // with Gradient Method
  x=0*x; tol = 1.e-8;
  result = GRAD(A, x, b, D, maxit, tol);  // with Gradient Method
  std::cout <<" Gradient method " << endl;
  cout << "iterations performed: " << maxit << endl;
  cout << "relative residual  : " << tol << endl;
  std::cout << "absolute error: "<<(x-e).norm()<< endl;

  // Solve with GMRES method
  x=0*x; maxit = 1000; tol = 1.e-8;
  int restart = 200;
  result = GMRES(A, x, b, D, restart, maxit, tol);
  cout << " GMRES method " << endl;
  cout << "iterations performed: " << maxit << endl;
  cout << "relative residual  : " << tol << endl;
  cout << "absolute error:      " << (x-e).norm()<< endl;

  return result;
}




//vedere come sparseLU e SimplicialLDLT (cholesky) compared to cg

#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include "cg.hpp"

int main(int argc, char** argv)
{
  using namespace LinearAlgebra;
  // Some useful alias
  using SpMat=Eigen::SparseMatrix<double>;
  using SpVec=Eigen::VectorXd;

  int n = 100;
  SpMat A(n,n);                       // define Hilbert matrix
  for (int i=0; i<n; i++) {
      for (int j=0; j<n; j++){
          A.coeffRef(i, j) = 1.0/(i+j+1);
      }
  }

  double tol = 1.e-8;                  // Convergence tolerance
  int result, maxit = 1000;            // Maximum iterations for CG
  SpMat B = SpMat(A.transpose()) - A;  // Check symmetry
  std::cout<<"Norm of A-A.t: "<<B.norm()<<std::endl;
  // Create Rhs b
  SpVec e = SpVec::Ones(A.rows());
  SpVec b = A*e;
  SpVec x(A.rows());
  Eigen::DiagonalPreconditioner<double> D(A);

  // First with Eigen Choleski direct solver
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
  Eigen::SparseLU<Eigen::SparseMatrix<double> > solvelu;        // LU factorization
  solvelu.compute(A);
  if(solvelu.info()!=Eigen::Success) {                          // sanity check
      std::cout << "cannot factorize the matrix" << std::endl;  // decomposition failed
      return 0;
  }

  x = solvelu.solve(b);                                         // solving
  std::cout << "Solution with Eigen LU:" << std::endl;
  std::cout << "effective error: "<<(x-e).norm()<< std::endl;

  // Finally with hand-made CG
  x=0*x;
  result = CG(A, x, b, D, maxit, tol);                     // Call CG function
  std::cout << "Solution with Conjugate Gradient:" << std::endl;
  std::cout << "iterations performed: " << maxit << std::endl;
  std::cout << "tolerance achieved  : " << tol << std::endl;
  std::cout << "Error norm: "<<(x-e).norm()<< std::endl;

  return result;
}





//direct solvers vs iterative solver on block made matrix

#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include "cg.hpp"

int main(int argc, char** argv)
{
  using namespace LinearAlgebra;
  // Some useful alias
  using SpMat=Eigen::SparseMatrix<double>;
  using SpVec=Eigen::VectorXd;

  int n = 1000;
  SpMat A(n,n);                       // define Hilbert matrix
  for (int i=0; i<20; i++) {
      for (int j=0; j<20; j++){
          A.coeffRef(i, j) = 1.0/(i+j+1);
      }
  }
  for (int i=0; i<n; i++) {
      A.coeffRef(i, i) += 2.0;
      if (i < n-1) A.coeffRef(i, i+1) += -1.0;
      if (i > 0) A.coeffRef(i,i-1) += -1.0;
  }

  double tol = 1.e-8;                  // Convergence tolerance
  int result, maxit = 1000;            // Maximum iterations for CG
  SpMat B = SpMat(A.transpose()) - A;  // Check symmetry
  std::cout<<"Norm of A-A.t: "<<B.norm()<<std::endl;

  // Create Rhs b
  SpVec e = SpVec::Ones(A.rows());
  SpVec b = A*e;
  SpVec x(A.rows());
  Eigen::DiagonalPreconditioner<double> D(A);

  // First with Eigen Choleski direct solver
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;  // LDLT factorization
  solver.compute(A);
  if(solver.info()!=Eigen::Success) {                          // sanity check
      std::cout << "cannot factorize the matrix" << std::endl;
      return 0;
  }
  x = solver.solve(b);                                         // solving
  std::cout << "Solution with Eigen Choleski:" << std::endl;
  std::cout << "effective error: "<<(x-e).norm()<< std::endl;

  // Then with Eigen SparseLU solver
  Eigen::SparseLU<Eigen::SparseMatrix<double> > solvelu;        // LU factorization
  solvelu.compute(A);
  if(solvelu.info()!=Eigen::Success) {                          // sanity check
      std::cout << "cannot factorize the matrix" << std::endl;  // decomposition failed
      return 0;
  }

  x = solvelu.solve(b);                                         // solving
  std::cout << "Solution with Eigen LU:" << std::endl;
  std::cout << "effective error: "<<(x-e).norm()<< std::endl;

  // Now with hand-made CG
  x=0*x;
  result = CG(A, x, b, D, maxit, tol);              // Call CG function
  std::cout << "Solution with Conjugate Gradient:" << std::endl;
  std::cout << "iterations performed: " << maxit << std::endl;
  std::cout << "tolerance achieved  : " << tol << std::endl;
  std::cout << "Error norm: "<<(x-e).norm()<< std::endl;

  return result;
}



//how cholesky fail with non symmetric
//loading matrix froma a file sepified by the user

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

  double tol = 1.e-8;                  // Convergence tolerance
  int result, maxit = 2000;            // Maximum iterations for CGS
  SpMat B = SpMat(A.transpose()) - A;  // Check symmetry
  std::cout<<"Norm of A-A.t: "<<B.norm()<<std::endl;

  // Create Rhs b
  SpVec e = SpVec::Ones(A.rows());
  SpVec b = A*e;
  SpVec x(A.rows());
  Eigen::DiagonalPreconditioner<double> D(A);

  // First with Eigen Choleski direct solver
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;   // LDLT factorization
  solver.compute(A);
  if(solver.info()!=Eigen::Success) {                          // sanity check
      std::cout << "cannot factorize the matrix" << std::endl;
      return 0;
  }
  x = solver.solve(b);                                         // solving
  std::cout << "Solution with Eigen Choleski:" << std::endl;
  std::cout << "effective error: "<<(x-e).norm()<< std::endl;

  // Then with Eigen SparseLU solver
  Eigen::SparseLU<Eigen::SparseMatrix<double> > solvelu;        // LU factorization
  solvelu.compute(A);
  if(solvelu.info()!=Eigen::Success) {                          // sanity check
      std::cout << "cannot factorize the matrix" << std::endl;  // decomposition failed
      return 0;
  }

  x = solvelu.solve(b);                                         // solving
  std::cout << "Solution with Eigen LU:" << std::endl;
  std::cout << "effective error: "<<(x-e).norm()<< std::endl;

  // Now with hand-made CGS method
  x=0*x;
  // result = CG(A,x,b,D,maxit,tol);
  result = CGS(A, x, b, D, maxit, tol);          // Call CG - CGS function
  std::cout << "Solution with (Squared) Conjugate Gradient:" << std::endl;
  std::cout << "iterations performed: " << maxit << std::endl;
  std::cout << "tolerance achieved  : " << tol << std::endl;
  std::cout << "Error norm: "<<(x-e).norm()<< std::endl;

  return result;
}