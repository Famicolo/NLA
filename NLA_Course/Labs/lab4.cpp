//testing homemade conjugate gradient

#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>

using std::endl;
using std::cout;

#include "cg.hpp"                       // Include the template cg

int main(int argc, char** argv)
{
  using namespace LinearAlgebra;
  // Some useful alias
  using SpMat=Eigen::SparseMatrix<double>;
  using SpVec=Eigen::VectorXd;

  int n = 1000;
  SpMat A(n,n);                       // define matrix
  A.reserve(2998);
  for (int i=0; i<n; i++) {
      A.coeffRef(i, i) = 2.0*(i+1);
      if(i>0) A.coeffRef(i, i-1) = -i;
      if(i<n-1) A.coeffRef(i, i+1) = -(i+1);
  }
  double tol = 1.e-10;                // Convergence tolerance
  int result, maxit = 1000;           // Maximum iterations

  std::cout<<"Matrix size:"<<A.rows()<<"X"<<A.cols()<<endl;
  std::cout<<"Non zero entries:"<<A.nonZeros()<<endl;

  SpMat B = SpMat(A.transpose()) - A;  // Check symmetry
  std::cout<<"Norm of A-A.t: "<<B.norm()<<endl;

  // Create Rhs b
  SpVec e = SpVec::Ones(A.rows());
  SpVec b = A*e;
  SpVec x(A.rows());
  Eigen::DiagonalPreconditioner<double> D(A); // Create diagonal preconditioner

  // First with eigen CG
  Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper> cg;
  cg.setMaxIterations(maxit);
  cg.setTolerance(tol);
  cg.compute(A);
  x = cg.solve(b);
  std::cout <<" Eigen native CG"<< endl;
  std::cout << "#iterations:     " << cg.iterations() << endl;
  std::cout << "estimated error: " << cg.error()      << endl;
  std::cout << "effective error: "<<(x-e).norm()<< endl;

  // Now with hand-made CG
  x=0*x;
  result = CG(A, x, b, D, maxit, tol);        // Solve system

  std::cout <<" hand-made CG "<< endl;
  cout << "CG flag = " << result << endl;
  cout << "iterations performed: " << maxit << endl;

  return result;
}

//export the matrix and use lis

/**
 mpicc -DUSE_MPI -I${mkLisInc} -L${mkLisLib} -llis test1.c -o test1
 mpirun -n 4 ./test1 AtestCG.mtx 2 sol.mtx hist.txt -i cg 
mpirun -n 4 ./test1 AtestCG.mtx 2 sol.mtx hist.txt -i cg -p jacobi

mpirun -n 4 ./test1 AtestCG.mtx 2 sol.mtx hist.txt -i cg -p ssor 
mpirun -n 4 ./test1 AtestCG.mtx 2 sol.mtx hist.txt -i cg -p ssor -ssor_omega 0.8

mpirun -n 4 ./test1 AtestCG.mtx 2 sol.mtx hist.txt -i cg -p ilu 
./test1 AtestCG.mtx 2 sol.mtx hist.txt -i cg -p ilu
 **/




//testing jacobi homemade

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

  int n = 100;
  SpMat A(n,n);                      // define matrix
  A.reserve(298);
  for (int i=0; i<n; i++) {
      A.coeffRef(i, i) = 2.0;
      if(i>0) A.coeffRef(i, i-1) = -1.0;
      if(i<n-1) A.coeffRef(i, i+1) = -1.0;
  }

  double tol = 1.e-6;                // Convergence tolerance
  int result, maxit = 100;          // Maximum iterations

  std::cout<<"Matrix size:"<<A.rows()<<"X"<<A.cols()<<std::endl;
  std::cout<<"Non zero entries:"<<A.nonZeros()<<std::endl;
  SpVec e = SpVec::Ones(A.rows());
  SpVec b = A*e;
  SpVec x(A.rows());

  // Solve with CG method
  x=0*x;
  Eigen::DiagonalPreconditioner<double> D(A);// Create diagonal preconditioner
  result = CG(A, x, b, D, maxit, tol);
  cout << "CG   flag = " << result << endl;
  cout << "iterations performed: " << maxit << endl;
  cout << "tolerance achieved  : " << tol << endl;
  cout << "Error:                " << (x-e).norm()<< endl;

  // Solve with Jacobi method
  x=0*x; maxit = 20000; tol = 1.e-5;
  result = Jacobi(A, x, b, D, maxit, tol);
  cout << "Jacobi flag = " << result << endl;
  cout << "iterations performed: " << maxit << endl;
  cout << "tolerance achieved  : " << tol << endl;
  cout << "Error:                " << (x-e).norm()<< endl;

  return 0;
}




//testing non simmetric solvers
//cgs (A.t as preconditioner) gmres  bicgstab

// ... usual include here ...

#include "cgs.hpp"
#include "bcgstab.hpp"
#include "gmres.hpp"

int main(int argc, char** argv)
{
  using namespace LinearAlgebra;
  // Some useful alias
  using SpMat=Eigen::SparseMatrix<double>;
  using SpVec=Eigen::VectorXd;

  int n = 1000;
  double gam = -0.5;
  SpMat A(n,n);                      // define matrix
  A.reserve(2997);
  for (int i=0; i<n; i++) {
      A.coeffRef(i, i) = 2.0;
      if(i>1) A.coeffRef(i, i-2) = gam;
      if(i<n-1) A.coeffRef(i, i+1) = 1.0;
  }

  double tol = 1.e-8;                // Convergence tolerance
  int result, maxit = 100;           // Maximum iterations
  int restart = 30;                 // Restart for gmres

  std::cout<<"Matrix size:"<<A.rows()<<"X"<<A.cols()<<std::endl;
  std::cout<<"Non zero entries:"<<A.nonZeros()<<std::endl;
  SpVec e = SpVec::Ones(A.rows());
  SpVec b = A*e;
  SpVec x(A.rows());
  Eigen::LeastSquareDiagonalPreconditioner<double> SD(A);

  // Solve with CGS method
  x=0*x;
  result = CGS(A, x, b, SD, maxit, tol);
  cout << "CGS   flag = " << result << endl;
  cout << "iterations performed: " << maxit << endl;
  cout << "tolerance achieved  : " << tol << endl;
  cout << "Error:                " << (x-e).norm()<< endl;

  // Solve with BiCGSTAB method
  x=0*x; maxit = 100; tol = 1.e-8;
  Eigen::DiagonalPreconditioner<double> D(A);
  result = BiCGSTAB(A, x, b, D, maxit, tol);
  cout << "BiCGSTAB   flag = " << result << endl;
  cout << "iterations performed: " << maxit << endl;
  cout << "tolerance achieved  : " << tol << endl;
  cout << "Error:                " << (x-e).norm()<< endl;

  // Solve with GMRES method
  x=0*x; maxit = 100; tol = 1.e-8;
  result = GMRES(A, x, b, D, restart, maxit, tol);
  cout << "GMRES   flag = " << result << endl;
  cout << "iterations performed: " << maxit << endl;
  cout << "tolerance achieved  : " << tol << endl;
  cout << "Error:                " << (x-e).norm()<< endl;

  return 0;
}


//to compare use test 5 in lis for non symmetric matrix



//testing different preconditioner D(jacobi) ILU
// ... usual include modules
#include "cg.hpp"

int main(int argc, char** argv)
{
  using namespace LinearAlgebra;
  // Some useful alias
  using SpMat=Eigen::SparseMatrix<double>;
  using SpVec=Eigen::VectorXd;

  // Load matrix
  SpMat A;
  Eigen::loadMarket(A, "bcsstm12.mtx");
  A = SpMat(A.transpose()) + A;

  double tol = 1.e-14;                 // Convergence tolerance
  int result, maxit = 5000;            // Maximum iterations

  std::cout<<"Matrix size:"<<A.rows()<<"X"<<A.cols()<<std::endl;
  std::cout<<"Non zero entries:"<<A.nonZeros()<<std::endl;

  // Create Rhs b
  SpVec e = SpVec::Ones(A.rows());
  SpVec b = A*e;
  SpVec x(A.rows());

  // Create preconditioners
  Eigen::DiagonalPreconditioner<double> D(A);
  Eigen::IncompleteLUT<double> ILU(A);

  // Hand-made CG with diagonal precond
  x = 0*x;
  result = CG(A, x, b, D, maxit, tol);        // Solve system

  std::cout <<" hand-made CG "<<std::endl;
  cout << "iterations performed: " << maxit << endl;
  cout << "tolerance achieved  : " << tol << endl;
  std::cout << "Error norm: "<<(x-e).norm()<<std::endl;

    // Hand-made CG with ILU precond
  x = 0*x;
  result = CG(A, x, b, ILU, maxit, tol);
  std::cout <<" hand-made CG "<< std::endl;
  cout << "iterations performed: " << maxit << endl;
  cout << "tolerance achieved  : " << tol << endl;
  std::cout << "Error norm: "<<(x-e).norm()<<std::endl;

  return result;
}


//testing preconditioner in lis

mpicc -DUSE_MPI -I${mkLisInc} -L${mkLisLib} -llis test1.c -o test1

mpirun -n 4 ./test1 bcsstm12.mtx 2 sol.txt hist.txt -i bicgstab -maxiter 5000 -tol 1e-12

mpirun -n 4 ./test1 bcsstm12.mtx 2 sol.txt hist.txt -i bicgstab -tol 1e-12 -p ilu

mpirun -n 4 ./test1 bcsstm12.mtx 2 sol.txt hist.txt -i bicgstab -tol 1e-12 -p ilu -ilu_fill 2

