#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>

using std::endl;
using std::cout;

#include "cg.hpp"
#include "jacobi.hpp"
#include "cgs.hpp"
#include "bcgstab.hpp"
#include "gmres.hpp"

int main(int argc, char** argv)
{
  using namespace LinearAlgebra;
  // Some useful alias  ...

  using SpMat = Eigen::SparseMatrix<double>;
  using SpVec = Eigen::VectorXd;
  using namespace LinearAlgebra;

  

  int n = 1000;
  double gam = -1;  //cambiando gamma la convergenza va amignotte ma GMRS funziona meglio
  // a -0.5 funziona, a -1.5 non funziona, a 1.1 non converge nulla, a -0.9 mettendo restart a 40 torno a convergere
  //CGS funziona bene con buoni condizionamenti...altrimenti funziona peggio
//a gam -1 non converge nulla ma vedendo gli errori CGS è quello che funziona peggio
  
  SpMat A(n,n);                      // define matrix
  A.reserve(2997);
  for (int i=0; i<n; i++) {
      A.coeffRef(i,i) = 2.0;
      if(i>1) A.coeffRef(i,i-2) = gam;
      if(i<n-1) A.coeffRef(i,i+1) = 1;
  }

  double tol = 1.e-8;                // Convergence tolerance
  int result, maxit = 100;           // Maximum iterations
  int restart = 100;                  // Restart for gmres
  //restatr ferma il loop, prende la soluzione trovata e la usa come initial guess del
  //nuovo sistema per evitare che il krilog space diventi troppo grande
  //in genere è un numero tra 20 e 30, chiaro che se le mie iterazioni sono meno del restart..non restarto niente !!


  std::cout<<"Matrix size:"<<A.rows()<<"X"<<A.cols()<<std::endl;
  std::cout<<"Non zero entries:"<<A.nonZeros()<<std::endl;
  SpVec e = SpVec::Ones(A.rows());
  SpVec b = A*e;
  SpVec x(A.rows());
  Eigen::LeastSquareDiagonalPreconditioner<double> SD(A);

  // Solve with CGS method
  x=0*x;
  result = CGS(A, x, b, SD, maxit, tol);
  cout << "CGS flag = " << result << endl;
  cout << "iterations performed: " << maxit << endl;
  cout << "tolerance achived: " << tol <<endl;
  cout << "effective error: " << (x-e).norm() << endl;
  cout << endl;

  // Solve with BiCGSTAB method

  x=0*x;
  //fai attenzione a ridefinire iterazioni e risultati perchè sono passati by reference e vengono cambiati nel codice !!
  Eigen::DiagonalPreconditioner<double> D(A);
  maxit = 100; tol = 1.e-8;
  result = BiCGSTAB(A, x, b, D, maxit, tol);
  cout << "BiCGSTAB flag = " << result << endl;
  cout << "iterations performed: " << maxit << endl;
  cout << "tolerance achived: " << tol <<endl;
  cout << "effective error: " << (x-e).norm() << endl;
  cout << endl;
  

  // Solve with GMRES method

  x=0*x; maxit = 100; tol = 1.e-8;
  result = GMRES(A, x, b, D, restart, maxit, tol);
  cout << "GMRES flag = " << result << endl;
  cout << "iterations performed: " << maxit << endl;
  cout << "tolerance achived: " << tol <<endl;
  cout << "effective error: " << (x-e).norm() << endl;
  cout << endl;
  

  return 0;
}