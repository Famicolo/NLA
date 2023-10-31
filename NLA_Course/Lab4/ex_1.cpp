#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/SparseExtra>

using std::endl;
using std::cout;

#include "cg.hpp"                       // Include the template cg

int main(int argc, char** argv)
{
  using namespace LinearAlgebra;
  // Some useful alias
  using SpMat=Eigen::SparseMatrix<double>;
  using SpVec=Eigen::VectorXd;
  using namespace Eigen;

  int n = 10;
  SpMat A(n,n);                       // define matrix
  //A.reserve(2998);
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
  double temp;
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
  x=0*x; tol = 1.e-10; maxit = 1000;
  result = CG(A, x, b, D, maxit, tol);        // Solve system

  std::cout <<" hand-made CG "<< endl;
  cout << "CG flag = " << result << endl;
  cout << "iterations performed: " << maxit << endl;
  cout << "tolerance achived" << tol <<endl;
  cout << "effective error" << (x-e).norm() << endl;
  cout << "x_cg: " << x <<endl;
  cout<<endl;
  //cout << "Matrice A" << A <<endl;


  //implemento un metodo di Gauss-siedel
  x=0*x; tol = 1.e-10; maxit = 100;
  SpVec r = b - A * x;
  double normb = b.norm(), resid;
  int result2 = 2;

  
  int maxit2 = maxit;

  for (int i = 1; i <= maxit2; i++)
  {
    for (int ri = 0; ri < D.rows(); ri++)
  {
    temp = b.coeffRef(ri);
    for (int co = 0; co < D.cols(); co++)
    {
        if(ri != co) temp = temp - A.coeffRef(ri,co)*x.coeffRef(co);
    }
    x.coeffRef(ri) = temp/A.coeffRef(ri,ri);
  }
   if((resid = r.norm() / normb) <= tol)
        {
          tol = resid;
          maxit2 = i;
          result2 = 0;
        }
  }

  cout << "risultato: " << result2 << endl;
  cout << "iterazioni " << maxit2 << endl;
  cout << "x: " << x <<endl;
  


  


  

  //salvo la matrice A
//saveMarket(A, "ALab4Ex1.mtx");
//questa parte non serve dal momento che in Lis posso automaticamente creare un vettore b
//costruito dalla soluzione esatta [1,1,1...1] usando il parametro "2" al posto del vettore
/**FILE* out = fopen("bLab4Ex1.mtx","w");
    fprintf(out,"%%%%MatrixMarket vector coordinate real general\n");
    fprintf(out,"%d\n", n);
    for (int i=0; i<n; i++) {
        fprintf(out,"%d %f\n", i ,b(i));
    }
    fclose(out);**/

    return result;
}

