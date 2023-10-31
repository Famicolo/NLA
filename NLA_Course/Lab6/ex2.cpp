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