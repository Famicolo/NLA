#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

using namespace std;
using SpMat=Eigen::SparseMatrix<double>;
using namespace Eigen;

int main(int argc, char** argv)
{
  // Load matrix
  SpMat mat;
  loadMarket(mat, "nos1.mtx");
  // Check matrix properties
  cout << "Matrix A is a: " << mat.rows() << " x "<< mat.cols() << endl;
  cout << "Norm A: " << mat.norm() <<endl;
  cout << "non zero entries in A: " << mat.nonZeros() << endl;

  // Compute Eigenvalues of the original matrix
  // Eigensolver uses dense matrix format!!!
  MatrixXd A;
  A = MatrixXd(mat);    
  EigenSolver<MatrixXd> eigensolver(A);  //funziona per le simmetriche
   if (eigensolver.info() != Eigen::Success) abort();
   std::cout << "The eigenvalues of A are:\n" << eigensolver.eigenvalues() << std::endl;
   // std::cout << "Here's a matrix whose columns are eigenvectors of A \n"
   //           << eigensolver.eigenvectors() << std::endl;

  // Compute Eigenvalues of symmetric matrix
  MatrixXd Asym = (MatrixXd(A.transpose())+A)/2;
  MatrixXd B = MatrixXd(Asym.transpose())-Asym;
  cout << "check for symmetry of the A+At matrix: " << B.norm() << endl;
  SelfAdjointEigenSolver<MatrixXd> eigensolverSym(Asym);  //funziona per le simmetriche
   if (eigensolverSym.info() != Eigen::Success) abort();
   std::cout << "The eigenvalues of A are:\n" << eigensolverSym.eigenvalues() << std::endl;
   // std::cout << "Here's a matrix whose columns are eigenvectors of A \n"
   //           << eigensolver.eigenvectors() << std::endl;
  return 0;
}