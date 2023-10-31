#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/SparseExtra>
#include "bcgstab.hpp"

using namespace std;

// Some useful alias
using SpMat=Eigen::SparseMatrix<double>;
using SpVec=Eigen::VectorXd;

int main(int argc, char** argv){

    SpMat A;
    Eigen::loadMarket(A, "AEigPb.mtx");
    std::cout << "Matrix size:"<< A.rows() << "X" << A.cols() << endl;
    std::cout << "Norm of A: :" << A.norm() << endl;
    SpMat B = SpMat(A.transpose()) - A;  // Check symmetry
    std::cout << "Norm of skew-symmetric part: " << B.norm() << endl;

    Eigen::MatrixXd A2;
    A2 = Eigen::MatrixXd(A);  
   Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(A2);  
   if (eigensolver.info() != Eigen::Success) abort();
   std::cout << "The eigenvalues of A are:\n" << eigensolver.eigenvalues() << std::endl;
   
    return 0;    



}