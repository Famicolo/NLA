//iterative solvers with eigen

//example
#include <iostream>
#include <Eigen/Dense>
 
using Eigen::MatrixXd;
using Eigen::VectorXd;
 
int main()
{
  MatrixXd m = MatrixXd::Random(3,3);
  m = (m + MatrixXd::Constant(3,3,1.0)) * 10;
  std::cout << "m =" << std::endl << m << std::endl;
  VectorXd v(3);
  v << 1, 0, 0;
  std::cout << "m * v =" << std::endl << m * v << std::endl;
}

//compiler
//g++ -I ${mkEigenInc} eigen-test1.cpp -o test1

//using block

#include <Eigen/Dense>
#include <iostream>
 
using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
 
int main()
{
  VectorXd v(6);
  v << 1, 2, 3, 4, 5, 6;
  cout << "v.head(3) =" << endl << v.head(3) << endl << endl;
  cout << "v.tail<3>() = " << endl << v.tail<3>() << endl << endl;
  v.segment(1,4) *= 2;
  cout << "after 'v.segment(1,4) *= 2', v =" << endl << v << endl;

  MatrixXd A = MatrixXd::Random(9,9);
  MatrixXd B = A.topLeftCorner(3,6);
  VectorXd w = B*v;
  cout << "norm of B*v = " << w.norm() << endl;
}


//ex construct matrix norm and simmetric part of A

#include <Eigen/Dense>
#include <iostream>
 
using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
 
int main()
{
  int n = 100;
  MatrixXd A = MatrixXd::Zeros(n,n);
    for (int i=0; i<n; i++) {
        A(i, i) = 2.0;
        if(i>0) A(i, i-1) = 1.0;
        if(i<n-1) A(i, i+1) = -1.0;
    }

  VectorXd v = VectorXd::Constant(A.rows(), 1);     // define vector
  cout << "matrix vector multiplication =" << endl << A.topLeftCorner(50,50)*v << endl;

  cout << "norm of A = " << A.norm() << endl;
  cout << "norm of symmetric part " << (A.transpose() + A).norm() << endl;
  cout << "dot product " << v.dot((A.row(0)).head(50)) << endl;
}


//Solution with QR decomposition

#include <iostream>
#include <Eigen/Dense>
 
int main()
{
   Eigen::Matrix3f A;
   Eigen::Vector3f b;
   A << 1,2,3,  4,5,6,  7,8,10;
   b << 3, 3, 4;
   std::cout << "Here is the matrix A:\n" << A << std::endl;
   std::cout << "Here is the vector b:\n" << b << std::endl;
   Eigen::Vector3f x = A.colPivHouseholderQr().solve(b);
   std::cout << "The solution is:\n" << x << std::endl;
}


//solution with lu computing the error

#include <iostream>
#include <Eigen/Dense>
 
using Eigen::MatrixXd;
 
int main()
{
   MatrixXd A = MatrixXd::Random(100,100);
   MatrixXd b = MatrixXd::Random(100,50);
   MatrixXd x = A.fullPivLu().solve(b);
   double relative_error = (A*x - b).norm() / b.norm(); // norm() is L2 norm
   std::cout << "The relative error is:\n" << relative_error << std::endl;
}


//definizione per matrici sparse

SparseMatrix<std::complex<float> > mat(1000,2000);   
// declares a 1000x2000 column-major compressed sparse matrix of complex<float>

SparseMatrix<double,RowMajor> mat(1000,2000);              
// declares a 1000x2000 row-major compressed sparse matrix of double

SparseVector<std::complex<float> > vec(1000);              
// declares a column sparse vector of complex<float> of size 1000

SparseVector<double,RowMajor> vec(1000);                   
// declares a row sparse vector of double of size 1000

//esempio soluzione per matrici sparse

#include <iostream>
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv)
{
    SparseMatrix<double> mat(10,10);                // define matrix
    for (int i=0; i<10; i++) {
        mat.coeffRef(i, i) = 1.0;
    }

    VectorXd b = VectorXd::Constant(mat.rows(), 1); // define right-hand side

    // Solving 
    SimplicialLDLT<Eigen::SparseMatrix<double> > solver(mat);   // factorization 
    solver.compute(mat);
    if(solver.info()!=Success) {                                // sanity check 
        cout << "cannot factorize the matrix" << endl;          
        return 0;
    }
    
    VectorXd x = solver.solve(b);                   // solving
    cout << x << endl;                              // display solution
    return 0;    
}


//save matricies
#include <unsupported/Eigen/SparseExtra>
...
Eigen::saveMarket(A, "filename.mtx");
Eigen::saveMarketVector(B, "filename_b.mtx");


//load matrix
SparseMatrix<double> mat;
loadMarket(mat, "mhd416a.mtx");

//to export matricies in matrix market format
std::string matrixFileOut("./matrixName.mtx");
Eigen::saveMarket(mat, matrixFileOut);








//example in loading and exporting matrix
#include <iostream>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv)
{
    // Load matrix
    SparseMatrix<double> mat;
    loadMarket(mat, "mhd416a.mtx");

    VectorXd xe = VectorXd::Constant(mat.rows(), 1);      // define exact solution
    VectorXd b = mat*xe;                               // compute right-hand side
    cout << b << endl;

    return 0;    
}



//solving with conjugate gradient in eigen
#include <iostream>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

using namespace std;

// Some useful alias
using SpMat=Eigen::SparseMatrix<double>;
using SpVec=Eigen::VectorXd;

int main(int argc, char** argv)
{
    // Load matrix
    SpMat mat;
    Eigen::loadMarket(mat, "Asym.mtx");
    
    // Check matrix properties
    std::cout << "Matrix size:"<< mat.rows() << "X" << mat.cols() << endl;
    std::cout << "Non zero entries:" << mat.nonZeros() << endl;
    SpMat B = SpMat(mat.transpose()) - mat;  // Check symmetry
    std::cout << "Norm of skew-symmetric part: " << B.norm() << endl;

    // Create Rhs b
    SpVec e = SpVec::Ones(mat.rows());    // Define exact solution
    SpVec b = mat*e;                      // Compute rhs
    SpVec x(mat.rows());

    // Set parameters for solver
    double tol = 1.e-8;                 // Convergence tolerance
    int result, maxit = 1000;           // Maximum iterations
    Eigen::DiagonalPreconditioner<double> D(mat); // Create diag preconditioner

    // Solving 
    Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper> cg;
    cg.setMaxIterations(maxit);
    cg.setTolerance(tol);
    cg.compute(mat);
    x = cg.solve(b);
    std::cout << " Eigen native CG" << endl;
    std::cout << "#iterations:     " << cg.iterations() << endl;
    std::cout << "relative residual: " << cg.error()      << endl; // |Ax-b|/|b|
    std::cout << "effective error: " << (x-e).norm() << endl;

    return 0;    
}




//ex buld matrix
//display non zero
//solve with cg
//relative error
//export matrix

#include <iostream>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv)
{
    int n = 50;	
    SparseMatrix<double> mat(n,n);                           // define matrix
    for (int i=0; i<n; i++) {
        mat.coeffRef(i, i) = 2.0;
	if(i>0) mat.coeffRef(i, i-1) = -1.0;
        if(i<n-1) mat.coeffRef(i, i+1) = -1.0;	
    }

    VectorXd xe = VectorXd::Constant(mat.rows(), 1);         // define sol
    VectorXd b = mat*xe;                                     // compute rhs

    // Solving 
    ConjugateGradient<SparseMatrix<double>, Lower|Upper> solver;
    solver.compute(mat);
    VectorXd x = solver.solve(b);
    std::cout << "#iterations:     " << solver.iterations() << std::endl;

    double relative_error = (x-xe).norm()/(xe).norm();       // compute err 
    cout << relative_error << endl;

    // Export matrix and rhs
    std::string matrixFileOut("./Alapl50.mtx");
    Eigen::saveMarket(mat, matrixFileOut);

    // Eigen::saveMarketVector(b, "./rhs.mtx");
    FILE* out = fopen("rhs.mtx","w");
    fprintf(out,"%%%%MatrixMarket vector coordinate real general\n");
    fprintf(out,"%d\n", n);
    for (int i=0; i<n; i++) {
        fprintf(out,"%d %f\n", i ,b(i));
    }
    fclose(out);

    return 0;    
}