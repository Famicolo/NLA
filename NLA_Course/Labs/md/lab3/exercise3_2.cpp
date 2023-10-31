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
