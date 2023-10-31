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