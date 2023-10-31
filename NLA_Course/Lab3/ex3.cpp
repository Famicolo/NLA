#include <iostream>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

using namespace std;
using SpMat=Eigen::SparseMatrix<double>;
using SpVec=Eigen::VectorXd;
using namespace Eigen;

int main(int argc, char** argv)
{
SpMat mat, mat2;
loadMarket(mat, "Asym.mtx");
string test = "false";
mat2 = SpMat(mat.transpose())-mat;

if(mat2.nonZeros()==0)test = "true";
cout << "matrix size " << mat.rows() << "x" << mat.cols() << endl;
cout <<  mat2.norm() << endl;  //se è vicina a 0 è perchè è una matrice di zeri
//cout <<  mat.nonZeros() << endl;  //non può dare la soluzione esatta per colpa delle approssimazioni

//costruisco il vettore soluzione
SpVec xe = SpVec::Constant(mat.rows(),1);
//define right hand side b
SpVec x(mat.rows()), b = mat*xe;

//cout << "b = " << b <<endl;

//solver parameters
double tol = 1.e-12;
int maxiter = 1000;

//Solver
ConjugateGradient<SpMat, Lower|Upper> conGrad;
conGrad.setTolerance(tol);
conGrad.setMaxIterations(maxiter);
conGrad.compute(mat);
x = conGrad.solve(b);

cout << "solution with cojugate gradient" <<endl;
cout << "iterations " << conGrad.iterations() << endl; 
cout << "relative residual: " << conGrad.error() << endl;
cout << "effective error: " << (xe-x).norm() << endl;
//cout << x <<endl;

//ex 3.2
//define matrix A
SpMat A(50,50);
for (int i = 1; i < 49; i++)
{
    A.insert(i,i) = 2.;
    A.insert(i,i+1) = -1.;
    A.insert(i,i-1) = -1.;
}
A.insert(0,0) = 2.;
A.insert(0,1) = -1.;
A.insert(49,49) = 2.;
A.insert(49,48) = -1.;


//define exact sol and b vector
SpVec x2(A.rows()), xe2 = SpVec::Constant(A.rows(),1), b2(A.rows());
b2 = A*xe2;

//define a new solver for different matrix
ConjugateGradient<SpMat, Lower|Upper> cg;
cg.setMaxIterations(maxiter);
cg.setTolerance(tol);
cg.compute(A);
x2 = cg.solve(b2);

//cout << A <<endl;
cout << "non zero entries A: " << A.nonZeros() <<endl;
cout << "solution with cojugate gradient" <<endl;
cout << "iterations " << cg.iterations() << endl; 
cout << "relative residual: " << cg.error() << endl;
cout << "effective error: " << (xe2-x2).norm() << endl;
cout <<b2<<endl;
//cout << x2 << endl;

//salvo le matrici
saveMarket(A, "ALab3Ex3.mtx");
//saveMarketVector(b2, "b2Lab3Ex3.mtx"); non salva il formato corretto

// Eigen::saveMarketVector(b, "./rhs.mtx");
    FILE* out = fopen("bLab3Ex3.mtx","w");
    fprintf(out,"%%%%MatrixMarket vector coordinate real general\n");
    fprintf(out,"%d\n", 50);
    for (int i=0; i<50; i++) {
        fprintf(out,"%d %f\n", i ,b2(i));
    }
    fclose(out);




}