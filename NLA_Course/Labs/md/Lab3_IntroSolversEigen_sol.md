# 1. Numerical linear algebra with Eigen

Read the documentation available at
- [Eigen user guide](https://eigen.tuxfamily.org/dox/GettingStarted.html).

Eigen is a high-level C++ library of template headers for linear algebra, matrix and vector operations, geometrical transformations, numerical solvers and related algorithms.

### 1.1 A simple example

In the following example we declare a 3-by-3 (dynamic) matrix m which is initialized using the 'Random()' method with random values between -1 and 1. The next line applies a linear mapping such that the values are between 0 and 20. 

The next line of the main function introduces a new type: 'VectorXd'. This represents a (column) vector of arbitrary size. Here, the vector is created to contain 3 coefficients which are left uninitialized. The one but last line uses the so-called comma-initializer and the final line of the program multiplies the matrix m with the vector v and outputs the result.

To access the coefficient $A_{i,j}$ of the matrix $A$, use the syntax `A(i,j)`. Similarly, for accessing the component $v_{i}$ of a vector $\boldsymbol{v}$ 
use `v(i)`

```
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
```

1. Using VS Code, open the shared folder and create a file `eigen-test1.cpp` with the content of the previous example

2. Change the current directory to the shared folder `cd /home/jellyfish/shared-folder`.

3. In the container, make sure the Eigen module is loaded by typing: `module list`.

4. Compile and run the test.
```
g++ -I ${mkEigenInc} eigen-test1.cpp -o test1
./test-installation
```

### 1.2 Block operations on matrices and vectors

Eigen provides a set of block operations designed specifically for the special case of vectors:
- Block containing the first $n$ elements: `vector.head(n)`
- Block containing the first $n$ elements: `vector.tail(n)`
- Block containing $n$ elements, starting at position $i$: `vector.segment(i,n)`

Eigen also provides special methods for blocks that are flushed against one of the corners or sides of a matrix. For instance:
- Top-left p by q block: `matrix.topLeftCorner(p,q)`
- Bottom-left p by q block: `matrix.bottomLeftCorner(p,q)`
- Top-right p by q block: `matrix.topRightCorner(p,q)`
- Bottom-right p by q block: `matrix.bottomRightCorner(p,q)`

Individual columns and rows are special cases of blocks. Eigen provides methods to easily address them (`.col()` and `.row()`). The argument is the index of the column or row to be accessed. As always in Eigen, indices start at 0.

1. Example - compile and run the following code

```
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
```

2. Exercise: 

- In Eigen, construct the $100\times 100$ matrix $\tilde{A}$ be defined such that
$$ 
\tilde{A} = \begin{pmatrix}
    2 & -1 & 0 & 0&\ldots & 0  \\
    1 & 2 & -1 & 0& \ldots & 0  \\
    0 & 1 & \ddots  & \ddots &\ldots  & \vdots \\
    0 & 0 & \ddots  & \ddots  & \ddots & 0 \\
   \vdots& \vdots &  \vdots &\ddots &\ddots  & -1\\
    0 & 0  &\ldots & 0& 1   & 2
\end{pmatrix}.
$$ 

- Display the Euclidean norm of $\tilde{A}$ denoted by $||\tilde{A}||$. Display also $||\tilde{A}_S||$ where $\tilde{A}_S$ is the symmetric part of $\tilde{A}$, namely $2\tilde{A}_S = \tilde{A} + \tilde{A}^{T}$

- Declare a vector $\tilde{v}$ of length 50 with all the entries equal to $1$

- Compute the matrix-vector product of the $50\times50$ bottom right block of $\tilde{A}$ times the vector $\tilde{v}$ and display the result.

- Compute the scalar product (`.dot()`) between the vector $\tilde{v}$ and the vector obtained by taking the first 50 entries of the first row of $\tilde{A}$.

### Solution
```
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
```


### 2. Linear systems 

We aim to solve a linear system of equations `Ax = b`, where `A` and `b` are matrices (b could be a vector, as a special case). In Eigen we can choose between various decompositions, depending on the properties of your matrix `A`, and depending on the desired speed and accuracy. 

In this example, the `colPivHouseholderQr()` method returns an object of class ColPivHouseholderQR, which is a QR decomposition with column pivoting.

```
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
```

A table of some other decomposition methods available in Eigen is reported [here](https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html). 
If you know more about the properties of your matrix, you can use the above table to select the best method. For example, a good choice for solving linear systems with a non-symmetric matrix of full rank is `PartialPivLU`. If you know that your matrix is also symmetric and positive definite, the above table says that a very good choice is the LLT or LDLT decomposition. 

### 2.1 Compute the error

Only you know what error margin you want to allow for a solution to be considered valid. So Eigen lets you do this computation for yourself, if you want to, as in this example:

```
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
```

### 2.2 Sparse linear systems

The class `SparseMatrix` is the main sparse matrix representation of Eigen's sparse module; it offers high performance and low memory usage. It implements a more versatile variant of the widely-used Compressed Column (or Row) Storage scheme.

The `SparseMatrix` and `SparseVector` classes take three template arguments: the scalar type (e.g., double) the storage order (ColMajor or RowMajor, the default is ColMajor) the inner index type (default is int). As for dense Matrix objects, constructors takes the size of the object. Here are some examples:

```
SparseMatrix<std::complex<float> > mat(1000,2000);   
// declares a 1000x2000 column-major compressed sparse matrix of complex<float>

SparseMatrix<double,RowMajor> mat(1000,2000);              
// declares a 1000x2000 row-major compressed sparse matrix of double

SparseVector<std::complex<float> > vec(1000);              
// declares a column sparse vector of complex<float> of size 1000

SparseVector<double,RowMajor> vec(1000);                   
// declares a row sparse vector of double of size 1000
```

In Eigen, there are several methods available to solve linear systems when the coefficient matrix is sparse. Because of the special representation of this class of matrices, special care should be taken in order to get a good performance. [This page](https://eigen.tuxfamily.org/dox/group__TopicSparseSystems.html) lists the sparse solvers available in Eigen. All the available solvers follow the same general concept.

```
#include <Eigen/RequiredModuleName>
// ...
SparseMatrix<double> A;
// fill A
VectorXd b, x;
// fill b
// solve Ax = b
SolverClassName<SparseMatrix<double> > solver;
solver.compute(A);
if(solver.info()!=Success) {
  // decomposition failed
  return;
}
x = solver.solve(b);
if(solver.info()!=Success) {
  // solving failed
  return;
}
// solve for another right hand side:
x1 = solver.solve(b1);
```

A simple example:

```
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
```

N.B. Block operations works also for sparse matrix, but it is recommended not to use for WRITING or MODIFYING existing matrices. We will only use these operations to extract blocks.

### 2.3 Load and export sparse matrices

To export your matrices and right-hand-side vectors in the matrix-market format, we can use the `unsupported SparseExtra` module.

```
#include <unsupported/Eigen/SparseExtra>
...
Eigen::saveMarket(A, "filename.mtx");
Eigen::saveMarketVector(B, "filename_b.mtx");
```

To load a matrix in the matrix market format, follow the instructions below:

- in the terminal, use `wget` to download a matrix from the matrix market, e.g. `wget https://math.nist.gov/pub/MatrixMarket2/NEP/mhd/mhd416a.mtx.gz`.

- unzip the file by typing `gzip -dk mhd416a.mtx.gz`

- in Eigen, include the `unsupported SparseExtra` module and use 

```
SparseMatrix<double> mat;
loadMarket(mat, "mhd416a.mtx");
```

To export a matrix in the matrix market format, follow the instructions below:

```
std::string matrixFileOut("./matrixName.mtx");
Eigen::saveMarket(mat, matrixFileOut);
```

### Exercise

Compile and test the following example 

```
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
```

## 3. The CG Sparse iterative solver of Eigen

Eigen provides a built-in `Eigen::ConjugateGradient` solver. This class allows to solve for $A\boldsymbol{x} = \boldsymbol{b}$ linear problems using an iterative conjugate gradient algorithm. The matrix A must be selfadjoint. The matrix A and the vectors x and b can be either dense or sparse.

This class follows the sparse solver concept and has the following inputs:
- `MatrixType_`	the type of the matrix $A$, can be a dense or a sparse matrix.
- `UpLo_` the triangular part that will be used for the computations. It can be Lower, Upper, or Lower|Upper in which the full matrix entries will be considered. 
- `Preconditioner_` the type of the preconditioner. Default is DiagonalPreconditioner

The maximal number of iterations and tolerance value can be controlled via the setMaxIterations() and setTolerance() methods. The defaults are the size of the problem for the maximal number of iterations and NumTraits<Scalar>::epsilon() for the tolerance.

The tolerance corresponds to the relative residual error: 
$$
tol = |A\boldsymbol{x}- \boldsymbol{b}|/|\boldsymbol{b}|
$$

N.B. Even though the default value of `UpLo_` is `Lower`, significantly higher performance is achieved when using a complete matrix and `Lower|Upper` as the `UpLo_` template parameter.

### 3.1 Exercise

- Download the matrix `Asym.mtx` from webeep Lab3 folder and move it to the working directory.
- Display the size of the matrix and check if it is symmetric. 
- Take as exact solution a vector `xe` defined as in the previous example and compute the right-hand side `b`. 
- Solve the resulting linear system using the Conjugate Gradient (CG) solver available in Eigen. 
- Compute and display the relative error between the exact solution `xe` and the approximated solution.

```
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
    std::cout << "relative residual: " << cg.error()      << endl;
    std::cout << "effective error: " << (x-e).norm() << endl;

    return 0;    
}
```

### 3.2 Exercise

- In Eigen, construct the $50\times 50$ symmetric matrix $A$ defined such that
$$ 
A = \begin{pmatrix}
    2 & -1 & 0 & 0&\ldots & 0  \\
    -1 & 2 & -1 & 0& \ldots & 0  \\
    0 & -1 & \ddots  & \ddots &\ldots  & \vdots \\
    0 & 0 & \ddots  & \ddots  & \ddots & 0 \\
   \vdots& \vdots &  \vdots &\ddots &\ddots  & -1\\
    0 & 0  &\ldots & 0& -1   & 2
\end{pmatrix}.
$$
- Display the number of nonzero entries and check if it is symmetric. 
- Take as exact solution a vector `xe` defined as in the previous example and compute the right-hand side `b`. 
- Solve the resulting linear system using the Conjugate Gradient (CG) solver available in Eigen. 
- Compute and display the relative error between the exact solution `xe` and the approximated solution.
- Export matrix $A$ and the right-hand side vector $\boldsymbol{b}$ in the `.mtx` format. Move the files in the test folder of the LIS library and repeat the previous exercise using LIS.

### Solution

```
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
```

### HOMEWORK
Repeat the previous exercise using the BiCGSTAB solver of Eigen.