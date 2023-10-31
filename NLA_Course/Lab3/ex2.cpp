#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
int main()
{
    int size = 100;
 MatrixXd A = MatrixXd::Zero(size,size);  //matrice di soli zeri
 for (int i = 1; i < size-1; i++)
 {
    A(i,i)=2;
    A(i,i-1)=1;
    A(i,i+1)=-1;
 }
 A(0,0)=2;
 A(0,1)=-1;
 A(size-1,size-1)=2;
 A(size-1,size-2)=1;

 double normA=A.norm();

 MatrixXd As = A/2 + A.transpose()/2; 
 double normAs = As.norm();

 VectorXd v(50), v2(50), result(50), temp(50);
 v = VectorXd::Constant(50,1);
 for (int i = 0; i < 50; i++)
 {
    v2(i)=1;
 }

result = A.bottomRightCorner(50,50)*v;
temp = A.topLeftCorner(1,50).transpose();
int scalarProduct =  temp.dot(v);

 
 //std::cout << A <<std::endl;
 //std::cout << As <<std::endl;
 std::cout << normA <<std::endl;
 std::cout << normAs <<std::endl;  
 //std::cout << result <<std::endl; 
 std::cout << scalarProduct <<std::endl;
}
