#include<iostream>
#include<algorithm>
#include<iomanip>
#include<vector>
#include<math.h>
using namespace std;

// 矩阵乘法
template<typename T>
vector<vector<T>> matrix_multiply(vector<vector<T>> const arrL, vector<vector<T>> const arrR)
{
    int rowL = arrL.size();// 左矩阵行数
    int colL = arrL[0].size();// 左矩阵列数
    int rowR = arrR.size();// 右矩阵行数
    int colR = arrR[0].size();// 右矩阵列数
    // 判断是否能够相乘
    if(colL != rowR)
    {
        throw "left matrix's row not should equal with right matrix!";
    }
    // initialize result matrix 
    vector<vector<T>> res(rowL);
    for(int i=0; i<res.size();i++){
        res[i].resize(colR);
    }
    // compute matrix multiplication
    for(int i=0; i<rowL; i++){
        for(int j=0; j<colR; j++){
            for(int k=0; k<colL; k++){
                res[i][j] += arrL[i][k]*arrR[k][j];
            }
        }
    }

    return res;

}
//打印矩阵
template<typename T>
void display_matrix(vector<vector<T>>  const arr)
{
    for(int i = 0; i<arr.size(); i++){
        for(int j=0; j<arr[0].size(); j++){
            cout << setw(10)<<setprecision(4)<<arr[i][j];
        }
        cout <<'\n';
    }
}
//打印向量
template<typename T>
void display_vector(vector<T>  const arr)
{
    for(int i = 0; i<arr.size(); i++){
        cout << setw(8) << setprecision(5) << arr[i];
    }
    cout <<'\n';
}
// 矩阵转置
template<typename T>
vector<vector<T>> transpose(vector<vector<T>> const arr)
{
    int row = arr.size();
    int col = arr[0].size();
    // initialize transpose matrix col*row
    vector<vector<T>> trans(col);
    for(int i=0;i<col;i++){
        trans[i].resize(row);
    }
    // fill elements
    for(int i=0; i<col;i++){
        for(int j=0;j<row;j++){
            trans[i][j] = arr[j][i];
        }
    }
    return trans;
}
// 实现argsort功能
// 返回排序好的下标
template<typename T> 
vector<int> argsort(const vector<T>& array)
{
	const int array_len(array.size());
	vector<int> array_index(array_len, 0);
	for (int i = 0; i < array_len; ++i)
		array_index[i] = i;

	sort(array_index.begin(), array_index.end(),
		[&array](int pos1, int pos2) {return (array[pos1] > array[pos2]);});//>从大到小

	return array_index;
}

// 实对称矩阵特征值特征向量
// param: arr   :input array
// param: E     :eigen vectors
// param: e     :eigen values
template<typename T>
void eigen(vector<vector<T>> arr, vector<vector<T>> &E, vector<T> &e)
{
    //vector<vector<T>> arr = arr_ori;
    int n = arr.size();// size of matrix
    int row = 0;// row index of max
    int col = 0;// col index of max
    int iter_max_num = 10000;//迭代总次数
    int iter_num = 0;
    double eps = 1e-40;//误差
    double max = eps;// 非对角元素最大值
    // 初始化特征向量矩阵为单位阵,初始化特征值
    E.resize(n);
    e.resize(n);
    for(int i=0; i<n; i++){
        E[i].resize(n,0);
        E[i][i] = 1;
    }

    while(iter_num<iter_max_num && max>=eps)
    {
        max = fabs(arr[0][1]);
        row = 0;
        col = 1;
        // find max value and index
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                if(i!=j && fabs(arr[i][j])>max){
                    max = fabs(arr[i][j]);
                    row = i;
                    col = j;
                }
            }
        }
        double theta = 0.5*atan2(2 * arr[row][col] , (arr[row][row] - arr[col][col]));
        //update arr
        double aii = arr[row][row];
        double ajj = arr[col][col];
        double aij = arr[row][col];
        double sin_theta = sin(theta);
        double cos_theta = cos(theta);
        double sin_2theta = sin(2 * theta);
        double cos_2theta = cos(2 * theta);
        arr[row][row] = aii*cos_theta*cos_theta + ajj*sin_theta*sin_theta + aij*sin_2theta;//Sii'
        arr[col][col] = aii*sin_theta*sin_theta + ajj*cos_theta*cos_theta - aij*sin_2theta;//Sjj'
        arr[row][col] = 0.5*(ajj - aii)*sin_2theta + aij*cos_2theta;//Sij'
        arr[col][row] = arr[row][col];//Sji'
        for (int k = 0; k < n; k++)
        {
            if (k != row && k != col)
            {
                double arowk = arr[row][k];
                double acolk = arr[col][k];
                arr[row][k] = arowk * cos_theta + acolk * sin_theta;
                arr[k][row] = arr[row][k];
                arr[col][k] = acolk * cos_theta - arowk * sin_theta;
                arr[k][col] = arr[col][k];
            }
        }
        // update E
        double Eki;
        double Ekj;
        for(int k=0; k<n; k++){
            Eki = E[k][row];
            Ekj = E[k][col];
            E[k][row] = Eki*cos_theta + Ekj*sin_theta;
            E[k][col] = Ekj*cos_theta - Eki*sin_theta;
        }
        iter_num++;
    }
    //update e
    for(int i=0;i<n;i++){
        e[i] = arr[i][i];
    }
    // sort E by e
    vector<int> sort_index;
    sort_index = argsort(e);
    // initialize E_sorted, e_sorted
    vector<vector<T>> E_sorted(n);
    for(int i=0;i<n;i++){
        E_sorted[i].resize(n);
    }
    vector<T> e_sorted(n);
    for(int i=0;i<n;i++){
        e_sorted[i] = e[sort_index[i]];
        for(int j=0;j<n;j++){
            E_sorted[i][j] = E[i][sort_index[j]];
        }
    }
    E = E_sorted;
    e = e_sorted;
    //delete &E_sorted, &e_sorted;
    cout<<"Maximum off-diagonal element is: "<<max<<", iterate: "<<iter_num<<" times"<<endl;

}
//*****************************************************//
//########################SVD##########################
//*****************************************************//
//params:
//      --arr   :input matrix m*n
//      --U     :left matrix m*r , r <= rank(arr)
//      --S     :medium matrix r*r 
//      --V     :right matrix n*r
class SVD
{
    public:
        vector<vector<double>> U,S,V,ATA,A;
        int n,m,r;
        vector<vector<double>> E; //特征向量矩阵
        vector<double> e; // 特征值向量
        SVD(vector<vector<double>> arr);
        void tight_svd();//紧奇异值分解
        void truncated_svd(int);//截断奇异值分解
    
};
SVD::SVD(vector<vector<double>> arr){
    m = arr.size();
    n = arr[0].size();
    A = arr;
    ATA = matrix_multiply(transpose(A),A);
    // 计算ATA特征值特征向量
    eigen(ATA,E,e);
}
void SVD::tight_svd(){
    r = 0;
    // 确定秩
    for(int i=0;i<e.size();i++){
        if(e[i]>1e-10){
            r++;
        }
        else break;
    }
    //确定V
    V = E;
    for(int i=0; i<n;i++){
        V[i].resize(r);
    }
    //确定S
    S.resize(r);
    for(int i=0;i<r;i++){
        S[i].resize(r);
        S[i][i] = sqrt(e[i]);
    }
    //确定U
    vector<vector<double>> Sinv = S;
    for(int i=0;i<r;i++){
        Sinv[i][i] = 1/S[i][i];
    }
    U = matrix_multiply(matrix_multiply(A,V),Sinv);
}
void SVD::truncated_svd(int rr){
    r = rr;
    //确定V
    V = E;
    for(int i=0; i<n;i++){
        V[i].resize(r);
    }
    //确定S
    S.resize(r);
    for(int i=0;i<r;i++){
        S[i].resize(r);
        S[i][i] = sqrt(e[i]);
    }
    //确定U
    vector<vector<double>> Sinv = S;
    for(int i=0;i<r;i++){
        Sinv[i][i] = 1/S[i][i];
    }
    U = matrix_multiply(matrix_multiply(A,V),Sinv);
}

int main()
{
    vector<vector<double>> A = {{1,0,0,0},{0,0,0,4},{0,3,0,0},{0,0,0,0},{2,0,0,0}};
    SVD svd(A);
    // tight SVD
    svd.tight_svd();
    cout<<endl;
    cout<<"matrix A:"<<endl;
    display_matrix(A);
    cout<<endl;
    cout<<"matrix U:"<<endl;
    display_matrix(svd.U);
    cout<<endl;
    cout<<"matrix Sigma:"<<endl;
    display_matrix(svd.S);
    cout<<endl;
    cout<<"matrix V':"<<endl;
    display_matrix(transpose(svd.V));
    //display_matrix(matrix_multiply(matrix_multiply(svd.U,svd.S),transpose(svd.V)));//合成A
}