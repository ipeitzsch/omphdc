#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <time.h>  
#include <sstream>

using namespace std;

typedef vector<vector<int8_t>> ByteMatrix;
typedef vector<int> IntVector;
typedef vector<vector<int>> IntMatrix;
typedef vector<int8_t> ByteVector;
typedef vector<double> DoubleVector;


struct Compare { float val; size_t index; };    
#pragma omp declare reduction(maximum : struct Compare : omp_out = omp_in.val >= omp_out.val ? omp_in : omp_out)


IntMatrix genRandomData(int numData, int feats) {
    IntMatrix ret(numData);
    for(int i = 0; i < numData; i++) {
        IntVector v(feats);
        for(int j = 0; j < feats; j++) {
            v[j] = rand() % 256;
        }
        ret[i] = v;
    }
    return ret;
}

ByteMatrix genBases(int dim, int size) {
    ByteMatrix ret(size);
    for(int i = 0; i < size; i++) {
        ByteVector v(dim);
        for(int j = 0; j < dim; j++) {
            if(rand() % 2) {
                v[j] = 1;
            }
            else {
                v[j] = -1;
            }
        }
        ret[i] = v;
    }
    return ret;
}

ByteMatrix genClasses(int dim, int numClasses) {
    ByteMatrix ret(numClasses);
    for(int i = 0; i < numClasses; i++) {
        ByteVector v(0, dim);
        ret[i] = v;
    }
    return ret;
}


int serialInfer(ByteMatrix bases, ByteMatrix classes, IntVector input, int dim) {
    ByteVector H(dim);
    for(int i = 0; i < input.size(); i++) {
        int val = input[i];
        ByteVector temp(bases[val]);
        rotate(temp.begin(), temp.begin()+temp.size()-i, temp.end());
        for(int j = 0; j < dim; j++) {
            H[j] += temp[j];
        }
    }
    for(int j = 0; j < dim; j++) {
        H[j] = H[j] < 0 ? -1 : 1;
    }

    int max = 0;
    int maxInd = -1;
    for(int i = 0; i < classes.size(); i++) {
        int sum = 0;
        for(int j = 0; j < classes[i].size(); j++) {
            sum += H[j] * classes[i][j];
        }
        cout << "SERIAL " << i << " " << sum << endl;
        if(sum >= max) {
            maxInd = i;
            max = sum;
        }
    }
    return maxInd;
}


int ompInfer(ByteMatrix bases, ByteMatrix classes, IntVector input, int dim) {
    ByteVector H(dim);
    #pragma omp parallel for shared(bases, classes, input, dim, H)
    for(int j = 0; j < dim; j++) {
        for(int i = 0; i < input.size(); i++) {
            int val = input[i];
            H[j] += bases[val][(dim + j - i) % dim];
        }
        H[j] = H[j] < 0 ? -1 : 1;
    }
    struct Compare max;
    max.val = 0;
    max.index = 0;
    #pragma omp parallel for shared(bases, classes, input, dim, H) reduction(maximum:max)
    for(int i = 0; i < classes.size(); i++) {
        int sum = 0;
        for(int j = 0; j < classes[i].size(); j++) {
            sum += H[j] * classes[i][j];
        }
        std::ostringstream ss;
        ss << "OMP " << i << " " << sum << endl;
        std::string str = ss.str();
        cout << str;
        if(sum >= max.val) {
            max.index = i;
            max.val = sum;
        }
    }

    return max.index;
}
#define S_SIZE 16
int ompInfer2(ByteMatrix bases, ByteMatrix classes, IntVector input, int dim) {
    int sums[S_SIZE] = {0};
    #pragma omp parallel for shared(bases, classes, input, dim) reduction(+:sums[:S_SIZE])
    for(int j = 0; j < dim; j++) {
        int temp = 0;
        for(int i = 0; i < input.size(); i++) {
            int val = input[i];
            temp += bases[val][(dim + j - i) % dim];
        }
        int8_t encoded = temp < 0 ? -1 : 1;
        for(int i = 0; i < classes.size(); i++) {
            sums[i] += classes[i][j] * encoded;
        }
    }

    int maxInd = 0;
    int max = 0;
    for(int i = 0; i < classes.size(); i++) {
        if(sums[i] >= max) {
            max = sums[i];
            maxInd = i;
        }
    }
    return maxInd;
}

int main() {
    srand (time(NULL));
    ByteMatrix bases = genBases(10000, 256);
    // ByteMatrix classes = genClasses(10000, 16);
    ByteMatrix classes = genBases(10000, 16); // just to make them random
    IntMatrix data = genRandomData(100, 784);
    int count = 1;
    double serialAvg = 0.0;
    double ompAvg = 0.0;
    double omp2Avg = 0.0;
    for(IntVector datum : data) {
        //datum = IntVector({106, 99, 50, 10, 80, 211, 29, 51, 175, 100, 204, 89, 70, 184, 127, 217, 149, 82, 23, 48, 234, 10, 10, 167, 180, 53, 110, 201, 154, 55, 76, 5, 155, 126, 15, 235, 81, 44, 30, 0, 145, 234, 90, 215, 162, 217, 176, 55, 43, 200, 103, 21, 210, 114, 188, 135, 167, 42, 80, 65, 98, 156, 70, 253, 26, 86, 232, 107, 130, 6, 108, 19, 240, 198, 234, 147, 159, 155, 202, 202, 99, 50, 223, 53, 164, 155, 188, 75, 197, 13, 140, 39, 169, 211, 36, 196, 41, 12, 47, 171, 18, 155, 191, 3, 97, 169, 150, 0, 68, 96, 202, 167, 146, 169, 221, 54, 68, 153, 129, 10, 166, 14, 49, 80, 225, 86, 20, 10, 98, 67, 181, 117, 223, 116, 120, 64, 30, 14, 65, 98, 110, 11, 10, 1, 181, 231, 55, 249, 128, 185, 3, 39, 199, 53, 119, 168, 139, 139, 178, 237, 206, 103, 98, 173, 220, 218, 238, 250, 232, 47, 92, 87, 58, 102, 88, 239, 77, 143, 233, 206, 72, 236, 245, 15, 33, 108, 183, 172, 247, 105, 154, 197, 209, 252, 115, 173, 215, 97, 167, 191, 144, 3, 22, 202, 106, 110, 186, 183, 254, 163, 133, 70, 143, 122, 86, 177, 230, 13, 93, 221, 119, 247, 163, 72, 244, 22, 245, 203, 119, 156, 138, 7, 159, 161, 209, 9, 15, 139, 193, 13, 46, 70, 84, 190, 193, 170, 111, 167, 183, 204, 133, 46, 196, 40, 118, 184, 62, 107, 131, 181, 7, 13, 188, 167, 174, 141, 176, 190, 25, 113, 203, 71, 184, 31, 5, 121, 201, 116, 32, 129, 65, 165, 175, 5, 205, 38, 189, 11, 145, 64, 192, 153, 77, 124, 64, 252, 10, 240, 186, 35, 98, 133, 106, 26, 165, 112, 147, 110, 228, 179, 239, 37, 89, 159, 42, 38, 197, 231, 50, 86, 39, 242, 239, 117, 111, 47, 113, 121, 32, 43, 156, 130, 176, 6, 156, 85, 118, 47, 196, 91, 226, 179, 128, 59, 82, 171, 98, 23, 146, 148, 110, 186, 134, 93, 47, 245, 141, 160, 110, 173, 203, 10, 47, 123, 17, 203, 209, 135, 250, 149, 226, 220, 72, 99, 24, 155, 14, 122, 178, 160, 14, 32, 90, 148, 126, 137, 138, 11, 41, 248, 184, 244, 3, 231, 112, 20, 178, 65, 155, 172, 214, 126, 136, 30, 225, 160, 185, 239, 26, 108, 143, 40, 140, 234, 189, 10, 115, 71, 21, 157, 63, 205, 145, 66, 180, 1, 86, 102, 66, 242, 18, 24, 112, 155, 55, 81, 59, 240, 64, 86, 92, 207, 126, 233, 185, 59, 243, 45, 130, 9, 202, 194, 214, 91, 4, 139, 93, 91, 241, 159, 77, 4, 184, 189, 159, 239, 14, 218, 223, 78, 48, 60, 29, 175, 37, 215, 234, 24, 4, 109, 33, 206, 47, 248, 41, 51, 131, 134, 142, 116, 38, 219, 120, 222, 152, 23, 205, 166, 242, 172, 244, 34, 232, 18, 209, 13, 233, 188, 38, 237, 41, 71, 187, 88, 63, 228, 139, 194, 107, 26, 55, 145, 245, 175, 111, 142, 199, 60, 52, 185, 232, 41, 219, 209, 59, 173, 222, 36, 105, 4, 17, 146, 76, 204, 234, 139, 176, 117, 78, 27, 143, 133, 172, 133, 52, 27, 19, 251, 87, 71, 180, 64, 112, 144, 17, 171, 61, 239, 207, 166, 244, 224, 56, 64, 172, 34, 203, 93, 151, 25, 120, 39, 158, 37, 172, 211, 64, 191, 206, 152, 6, 131, 216, 119, 19, 233, 34, 80, 216, 242, 246, 204, 210, 46, 12, 127, 80, 216, 220, 231, 241, 84, 14, 144, 121, 186, 99, 186, 121, 49, 82, 128, 180, 42, 247, 199, 19, 25, 23, 235, 11, 13, 184, 222, 59, 196, 93, 139, 156, 57, 115, 142, 141, 129, 30, 7, 60, 129, 193, 181, 178, 19, 53, 103, 61, 44, 46, 80, 70, 70, 59, 81, 83, 243, 47, 143, 184, 140, 26, 84, 197, 141, 226, 83, 15, 0, 90, 75, 129, 27, 0, 52, 46, 54, 155, 107, 98, 201, 187, 168, 15, 246, 250, 99, 234, 41, 242, 162, 182, 12, 246, 123, 154, 217, 206, 169, 217, 40, 244, 91, 67, 244, 143, 113, 42, 42, 220, 141, 243, 151, 53, 3, 142, 47, 102, 120, 89, 88, 26, 15, 100, 16, 138, 254, 233, 89, 167, 195, 129, 155, 30, 197, 144, 173, 54, 186, 215, 19, 71, 202, 170, 125, 205, 56, 172, 51, 176, 5, 139, 202, 20, 240, 219, 159, 238, 196, 248, 150, 135, 121, 49, 165, 62, 193, 82, 117, 124, 41, 136, 195, 244, 50, 64, 193});
        double start = omp_get_wtime();
        int serialAnswer = serialInfer(bases, classes, datum, 10000);
        double end = omp_get_wtime();
        serialAvg += (end - start);
         start = omp_get_wtime();
        int ompAnswer = ompInfer(bases, classes, datum, 10000);
         end = omp_get_wtime();
        ompAvg += (end - start);
        start = omp_get_wtime();
        int omp2Answer = ompInfer2(bases, classes, datum, 10000);
         end = omp_get_wtime();
        omp2Avg += (end - start);
        if(serialAnswer != ompAnswer) {
            cout << "Error with vector: " << count << " OMP: " << ompAnswer << " SERIAL: " << serialAnswer << endl;
            for(int d : datum) {
                cout << d << ", ";
            }
            cout << endl << endl;
        }
        count++;
    }
    serialAvg = serialAvg / data.size();
    ompAvg = ompAvg / data.size();
    omp2Avg = omp2Avg / data.size();
    cout << "Serial average: " << serialAvg << endl;
    cout << "OMP average: " << ompAvg << endl;
    cout << "OMP 2 average: " << omp2Avg << endl;
    #pragma omp parallel 
    { 
        if(omp_get_thread_num() == 0) {
            cout << omp_get_num_threads() << endl;
        }
    }
    return 0;
}
