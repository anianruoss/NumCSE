#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <eigen3/Eigen/Dense>


using namespace Eigen;

// returns a picture as a flattened vector
VectorXd load_pgm(const std::string &filename) {
    int row = 0, col = 0, rows = 0, cols = 0;

    std::ifstream infile(filename);
    std::stringstream ss;
    std::string inputLine = "";

    // First line : version
    std::getline(infile,inputLine);

    // Second line : comment
    std::getline(infile,inputLine);

    // Continue with a stringstream
    ss << infile.rdbuf();
    // Third line : size
    ss >> cols >> rows;

    VectorXd picture(rows*cols);

    // Following lines : data
    for (row = 0; row < rows; ++row) {
        for (col = 0; col < cols; ++col) {
            int val;
            ss >> val;
            picture(col*rows + row) = val;
        }
    }

    infile.close();

    return picture;
}


int main() {
    int h = 231;
    int w = 195;
    int M = 15;

    MatrixXd faces(h*w, M);
    VectorXd meanFace(h*w);

    // loads pictures as flattened vectors into faces
    std::cout << "--> Loading Images" << std::endl;
    for (int i = 0; i < M; ++i) {
        std::string file = "./basePictures/subject" + std::to_string(i+1) + ".pgm";
        VectorXd flatPic = load_pgm(file);
        faces.col(i) = flatPic;
        meanFace += flatPic;
    }

    // compute mean face and subtract it from all faces
    std::cout << "--> Computing Mean Face" << std::endl;
    meanFace /= M;
    faces.colwise() -= meanFace;

    // compute eigenveoctors of AAT
    std::cout << "--> Computing Eigenvectors" << std::endl;
    JacobiSVD<MatrixXd> svd(faces, ComputeThinU | ComputeThinV);
    MatrixXd U = svd.matrixU();

    // compute projection on the space spanned by eigenfaces
    std::cout << "--> Computing Projection" << std::endl;
    std::string testPicName = "./testPictures/subject01.happy.pgm";
    VectorXd newFace = load_pgm(testPicName);

    MatrixXd projFaces = U.transpose() * faces;
    VectorXd projNewFace = U.transpose() * (newFace - meanFace);

    // compute minimal distance between projections of newface and faces
    int indexMinNorm;
    (projFaces.colwise()-projNewFace).colwise().norm().minCoeff(&indexMinNorm);
    std::cout << testPicName << " is identified as subject "
              << 1 + indexMinNorm << std::endl;

    return 0;
}

