#include <dirent.h>
#include <eigen3/Eigen/Dense>
#include <sys/types.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


using namespace Eigen;

// returns list of files in directory (excluding hidden files)
void getFilesInDirectory(std::vector<std::string> & out,
                         const std::string & name) {
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;

    while ((dp = readdir(dirp)) != NULL) {
        if (dp->d_name[0] != '.')
            out.push_back(dp->d_name);
    }

    closedir(dirp);
}

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
    for (int i = 0; i < M; ++i) {
        std::string file = "./basePictures/subject"+std::to_string(i+1)+".pgm";
        VectorXd flatPic = load_pgm(file);
        faces.col(i) = flatPic;
        meanFace += flatPic;
    }

    // compute mean face and subtract it from all faces
    meanFace /= M;
    faces.colwise() -= meanFace;

    // compute eigenveoctors of AAT
    JacobiSVD<MatrixXd> svd(faces, ComputeThinU | ComputeThinV);
    MatrixXd U = svd.matrixU();

    // compute projection on the space spanned by eigenfaces
    std::vector<std::string> files;
    getFilesInDirectory(files, "./testPictures");

    for (std::string file : files) {
        VectorXd newFace = load_pgm("./testPictures/" + file);
        MatrixXd projFaces = U.transpose() * faces;
        VectorXd projNewFace = U.transpose() * (newFace - meanFace);

        // compute minimal distance between projections of newface and faces
        int indexMinNorm;
        VectorXd diff = (projFaces.colwise() - projNewFace).colwise().norm();
        diff.minCoeff(&indexMinNorm);

        std::cout << std::endl
                  << std::setw(22) << file
                  << std::setw(27) << " is identified as subject "
                  << std::setw(3) << 1 + indexMinNorm
                  << std::endl;
    }
    std::cout << std::endl;

    return 0;
}

