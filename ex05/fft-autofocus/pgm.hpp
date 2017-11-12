#pragma once

#include <iostream>

#include <Eigen/Dense>

class PGMObject {
public:
    PGMObject() { }

    PGMObject(const Eigen::MatrixXd & data) {
        set_data(data);
    }

    PGMObject & set_data(const Eigen::MatrixXd & mat) {
       width = mat.cols();
       height = mat.rows();
       double M = mat.maxCoeff();
       if(M > _upperBoundVal) {
           std::cerr << "Maximum value to high!"
                     << std::endl;
           return *this;
       }
       for(unsigned int p = 2; p <= _upperBoundVal; p *= 2) {
           if(p > M) {
               maxVal = p - 1;
               if(maxVal > 256) is_long = true;
               else is_long = false;
               break;
           }
       }
       std::size_t size = width * height * (is_long ? 2 : 1);
       data = new unsigned char[size];
       for(unsigned int i = 0; i < height; ++i) {
           for(unsigned int j = 0; j < width; ++j) {
               if( is_long ) {
                   ((unsigned short int*) data )[i*width + j] = mat(i,j);
               } else {
                   ((unsigned char*) data)[i*width + j] = mat(i,j);
               }
           }
       }
           return *this;
    }

    const PGMObject & get_data(Eigen::MatrixXd & mat) const {
        mat.resize(width, height);
        if( is_long ) {
            if(sizeof(unsigned short int) != 2) {
                std::cerr << "Internal error, invalid "
                          << "size of 'short unsigned int'."
                          << std::endl;
                return *this;
            }
            mat = Eigen::Map<
                    Eigen::Matrix<unsigned short int,
                    Eigen::Dynamic, Eigen::Dynamic,
                    Eigen::RowMajor>
                    >((unsigned short int*) data, height, width)
                    .cast<double>();
        } else {
            mat = Eigen::Map<
                    Eigen::Matrix<unsigned char,
                    Eigen::Dynamic, Eigen::Dynamic,
                    Eigen::RowMajor>
                    >(data, height, width)
                    .cast<double>();
        }
        return *this;
    }

    friend
    std::istream& operator>>(std::istream &i,
                             PGMObject &obj)
    {
        if( !i.good() ) {
            std::cerr << "Input stream in bad state (missing file?)!"
                      << std::endl;
            return i;
        }
        unsigned int stage = 0;
        unsigned int line = 1;
        while (stage != 3 ||
               i.peek() == std::char_traits<char>::to_int_type('#'))
        {
            if (i.peek() != std::char_traits<char>::to_int_type('#') )
            {
                std::string mn;
                switch(stage) {
                   case  0:
//                    std::cout << "Reading magic number on line = " << line
//                              << "."
//                              << std::endl;
                      if(!std::getline(i, mn)) {
                          std::cerr << "Bad, missing or corrupt "
                                    << "file at line '"
                                    << mn
                                    << "' in input stream!"
                                    << std::endl;
                      }
                      if(obj._magic_number != mn) {
                          std::cerr << "Bad magic number '" << mn
                                    << "' in input stream!"
                                    << std::endl;
                          return i;
                      }
                    break;
                   case 1:
//                    std::cout << "Reading image size on line = " << line
//                              << "."
//                              << std::endl;
                      i >> obj.width
                        >> obj.height;
                    break;
                   case 2:
//                    std::cout << "Reading max val on line = " << line
//                              << "."
//                              << std::endl;
                    i >> obj.maxVal;
                       if(obj.maxVal > obj._upperBoundVal) {
                           std::cerr << "Invalid max val!"
                                     << std::endl;
                           return i;
                       } if(obj.maxVal >= 256) {
                           obj.is_long = true;
                       } else {
                           obj.is_long = false;
                       }
                    break;
                }
                stage++;
            } else {
//                std::cout << "Skipping comment on line = " << line
//                          << "."
//                          << std::endl;
                std::string comment;
                std::getline(i, comment);
//                std::cout << comment << std::endl;
            }
            line++;
            if(line > obj._maxLines) {
                std::cerr << "Reading too many lines, line = " << line
                          << "."
                          << std::endl;
                return i;
            }
        }
//        std::cout << "Reading data on line = " << line
//                  << "."
//                  << std::endl;
        std::size_t size = obj.width * obj.height * (obj.is_long ? 2 : 1);
        obj.data = new unsigned char[size];
        i.read((char*) obj.data, size);

//        std::cout << "Succesfully loaded "
//                  << obj.width << "x" << obj.height
//                  << " PGM image with max. value "
//                  << obj.maxVal << std::endl;

        return i;
    }

    friend
    std::ostream& operator<<(std::ostream &o,
                             const PGMObject &obj)
    {
        if( !o.good() ) {
            std::cerr << "Output in bad state (cannot write to disk?)!"
                      << std::endl;
            return o;
        }
        o << obj._magic_number << std::endl
          << "# Created with PGMObject class." << std::endl
          << obj.width << " "
          << obj.height << std::endl
          << obj.maxVal << std::endl;
        std::size_t size = obj.width * obj.height * (obj.is_long ? 2 : 1);
        o.write((char*) obj.data, size);

//        std::cout << "Succesfully written "
//                  << obj.width << "x" << obj.height
//                  << " PGM image with max. value "
//                  << obj.maxVal << std::endl;

        return o;
    }

private:
    const std::string _magic_number = "P5";
    const unsigned int _upperBoundVal = 65536;
    const unsigned int _maxLines = 40;

    size_t width, height;

    unsigned int maxVal = 256;
    bool is_long =  false;

    unsigned char* data;
};
