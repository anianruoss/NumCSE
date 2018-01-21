#include <iostream>
#include <complex>
#include <cmath>


//! \brief Compute complex root
//! \param[in] w complex number with non negative imaginary part
//! \return the square root of w with non negative real and imaginary parts
std::complex<double> myroot( std::complex<double> w ) {
    double x,y;
    double u = w.real();
    double v = w.imag();

    // TODO: problem 1c: construct x and y as functions of u and v

    return std::complex<double> (x,y);
}

// Test the implementation
int main() {
    std::cout << "*** PROBLEM 1, testing:" << std::endl;

    std::complex<double> w(1e20,5);
    std::cout << "The square root of " << w << " is " << myroot(w) << std::endl;
    std::cout << "The correct square root of " << w << " is " << sqrt(w) << std::endl << std::endl;

    w=std::complex<double>(-5,1e20);
    std::cout << "The square root of " << w << " is " << myroot(w) << std::endl;
    std::cout << "The correct square root of " << w << " is " << sqrt(w) << std::endl << std::endl;

	return 0;
}

