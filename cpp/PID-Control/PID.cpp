#include <iostream>

#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
    Kp_ = Kp;
    Ki_ = Ki;
    Kd_ = Kd;
    p_error = 0.0;
    i_error = 0.0;
    d_error = 0.0;
}

void PID::UpdateError(double cte) {
    d_error = cte - p_error;
    p_error = cte;
    i_error += cte;
}

double PID::TotalError() {
    double totalError = (-Kp_ * p_error) - (Ki_ * i_error) - (Kd_ * d_error);

    std::cout << "Total Error: " << totalError << std::endl;

    if (totalError < -1 || totalError > 1) {
      totalError = totalError * 0.5;
      std::cout << "**Total Error: " << totalError << std::endl;
    }

    return totalError;
}



