#include <iostream>
#include <iomanip>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */

   VectorXd rmse(4);
   rmse << 0, 0, 0, 0;

   // Checking for:
   // The estimation vector size is not zero
   // The estimation vector size is equal to the ground truth vector size

   if(estimations.size() != ground_truth.size() || estimations.size() == 0) {
      std::cout << "Invalid estimation or ground truth data" << std::endl;
      return rmse;
   }

   // Calculating squared residuals
  if (estimations.size() == ground_truth.size()) {
      for (unsigned int i=0; i < estimations.size(); ++i) {
        VectorXd residual = estimations[i] - ground_truth[i];
        residual = residual.array()*residual.array();
        rmse += residual;
      }
  }else{
    std::cout << "Size of estimation data and ground truth data is different." << std::endl;
  }

   // Calculating the mean
   rmse = rmse / estimations.size();

   // Calculating the squared root
   rmse = rmse.array().sqrt();

   std::cout << "RMSE = [";
   for ( unsigned int i = 0; i <= rmse.size() - 1; i++){
         std::cout << std::fixed << std::setprecision(4) << std::setw(7) << rmse[i];
   }  
   std::cout << "]" << std::endl << std::endl;

   return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * TODO:
   * Calculate a Jacobian here.
   */
   MatrixXd Hj(3, 4);

   // Recovering state parameters
   float px = x_state(0);
   float py = x_state(1);
   float vx = x_state(2);
   float vy = x_state(3);

   // Calculating a group of variables to avoid repeting their calculation

   float c1 = pow(px,2) + pow(py, 2);
   float c2 = sqrt(c1);
   float c3 = (c1*c2);

   // Checking for division by zero
   if (fabs(c1) < 0.0000001){
      std::cout << "Tools::CalculateJacovian(). Error: Division by zero" << std::endl;
      return Hj;
   }

    Hj << (px/c2), (py/c2), 0.0, 0.0,
          -(py/c1), (px/c1), 0.0, 0.0,
          py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

   return Hj;

}
