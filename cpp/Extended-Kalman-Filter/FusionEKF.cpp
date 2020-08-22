#include <iostream>
#include <iomanip>
#include "FusionEKF.h"
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;
using std::setw;
using std::setprecision;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
   * TODO: Finish initializing the FusionEKF.
   * TODO: Set the process and measurement noises
   */
   H_laser_ << 1.0, 0, 0, 0,
               0, 1.0, 0, 0;

   Hj_ << 1.0, 1.0, 0, 0,
          1.0, 1.0, 0, 0,
          1.0, 1.0, 1.0, 1.0;

   //Matrix's F initial transition
   ekf_.F_ = MatrixXd(4, 4);
   ekf_.F_ << 1.0, 0, 1.0, 0,
              0, 1.0, 0, 1.0,
              0, 0, 1.0, 0,
              0, 0, 0, 1.0;
  
  //Matrix's P state covariance
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1.0, 0, 0, 0,
            0, 1.0, 0, 0,
            0, 0, 1000.0, 0,
            0, 0, 0, 1000.0;

  noise_ax = 9.0;
  noise_ay = 9.0;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * TODO: Initialize the state ekf_.x_ with the first measurement.
     * TODO: Create the covariance matrix.
     * You'll need to convert radar from polar to cartesian coordinates.
     */

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1.0, 1.0, 1.0, 1.0;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Converting data radar from polar to cartesian coordinates 
      //         and initialize state.
      float ro = measurement_pack.raw_measurements_(0);
      float phi = measurement_pack.raw_measurements_(1);
      float ro_dot = measurement_pack.raw_measurements_(2);

      // Converting from polar to cartesian
			double px = ro*cos(phi);
      double py = ro*sin(phi);
      double vx = ro_dot*cos(phi);
      double vy = ro_dot*sin(phi);

      ekf_.x_ << px, py, vx, vy;

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initializing state.
			double px = measurement_pack.raw_measurements_(0);
      double py = measurement_pack.raw_measurements_(1);
      double vx = 0.0;
      double vy = 0.0;

      ekf_.x_ << px, py, vx, vy;
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  /**
   * TODO: Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   * TODO: Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  float dt_2 = pow(dt, 2);
  float dt_3 = pow(dt, 3);
  float dt_4 = pow(dt, 4);

  //Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  //set the process covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << (dt_4*noise_ax/4), 0.0, (dt_3*noise_ax/2), 0.0,
            0.0, (dt_4*noise_ay/4), 0.0, (dt_3*noise_ay/2),
            (dt_3*noise_ax/2), 0.0, (dt_2*noise_ax), 0.0,
            0.0, (dt_3*noise_ay/2), 0.0, dt_2*noise_ay;

  ekf_.Predict();

  /**
   * Update
   */

  /**
   * TODO:
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    Tools tools;
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    H_laser_ << 1.0, 0.0, 0.0, 0.0,
    		       0.0, 1.0, 0.0, 0.0;
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // Printing the output X_
  std::cout << "x_ = [";
  for ( unsigned int i = 0; i <= ekf_.x_.size() - 1; i++){
      std::cout << std::fixed << std::setprecision(2) << std::setw(7) << ekf_.x_[i];
  }  
  std::cout << "]" << std::endl << std::endl;

 // Pringint the output P_
  std::cout << "P_ =";
  for (unsigned int i = 0; i <= 3; i++){
    std::cout << std::setw(1) << "";
    for (unsigned int j = 0; j <= 3; j++) {
      std::cout << std::fixed << std::setprecision(2) << std::setw(6) << ekf_.P_ (i, j);
    }
    std::cout << std::endl;
    std::cout << std::setw(4) << "";
  }
  std::cout << std::endl;

}
