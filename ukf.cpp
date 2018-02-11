#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <fstream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/4.0;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

 
  //Init
  is_initialized_ = false;
  x_ << 0, 0, 0, 0, 0;

  //State covariance matrix
  P_ << 0.1, 0,   0,   0,         0,
        0,   0.1, 0,   0,         0,
        0,   0,   100, 0,         0,
        0,   0,   0,   M_PI*M_PI, 0,
        0,   0,   0,   0,         M_PI*M_PI;
  
  //Init
  Xsig_pred_ = MatrixXd(5, 15);
  Xsig_pred_.fill(0.0);
  long long time_us_ = 0;

  //Transformation parameters
  weights_ = VectorXd(15);
  n_x_     = 5;
  n_aug_   = 7;
  lambda_  = 3 - n_aug_;
  weights_(0) = lambda_ / (lambda_+n_aug_);
  for ( int i=1; i < 2*n_aug_ + 1; i++ )
  {
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }

  //NIS
  nis_las_ = 0.0;
  nis_rad_ = 0.0;

  ofstream nis_las_file, nis_rad_file;
  nis_las_file.open("nis_las_file.csv", ios::out | ios::trunc);
  nis_las_file.close();
  nis_rad_file.open("nis_rad_file.csv", ios::out | ios::trunc);
  nis_rad_file.close();
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  //Init
	if ( !is_initialized_ )
  {
    time_us_ = meas_package.timestamp_;

    if ( use_laser_ && (meas_package.sensor_type_ == MeasurementPackage::LASER) )
    {
      cout << "UKF::ProcessMeasurement Using LASER\n";
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
      is_initialized_ = true;
    }
    else if ( use_radar_ && (meas_package.sensor_type_ == MeasurementPackage::RADAR) )
    {
      cout << "UKF::ProcessMeasurement iUsing RADAR\n";
      double cos_phi = cos(meas_package.raw_measurements_[1]);
      double sin_phi = sin(meas_package.raw_measurements_[1]);
      x_(0) = meas_package.raw_measurements_[0] * cos_phi;
      x_(1) = meas_package.raw_measurements_[0] * sin_phi;
      is_initialized_ = true;
    }

    return;
  }

  //Timestamp update
  double dt = double(meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_  = meas_package.timestamp_;
  //Exececute prediction step
  Prediction(dt);
  //Measurement processing
  if ( use_laser_ && (meas_package.sensor_type_ == MeasurementPackage::LASER) )
  {
    cout << "UKF::ProcessMeasurement Laser measurement update\n";
    UpdateLidar(meas_package);
  }
  else if ( use_radar_ && (meas_package.sensor_type_ == MeasurementPackage::RADAR) )
  {
    cout << "UKF::ProcessMeasurement Radar measurement update\n";
    UpdateRadar(meas_package);
  }

  // print the output
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  //Augment state
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(n_x_)      = 0;
  x_aug(n_x_+1)    = 0;

  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_,   n_x_)             = std_a_*std_a_;
  P_aug(n_x_+1, n_x_+1)           = std_yawdd_*std_yawdd_;

  //Sigma points
  MatrixXd L        = P_aug.llt().matrixL();
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);
  Xsig_aug.col(0)   = x_aug;

  for ( int i=0; i<n_aug_; i++ )
  {
    Xsig_aug.col(1+i)        = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(1+n_aug_+i) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  //Predict sigma points
  for ( int i=0; i<2*n_aug_+1; i++ )
  {
    double p_x      = Xsig_aug(0,i);
    double p_y      = Xsig_aug(1,i);
    double v        = Xsig_aug(2,i);
    double yaw      = Xsig_aug(3,i);
    double yawd     = Xsig_aug(4,i);
    double nu_a     = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //Predicted values
    double px_p, py_p;

    //Division by zero check
    if ( fabs(yawd) > 0.001 )
    {
        px_p = p_x + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * (cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else
    {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p    = v;
    double yaw_p  = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //Noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p  = v_p  + nu_a*delta_t;

    yaw_p  = yaw_p  + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //Transfer predicted sigma point
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  //Predicted mean
  x_.fill(0.0);
  for ( int i=0; i<2*n_aug_+1; i++ )
  {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  //Predicted covariance matrix
  P_.fill(0.0);
  for ( int i=0; i<2*n_aug_+1; i++ )
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  //Transform sigma points to measurement space
  MatrixXd Zsig = MatrixXd(2, 2*n_aug_+1);
  for( int i=0; i<2*n_aug_+1; i++ )
  {
    Zsig(0,i) = Xsig_pred_(0,i);
    Zsig(1,i) = Xsig_pred_(1,i);
  }

  //Mean predicted measurement
  VectorXd z_pred = VectorXd(2);
  z_pred.fill(0.0);
  for ( int i=0; i<2*n_aug_+1; i++ )
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //Measurement covariance matrix S
  MatrixXd S = MatrixXd(2, 2);
  S.fill(0.0);
  for ( int i=0; i<2*n_aug_+1; i++ )
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //Measurement noise covariance matrix
  MatrixXd R = MatrixXd(2, 2);
  R << std_laspx_*std_laspx_, 0,
       0,                     std_laspy_*std_laspy_;
  S = S + R;

  //Cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, 2);
  Tc.fill(0.0);
  for ( int i=0; i<2*n_aug_+1; i++ )
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K and measurement error
  MatrixXd K      = Tc * S.inverse();
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //Ipdate state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  //Laser NIS
  nis_las_ = z_diff.transpose() * S.inverse() * z_diff;
  ofstream nis_las_file;
  nis_las_file.open("nis_las_file.csv", ios::app);
  nis_las_file << nis_las_ << ", ";
  nis_las_file.close();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  //Sigma points into measurement space
  MatrixXd Zsig = MatrixXd(3, 2*n_aug_+1);
  for( int i=0; i<2*n_aug_+1; i++ )
  {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v   = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    //Measurement model - r, phi, r_dot
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);
    Zsig(1,i) = atan2(p_y,p_x);
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);

  }

  //Predicted mean
  VectorXd z_pred = VectorXd(3);
  z_pred.fill(0.0);
  for ( int i=0; i<2*n_aug_+1; i++ )
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //Measurement covariance matrix S
  MatrixXd S = MatrixXd(3, 3);
  S.fill(0.0);
  for ( int i=0; i<2*n_aug_+1; i++ )
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //Measurement noise covariance matrix
  MatrixXd R = MatrixXd(3, 3);
  R << std_radr_*std_radr_, 0,                       0,
       0,                   std_radphi_*std_radphi_, 0,
       0,                   0,                       std_radrd_*std_radrd_;
  S = S + R;

  //Cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, 3);
  Tc.fill(0.0);
  for ( int i=0; i<2*n_aug_+1; i++ )
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K and measurement error
  MatrixXd K      = Tc * S.inverse();
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //State mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  //Radar NIS
  nis_rad_ = z_diff.transpose() * S.inverse() * z_diff;
  ofstream nis_rad_file;
  nis_rad_file.open("nis_rad_file.csv", ios::app);
  nis_rad_file << nis_rad_ << ", ";
  nis_rad_file.close();
}
