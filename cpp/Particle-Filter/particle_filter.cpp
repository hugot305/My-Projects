/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;

  // resize weights and particles vector
  weights.resize(num_particles);
  particles.resize(num_particles);

  // define random number generation engine
  random_device rd;
  default_random_engine gen(rd());

  // define noise
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // create particles
  for(int i = 0; i < num_particles; ++i) {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;

  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  // define random number generation engine
  default_random_engine gen;

  // define noise
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; ++i) {
    if (fabs(yaw_rate) < 0.00001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // add noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  // start looping over observations
  for(unsigned i = 0; i < observations.size(); ++i) {
    LandmarkObs current_obs = observations[i];
    double min_dist = INFINITY;
    int closest_particle_id = -1;

    // start looping over predicted measurements
    for(unsigned j = 0; j < predicted.size(); ++j) {
      LandmarkObs current_pred = predicted[j];
      double current_dist = dist(current_obs.x, current_obs.y, current_pred.x, current_pred.y);

      // get the current closest particle's id
      if (current_dist < min_dist) {
        min_dist = current_dist;
        closest_particle_id = j;
      }
    }

    // assign closest particle id to the observation
    observations[i].id = closest_particle_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  // set static values for multi-variate Gaussian
  auto cov_x = std_landmark[0] * std_landmark[0];
  auto cov_y = std_landmark[1] * std_landmark[1];
  auto normalizer = 2.0 * M_PI * std_landmark[0] * std_landmark[1];

  // start looping over each particle
  for(int i = 0; i < num_particles; ++i) {
    // get the particle x, y coordinates
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    // filter out landmarks outside sensor range of the particle
    vector<LandmarkObs> predictions;
    for(unsigned j = 0; j < map_landmarks.landmark_list.size(); ++j) {

      // get id and x,y coordinates
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;

//      if(fabs(lm_x - p_x) <= sensor_range && fabs(lm_y - p_y) <= sensor_range) {
      if(dist(lm_x, lm_y, p_x, p_y) <= sensor_range) {
        // add prediction to vector
        predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
      }
    }

    // only update weights when there's any landmark in sensor range
    // otherwise set the weight of this particle to zero
    if(predictions.size() == 0) {
      particles[i].weight = 0;
      weights[i] = 0;
    }
    else {
      // update step 1: transform
      vector<LandmarkObs> transformed_obs;
      for(unsigned k = 0; k < observations.size(); ++k) {
        double t_x = cos(p_theta) * observations[k].x - sin(p_theta) * observations[k].y + p_x;
        double t_y = sin(p_theta) * observations[k].x + cos(p_theta) * observations[k].y + p_y;
        transformed_obs.push_back(LandmarkObs{observations[k].id, t_x, t_y});
      }

      // update step 2: associate
      dataAssociation(predictions, transformed_obs);

      // update step 3: calculate the weight with multi-variate Gaussian
      double total_prob = 1.0;
      for(unsigned l = 0; l < transformed_obs.size(); ++l) {
        auto obs = transformed_obs[l];
        auto predicted = predictions[obs.id];

        auto dx = obs.x - predicted.x;
        auto dy = obs.y - predicted.y;
        total_prob *= exp(-(dx * dx / (2 * cov_x) + dy * dy / (2 * cov_y))) / normalizer;
      }
      particles[i].weight = total_prob;
      weights[i] = total_prob;
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  vector<Particle> new_particles(num_particles);
  random_device rd;
  default_random_engine gen(rd());
  discrete_distribution<int> weight_pmf(weights.begin(), weights.end());

  for (int i = 0; i < num_particles; ++i) {
    new_particles[i] = particles[weight_pmf(gen)];
  }

  // replace old particles with new ones by using std::move to avoid deep copy
  particles = move(new_particles);
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
