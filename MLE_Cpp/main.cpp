#include <iostream>
#include <vector>
#include <cmath>
#include <numeric> // For std::accumulate

double log_likelihood(const std::vector<double>& data, double mu, double sigma_squared) {
    int n = data.size();
    double sigma = std::sqrt(sigma_squared);
    double log_l = - (n / 2.0) * std::log(2 * M_PI) - n * std::log(sigma);
    for (double x : data){
        log_l -= (x - mu) * (x - mu) / (2 * sigma_squared);
    }
    return log_l;
}

int main(){
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0}; // Example data
    int n = data.size(); // Number of observations

    // Compute Sample Mean (MLE for μ)
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mu_hat = sum / n;

    // Compute Sample Variance (MLE for σ²)
    double sum_squared_diff = 0.0;
    for (double x : data) {
        sum_squared_diff += (x - mu_hat) * (x - mu_hat);
    }
    double sigma_squared_hat = sum_squared_diff / n;

    double ll = log_likelihood(data, mu_hat, sigma_squared_hat);
    std::cout << "Log-likelihood at MLE: " << ll << std::endl;

    // Output the results
    std::cout << "MLE Estimate for Mean: " << mu_hat << std::endl;
    std::cout << "MLE Estimate for Variance: " << sigma_squared_hat << std::endl;
    std::cout << "Standart Deviation: " << std::sqrt(sigma_squared_hat) << std::endl;

    return 0;
}

