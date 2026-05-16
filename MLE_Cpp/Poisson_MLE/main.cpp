#include <iostream>
#include <vector>
#include <cmath>
#include <numeric> // For std::accumulate

double log_likelihood_poisson(const std::vector<int>& data, double lambda) {
    double log_l = 0.0;
    for (int x: data) {
        // ln(x!) can be computed using std::lgamma(x+1)
        log_l += x * std::log(lambda) - lambda - std::lgamma(x + 1);
    }
    return log_l;
}

int main() {
    // Example data: Poisson-distributed counts (e.g arrivals per hour)
    // For reference, if lambda = 2.7, these are typical counts
    std::vector<int> data = {2,3,1,4,2,3,5,1,2,4};
    int n = data.size();

    // Compute MLE for lambda (just sample mean !)
    double sum_data = std::accumulate(data.begin(), data.end(), 0.0);
    double lambda_hat = sum_data / n;

    double ll = log_likelihood_poisson(data, lambda_hat);

    std::cout << "Poisson MLE Estimation:" << std::endl;
    std::cout << "Sample data: ";
    for (int x : data) std::cout << x << " ";
    std::cout << std::endl;
    std::cout << "MLE for lambda (expected value): " << lambda_hat << std::endl;
    std::cout << "Log-Likelihood at MLE: " << ll <<std::endl;
    std::cout << "\nKey insight: For poisson, MLE(lambda) = sample mean !" << std::endl;


    return 0;
}
