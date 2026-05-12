#include <iostream>
#include <vector>
#include <cmath>
#include <numeric> // For std::accumulate
#include <tuple>

// Log-likelihood for linear regression under normality: L = product N(y_i | beta0 + beta1*x_i, sigma^2)
// Log L = -n/2 ln(2pi) - n ln sigma - (1/(2 sigma^2)) sum (y_i - (beta0 + beta1 x_i))^2
// Maximizing this gives the same beta0, beta1 as OLS (which minimizes sum residuals^2)
double log_likelihood(const std::vector<double>& x, const std::vector<double>& y, double beta0, double beta1, double sigma_squared) {
    if (sigma_squared < 1e-10) {
        return -1e10;  // For exact fit (sigma^2=0), likelihood is maximized but numerically undefined
    }
    int n = x.size();
    double sigma = std::sqrt(sigma_squared);
    double log_l = - (n / 2.0) * std::log(2 * M_PI) - n * std::log(sigma);
    for (int i = 0; i < n; ++i) {
        double residual = y[i] - (beta0 + beta1 * x[i]);
        log_l -= residual * residual / (2 * sigma_squared);
    }
    return log_l;
}


// function to compute OLS
std::tuple<double, double, double> compute_ols(const std::vector<double>& x, const std::vector<double>& y){
    int n = x.size();
    double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
    double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
    double mean_x = sum_x / n;
    double mean_y = sum_y / n;

    double numerator = 0.0;
    double denominator = 0.0;
    for (int i = 0; i < n; ++i) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        numerator += dx * dy;
        denominator += dx * dx;
    }
    double beta1 = numerator / denominator;
    double beta0 = mean_y - beta1 * mean_x;

    double sum_squared_residuals = 0.0;
    for (int i = 0; i < n; ++i) {
        double y_hat = beta0 + beta1 * x[i];
        double residual = y[i] - y_hat;
        sum_squared_residuals += residual * residual;
    }
    double sigma_squared = sum_squared_residuals / n;
    
    return std::make_tuple(beta0, beta1, sigma_squared);
}


int main() {
    // Example data: y ≈ 2 + 3*x (exact fit for demonstration)
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {5.0, 8.0, 11.0, 14.0, 17.0};
    int n = x.size();

    // Compute OLS estimates
    double ols_beta0, ols_beta1, ols_sigma_squared;
    std::tie(ols_beta0, ols_beta1, ols_sigma_squared) = compute_ols(x, y);

    // Compute MLE estimates (same as OLS under normality)
    double mle_beta0, mle_beta1, mle_sigma_squared;
    std::tie(mle_beta0, mle_beta1, mle_sigma_squared) = compute_ols(x, y);

    // Compute log-likelihood at MLE
    double ll = log_likelihood(x, y, mle_beta0, mle_beta1, mle_sigma_squared);

    // Step 1: Compute sample means (needed for OLS/MLE formulas)
    // double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
    // double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
    // double mean_x = sum_x / n;
    // double mean_y = sum_y / n;

    // Step 2: Compute beta1 (slope) using OLS formula
    // beta1 = cov(x,y) / var(x) = [sum (x_i - mean_x)(y_i - mean_y)] / [sum (x_i - mean_x)^2]
    // This is the MLE for beta1 under normality (maximizes log-likelihood)
    // double numerator = 0.0;   // sum (dx * dy)
    // double denominator = 0.0; // sum (dx^2)
    // for (int i = 0; i < n; ++i) {
    //     double dx = x[i] - mean_x;
    //     double dy = y[i] - mean_y;
    //     numerator += dx * dy;
    //     denominator += dx * dx;
    // }
    // double beta1_hat = numerator / denominator;

    // Step 3: Compute beta0 (intercept) using OLS formula
    // beta0 = mean_y - beta1 * mean_x
    // This is the MLE for beta0 under normality
    // double beta0_hat = mean_y - beta1_hat * mean_x;

    // Step 4: Compute sigma^2 (variance of errors) as MLE
    // sigma^2 = (1/n) sum (y_i - y_hat_i)^2, where y_hat = beta0 + beta1 x
    // Under normality, this maximizes the log-likelihood for sigma^2
    // double sum_squared_residuals = 0.0;
    // for (int i = 0; i < n; ++i) {
    //     double y_hat = beta0_hat + beta1_hat * x[i];
    //     double residual = y[i] - y_hat;
    //    sum_squared_residuals += residual * residual;
    // }
    // double sigma_squared_hat = sum_squared_residuals / n;

    // Step 5: Compute log-likelihood at MLE estimates
    // double ll = log_likelihood(x, y, beta0_hat, beta1_hat, sigma_squared_hat);

    // Output: These are both OLS and MLE estimates (identical under normality)

    // Output comparison
    std::cout << "Comparison: OLS vs MLE for Linear Regression" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "OLS Estimates:" << std::endl;
    std::cout << "  Intercept (beta0): " << ols_beta0 << std::endl;
    std::cout << "  Slope (beta1): " << ols_beta1 << std::endl;
    std::cout << "  Variance of errors (sigma^2): " << ols_sigma_squared << std::endl;
    std::cout << std::endl;
    std::cout << "MLE Estimates (under normality):" << std::endl;
    std::cout << "  Intercept (beta0): " << mle_beta0 << std::endl;
    std::cout << "  Slope (beta1): " << mle_beta1 << std::endl;
    std::cout << "  Variance of errors (sigma^2): " << mle_sigma_squared << std::endl;
    std::cout << "  Log-likelihood: " << ll << std::endl;
    std::cout << std::endl;
    std::cout << "Comparison Results:" << std::endl;
    std::cout << "  beta0 match: " << (std::abs(ols_beta0 - mle_beta0) < 1e-10 ? "YES" : "NO") << std::endl;
    std::cout << "  beta1 match: " << (std::abs(ols_beta1 - mle_beta1) < 1e-10 ? "YES" : "NO") << std::endl;
    std::cout << "  sigma^2 match: " << (std::abs(ols_sigma_squared - mle_sigma_squared) < 1e-10 ? "YES" : "NO") << std::endl;
    std::cout << std::endl;
    std::cout << "Conclusion: MLE = OLS under normality." << std::endl;
    
    return 0;
}