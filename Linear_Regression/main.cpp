#include <iostream>
#include <fstream> // write to the file

using namespace std;




int main() {
    double x[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double y[] = {2, 4, 5, 4, 5, 6, 7, 6, 8, 9};
    int n = 10; // number of data points

    cout << "x values: ";
    for (int i = 0; i < n; ++i){
        cout << x[i] << " ";
    }
    cout << endl;

    cout << "y values ";
    for (int i = 0; i < n; ++i){
        cout << y[i] << " ";
    } 
    cout << endl;

    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;

    for (int i = 0; i < n; i++){
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
    }

    cout << "sum_x: " << sum_x << endl;
    cout << "sum_y: " << sum_y << endl;
    cout << "sum_xy: " << sum_xy << endl;
    cout << "sum_x2: " << sum_x2 << endl;

    double m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    double b = (sum_y - m * sum_x) / n;

    cout << "Slope (m): " << m << endl;
    cout << "Intercept (b): " << b << endl;

    double x_new = 6; // try predictiong for x=6
    double y_pred = m * x_new + b;
    cout << "Predicted y for x = " << x_new << " is:" << y_pred << endl;

    // Write data to file
    ofstream dataFile("data.txt");
    for (int i = 0; i < n; i++) {
        dataFile << x[i] << " " << y[i] << endl;
    }
    dataFile.close();

    // Write regression line to file
    ofstream lineFile("line.txt");
    for (int i = 0; i < n; i++) {
    lineFile << x[i] << " " << (m * x[i] + b) << endl;
    }
    lineFile.close();

    // Write residuals to file
    ofstream residualFile("residuals.txt");
    for (int i = 0; i < n; i++) {
        double y_pred_i = m * x[i] + b;
        double residual = y[i] - y_pred_i;
        residualFile << x[i] << " " << residual << endl;
    }
    residualFile.close();

    // Write predicted values to file
    // ofstream predFile("predicted.txt");
    // for (int i = 0; i < n; i++){
    //    double y_pred_i = m * x[i] + b;
    //    predFile << x[i] << " " << y_pred_i << endl;
    // }
    // predFile.close();

    // Run gnuplot command
    system("gnuplot -e \"set terminal png size 800,600; set output 'plot.png'; plot 'data.txt' with points title 'Real Data', 'line.txt' with lines title 'Regression Line', 'predicted.txt' with points pt 7 lc rgb 'red' title 'Predicted Values', 'residuals.txt' with boxes title 'Residuals';\"");
    
    cout << "Graphic is saved to plot.png " << endl;

    
    return 0;
}