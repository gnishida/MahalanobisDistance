/**
 * OpenCVを使って、マハラノビス距離を計算するサンプルプログラム。
 *
 * matlabで同様の計算するには、
 * > X = mvnrnd([0;0], [1 .9; .9 1], 20);
 * > Y = [1 1; 1 -1; -1 1; -1 -1];
 * > sqrt(mahal(Y, X))
 * ans = 1.9636
 *       6.3419
 *       5.9926
 *       0.5291
 * ※ matlabでは、mahal関数はマハラノビス距離の二乗を返すので、sqrtする必要がある。
 *
 *
 * @author	Gen Nishida
 * @date	3/17/2015
 * @version	1.0
 */

#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;

int main() {
	Mat_<double> samples = (Mat_<double>(10, 2) << 
		0.0229, 0.2473,
		-0.2620, -0.2445,
		-1.7502, -1.5903,
		-0.2857, -0.6050,
		-0.8314, -0.3042,
		-0.9792, -0.9394,
		-1.1564, -1.3522,
		-0.5336, 0.1089,
		-2.0026, -1.9003,
		0.9642, 0.6111);
	
	Mat covar, mean;
	calcCovarMatrix(samples, covar, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
	covar = covar / (samples.rows - 1);

	std::cout << covar << std::endl;
	std::cout << mean << std::endl;

	Mat invCovar;
	cv::invert(covar, invCovar);

	Mat_<double> vec1 = (Mat_<double>(1, 2) << 1, 1);
	Mat_<double> vec2 = (Mat_<double>(1, 2) << 1, -1);
	Mat_<double> vec3 = (Mat_<double>(1, 2) << -1, 1);
	Mat_<double> vec4 = (Mat_<double>(1, 2) << -1, -1);

	printf("dist: %lf\n", cv::Mahalanobis(vec1, mean, invCovar));
	printf("dist: %lf\n", cv::Mahalanobis(vec2, mean, invCovar));
	printf("dist: %lf\n", cv::Mahalanobis(vec3, mean, invCovar));
	printf("dist: %lf\n", cv::Mahalanobis(vec4, mean, invCovar));

	return 0;
}