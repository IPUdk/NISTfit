// This file is part of the NISTfit C++ fitting template library
//
// Created in 2017 by Jorrit Wronski <jowr@ipu.dk>
//
// The source code is not subject to copyright. This contribution 
// is in the public domain and can be redistributed and / or modified 
// freely provided that any derivative works bear some notice that 
// they are derived from it, and any modified versions bear some 
// notice that they have been modified.

#include "NISTfit/abc.h"
#include "NISTfit/optimizers.h"
#include "NISTfit/numeric_evaluators.h"
//#include "NISTfit/examples.h"

#include "Eigen/Core"
#include "Eigen/Dense"
#include "unsupported/Eigen/Polynomials"


#include <iostream>
#include <chrono>

using namespace NISTfit;

void get_inputs(const std::size_t Nmax, std::vector<std::shared_ptr<NumericInput> > &inputs) {
	inputs.resize(Nmax);
	for (std::size_t i = 0; i < Nmax; ++i) {
		double x = ((double)i) / ((double)Nmax);
		double y = 1 + x*(2 + x*(3 + x*(4 + x*(5 + x*(6 + x*(7 + x*(8 + x*(9 + x * 10))))))));
		inputs[i] = std::shared_ptr<NumericInput>(new NumericInput(x, y));
	}
}

void get_c0_solved(std::vector<double> &c0) {
	c0 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
}

void get_c0(std::vector<double> &c0) {
	c0 = { -10, -2, 2.5, -8, 5, 7, 6, -20, -13, -1 };
}

/**
* @brief The original polynomial example that illustrates the overhead of threaded calculations
* @param threading Should we use multi threading or not?
* @param Nmax The number of points to be fitted (maps automatically to 0..1)
* @param Nthreads The number of threads to be used
*/
double fit_polynomial(bool threading, std::size_t Nmax, short Nthreads)
{
	std::vector<std::shared_ptr<NumericInput> > inputs;
	get_inputs(Nmax, inputs);
    std::vector<std::shared_ptr<AbstractOutput> > outputs;
	outputs.resize(Nmax);
    for (std::size_t i = 0; i < Nmax; ++i) {
        outputs[i] = std::shared_ptr<AbstractOutput>(new PolynomialOutput(9, inputs[i]));
    }
    std::shared_ptr<AbstractEvaluator> eval(new NumericEvaluator());
    eval->add_outputs(outputs);
    auto opts = LevenbergMarquardtOptions();
	get_c0(opts.c0);
	opts.threading = threading; 
	opts.Nthreads = Nthreads;
	auto startTime = std::chrono::system_clock::now();
    auto cc = LevenbergMarquardt(eval, opts);
    auto endTime = std::chrono::system_clock::now();
    return std::chrono::duration<double>(endTime - startTime).count();
}

/**
* @brief The polynomial example that uses an alternative evaluator
* @param threading Should we use multi threading or not?
* @param Nmax The number of points to be fitted (maps automatically to 0..1)
* @param Nthreads The number of threads to be used
*/
double fit_polynomial_alt(bool threading, std::size_t Nmax, short Nthreads)
{
	std::vector<std::shared_ptr<NumericInput> > inputs;
	get_inputs(Nmax, inputs);
	std::vector<std::shared_ptr<AbstractOutput> > outputs;
	outputs.resize(Nmax);
	for (std::size_t i = 0; i < Nmax; ++i) {
		outputs[i] = std::shared_ptr<AbstractOutput>(new PolynomialOutputAlt(9, inputs[i]));
	}
	std::shared_ptr<AbstractEvaluator> eval(new NumericEvaluator());
	eval->add_outputs(outputs);
	auto opts = LevenbergMarquardtOptions();
	get_c0(opts.c0);
	opts.threading = threading;
	opts.Nthreads = Nthreads;
	auto startTime = std::chrono::system_clock::now();
	auto cc = LevenbergMarquardt(eval, opts);
	auto endTime = std::chrono::system_clock::now();
	return std::chrono::duration<double>(endTime - startTime).count();
}

/**
* @brief A least squares polynomial fit using Eigen
* @param x The independent variable (sampling points)
* @param y The dependent variable (sampled data)
* @param order The order of the fitted polynomial
* @param result The polynomial coefficients (length = 1 + order)
*/
void _fit_polynomial_eigen(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const size_t order, Eigen::VectorXd &result) 
{
	// Create the full Vandermonde matrix
	Eigen::MatrixXd A = Eigen::MatrixXd(x.size(), 1 + order);
	A.col(0).setConstant(1.0);
	for (size_t i = 1; i < order + 1; ++i) {
		A.col(i) = A.col(i - 1).cwiseProduct(x);
	}
	result = A.colPivHouseholderQr().solve(y);
	return;
}

/**
* @brief A least squares fit of a rational polynomial using Eigen
* @param x The independent variable (sampling points)
* @param y The dependent variable (sampled data)
* @param n_order The order of the numerator of the fitted polynomial
* @param d_order The order of the denominator of the fitted polynomial
* @param result The polynomial coefficients (length = 1 + n_order + 1 + d_order)
*/
void _fit_rational_polynomial_eigen(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const size_t n_order, const size_t d_order, Eigen::VectorXd &result)
{
	// Create the full Vandermonde matrix
	Eigen::MatrixXd A = Eigen::MatrixXd(x.size(), 1 + n_order + d_order); // Forcing B0 to 1!
	// Handle the numerator
	A.col(0).setConstant(1.0);
	for (size_t i = 1; i < n_order + 1; ++i) {
		A.col(i) = A.col(i - 1).cwiseProduct(x);
	}
	// Handle the denominator
	size_t idx = n_order + 1;
	A.col(idx) = -1 * y.cwiseProduct(x);
	for (size_t i = 1; i < d_order; ++i) {
		idx = i + n_order + 1;
		A.col(idx) = A.col(idx - 1).cwiseProduct(x);
	}
	Eigen::VectorXd tmp_res = A.colPivHouseholderQr().solve(y);

	result.resize(1 + n_order + 1 + d_order);
	result.segment(0, 1 + n_order) = tmp_res.segment(0, 1 + n_order);
	result[n_order + 1] = 1.0;
	result.segment(n_order + 2, d_order) = tmp_res.segment(1 + n_order, d_order);

	//Eigen::VectorXd y_new = Eigen::VectorXd::Zero(x.size());
	//result = A.fullPivLu().solve(yw);
	return;
}
//void _fit_rational_polynomial_eigen(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const size_t n_order, const size_t d_order, Eigen::VectorXd &result) 
//{
//	// Create the full Vandermonde matrix
//	Eigen::MatrixXd A = Eigen::MatrixXd(x.size(), 1 + n_order + 1 + d_order);
//	// Handle the numerator
//	A.col(0).setConstant(1.0);
//	for (size_t i = 1; i < n_order + 1; ++i) {
//		A.col(i) = A.col(i - 1).cwiseProduct(x);
//	}
//	// Handle the denominator
//	size_t idx = n_order + 1;
//	A.col(idx) = -1 * y;
//	for (size_t i = 1; i < d_order + 1; ++i) {
//		idx = i + n_order + 1;
//		A.col(idx) = A.col(idx - 1).cwiseProduct(x);
//	}
//	result = A.colPivHouseholderQr().solve(y);
//	//Eigen::VectorXd y_new = Eigen::VectorXd::Zero(x.size());
//	//result = A.fullPivLu().solve(yw);
//	return;
//}

/**
* @brief The polynomial example that uses the Eigen modules
* @param threading Should we use multi threading or not?
* @param Nmax The number of points to be fitted (maps automatically to 0..1)
* @param Nthreads The number of threads to be used
*/
double fit_polynomial_eigen(const bool threading, const std::size_t Nmax, const short Nthreads)
{
	std::vector<double> c0;
	get_c0_solved(c0);
	Eigen::VectorXd poly = Eigen::Map<Eigen::VectorXd>(c0.data(), c0.size());
	Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(Nmax, 0, 1);
	Eigen::VectorXd y(Nmax); // = Eigen::poly_eval(poly, x);		
	for (int i = 0; i < Nmax; ++i) {
		y[i] = Eigen::poly_eval(poly, x[i]);
	}
	int threads = Eigen::nbThreads();
	if (!threading) {
		Eigen::setNbThreads(1);
	} else {
		Eigen::setNbThreads(Nthreads);
	}
	Eigen::VectorXd cc;
	auto startTime = std::chrono::system_clock::now();
	_fit_polynomial_eigen(x, y, poly.size()-1, cc);
	auto endTime = std::chrono::system_clock::now();
	Eigen::setNbThreads(threads); // Reset thread counter to default (0?) 
	std::vector<double> c(&cc[0], cc.data() + cc.cols()*cc.rows());
	return std::chrono::duration<double>(endTime - startTime).count();
}

/**
* @brief The rational polynomial example that uses the Eigen modules
* @param threading Should we use multi threading or not?
* @param Nmax The number of points to be fitted (maps automatically to 0..1)
* @param Nthreads The number of threads to be used
*/
double fit_rational_polynomial_eigen(const bool threading, const std::size_t Nmax, const short Nthreads)
{
	Eigen::VectorXd n_poly(4); 
	n_poly << 1, 8, -50, 40;
	Eigen::VectorXd d_poly(3);
	d_poly << 5, 1, 8 ;
	Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(Nmax, 0, 1);
	Eigen::VectorXd y(Nmax);
	for (int i = 0; i < Nmax; ++i) {
		y[i] = Eigen::poly_eval(n_poly, x[i]) / Eigen::poly_eval(d_poly, x[i]);
	}
	//y = y.cwiseQuotient(Eigen::poly_eval(d_poly, x));
	int threads = Eigen::nbThreads();
	if (!threading) {
		Eigen::setNbThreads(1);
	} else {
		Eigen::setNbThreads(Nthreads);
	}
	Eigen::VectorXd cc;
	auto startTime = std::chrono::system_clock::now();
	_fit_rational_polynomial_eigen(x, y, n_poly.size() - 1, d_poly.size() - 1, cc);
	auto endTime = std::chrono::system_clock::now();
	Eigen::setNbThreads(threads); // Reset thread counter to default (0?) 
	std::vector<double> c(&cc[0], cc.data() + cc.cols()*cc.rows()); 
	return std::chrono::duration<double>(endTime - startTime).count();
}

void speedtest_fit_polynomial(short Nthread_max)
{
    std::cout << "XXXXXXXXXX POLYNOMIAL XXXXXXXXXX" << std::endl;
    for (std::size_t N = 1000; N < 100001; N *= 10) {
        auto time_serial_std = fit_polynomial(false, N, 1);
		auto time_serial_alt = fit_polynomial_alt(false, N, 1);
		auto time_serial_eig = fit_polynomial_eigen(false, N, 1);
        for (short Nthreads = 2; Nthreads <= Nthread_max; Nthreads+=2) {
            const bool threading = true;
            auto time_parallel_std = fit_polynomial(threading, N, Nthreads);
			auto time_parallel_alt = fit_polynomial_alt(threading, N, Nthreads);
			auto time_parallel_eig = fit_polynomial_eigen(threading, N, Nthreads);
            printf("Std: %2d %10d %10.5f %10.5f(nothread) %10.5f(threaded)\n", Nthreads, static_cast<int>(N), time_serial_std / time_parallel_std, time_serial_std, time_parallel_std);
			printf("Alt: %2d %10d %10.5f %10.5f(nothread) %10.5f(threaded)\n", Nthreads, static_cast<int>(N), time_serial_alt / time_parallel_alt, time_serial_alt, time_parallel_alt);
			printf("Eig: %2d %10d %10.5f %10.5f(nothread) %10.5f(threaded)\n", Nthreads, static_cast<int>(N), time_serial_eig / time_parallel_eig, time_serial_eig, time_parallel_eig);
		}
	}
}

void speedtest_fit_rational_polynomial(short Nthread_max)
{
	std::cout << "XXXXXXXXXX RATIONAL POLYNOMIAL XXXXXXXXXX" << std::endl;
	for (std::size_t N = 1000; N < 100001; N *= 10) {
		//auto time_serial_std = fit_polynomial(false, N, 1);
		//auto time_serial_alt = fit_polynomial_alt(false, N, 1);
		auto time_serial_eig = fit_rational_polynomial_eigen(false, N, 1);
		for (short Nthreads = 2; Nthreads <= Nthread_max; Nthreads += 2) {
			const bool threading = true;
			//auto time_parallel_std = fit_polynomial(threading, N, Nthreads);
			//auto time_parallel_alt = fit_polynomial_alt(threading, N, Nthreads);
			auto time_parallel_eig = fit_rational_polynomial_eigen(threading, N, Nthreads);
			//printf("Std: %2d %10d %10.5f %10.5f(nothread) %10.5f(threaded)\n", Nthreads, static_cast<int>(N), time_serial_std / time_parallel_std, time_serial_std, time_parallel_std);
			//printf("Alt: %2d %10d %10.5f %10.5f(nothread) %10.5f(threaded)\n", Nthreads, static_cast<int>(N), time_serial_alt / time_parallel_alt, time_serial_alt, time_parallel_alt);
			printf("Eig: %2d %10d %10.5f %10.5f(nothread) %10.5f(threaded)\n", Nthreads, static_cast<int>(N), time_serial_eig / time_parallel_eig, time_serial_eig, time_parallel_eig);
		}
	}
}


int main(){
	Eigen::initParallel();
    short Nthread_max = std::min(static_cast<short>(10), static_cast<short>(std::thread::hardware_concurrency()));
#ifdef NTHREAD_MAX
    Nthread_max = NTHREAD_MAX;
#endif
    std::cout << "Max # of threads: " << Nthread_max << std::endl;
	speedtest_fit_rational_polynomial(Nthread_max);
	speedtest_fit_polynomial(Nthread_max);	
}
