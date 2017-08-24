#ifndef NISTFIT_NUMERIC_EVALUATORS_
#define NISTFIT_NUMERIC_EVALUATORS_

#include "NISTfit/abc.h"

namespace NISTfit{
    
/// The class for the evaluation of a single output value for a single input value
class PolynomialOutput : public NumericOutput {
protected:
    std::size_t m_order; // The order of the polynomial (2: quadratic, 3: cubic, etc...)
public:
    PolynomialOutput(std::size_t order,
                     const std::shared_ptr<NumericInput> &in) 
        : NumericOutput(in), m_order(order) {
        resize(order + 1); // Set the size of the Jacobian row
    };
    /// The exception handler must be implemented; here we just 
    /// set the residue to a very large number
    void exception_handler() override { m_y_calc = 10000; };
    void evaluate_one() override {
        // Get the input
        double lhs = 0;
        // Do the calculation
        const std::vector<double> &c = get_AbstractEvaluator().get_const_coefficients();
        if (c.size() != m_order +1){ throw std::range_error("lengths do not agree"); }
        for (std::size_t i = 0; i < m_order+1; ++i) {
            double term = pow(m_in->x(), static_cast<int>(i));
            lhs += c[i]*term;
            Jacobian_row[i] = term;
        }
        m_y_calc = lhs;
    };
};

/// The class for the evaluation of a single output value for a single input value
class PolynomialOutputAlt : public NumericOutput {
protected:
	const std::size_t m_order; // The order of the polynomial (2: quadratic, 3: cubic, etc...)
public:
	PolynomialOutputAlt(std::size_t order,
		const std::shared_ptr<NumericInput> &in)
		: NumericOutput(in), m_order(order) {
		resize(order + 1); // Set the size of the Jacobian row
	};
	/// The exception handler must be implemented; here we just 
	/// set the residue to a very large number
	void exception_handler() override { m_y_calc = 10000; };
	void evaluate_one() override {
		// Do the calculation
		const std::vector<double> &c = get_AbstractEvaluator().get_const_coefficients();
		// m_order + 1 is always >= 1 since m_order is of type size_t and thus >=0
		if (c.size() != m_order + 1) { throw std::range_error("lengths do not agree"); }
		// Backwards Horner scheme
		/*m_y_calc = c[m_order];
		for (std::size_t i = m_order - 1; i >= 0; --i) {
			m_y_calc = m_y_calc*m_in->x() + c[i];
		}*/
		double exp_term = 1.0;
		m_y_calc = c[0];
		Jacobian_row[0] = 1.0;
		for (std::size_t i = 1; i < m_order + 1; ++i) {
			exp_term = exp_term *  m_in->x();
			m_y_calc += c[i] * exp_term;
			Jacobian_row[i] = Jacobian_row[i-1] * m_in->x();
		}
	};
};

}; /* namespace NISTfit */

#endif
