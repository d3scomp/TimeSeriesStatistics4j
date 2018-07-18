package cz.cuni.mff.d3s.tss.arima;

import java.util.Arrays;

public class LagPolynomial {

	final double[] parameters;
	private final double[] coefficients;
	private final int degree;

	/**
	 * Construct a new lag polynomial from the given parameters. Note that the given
	 * parameters are not the same as the polynomial coefficients since the
	 * coefficient at the zero-degree term is always equal to 1.
	 *
	 * @param parameters
	 *            the parameters of the polynomial.
	 */
	LagPolynomial(final double... parameters) {
		this.parameters = parameters.clone();
		this.coefficients = new double[parameters.length + 1];
		this.coefficients[0] = 1.0;
		System.arraycopy(parameters, 0, this.coefficients, 1, parameters.length);
		this.degree = parameters.length;
	}

	/**
	 * Create and return a new lag polynomial representing the first difference
	 * operator.
	 *
	 * @return a new lag polynomial representing the first difference operator.
	 */
	public static LagPolynomial firstDifference() {
		return new LagPolynomial(-1.0);
	}

	/**
	 * Create and return a new lag polynomial representing the first seasonal
	 * difference operator.
	 *
	 * @param seasonalLag
	 *            the number of periods in one seasonal cycle.
	 * @return a new lag polynomial representing the first seasonal difference
	 *         operator.
	 */
	public static LagPolynomial firstSeasonalDifference(final int seasonalLag) {
		double[] poly = new double[seasonalLag];
		poly[seasonalLag - 1] = -1.0;
		return new LagPolynomial(poly);
	}

	/**
	 * Create and return a new lag polynomial representing an arbitrary number of
	 * seasonal differences.
	 *
	 * @param seasonalLag
	 *            the number of periods in one seasonal cycle.
	 * @param D
	 *            the number of seasonal differences.
	 * @return a new lag polynomial representing an arbitrary number of seasonal
	 *         differences.
	 */
	public static LagPolynomial seasonalDifferences(final int seasonalLag, final int D) {
		if (D < 0) {
			throw new IllegalArgumentException(
					"The degree of differencing must be greater than or equal to 0, but was " + D);
		}
		if (D > 0) {
			LagPolynomial diff = LagPolynomial.firstSeasonalDifference(seasonalLag);
			for (int i = 1; i < D; i++) {
				diff = diff.times(diff);
			}
			return diff;
		} else {
			return new LagPolynomial();
		}
	}

	/**
	 * Create and return a new lag polynomial representing an arbitrary number of
	 * differences.
	 *
	 * @param d
	 *            the number of differences. An integer greater than or equal to 0.
	 * @return a new lag polynomial representing an arbitrary number of differences.
	 *
	 * @throws IllegalArgumentException
	 *             if the degree of differencing is less than 0.
	 */
	public static LagPolynomial differences(final int d) {
		if (d < 0) {
			throw new IllegalArgumentException(
					"The degree of differencing must be greater than or equal to 0, but was " + d);
		}
		if (d > 0) {
			LagPolynomial diff = LagPolynomial.firstDifference();
			for (int i = 1; i < d; i++) {
				diff = diff.times(diff);
			}
			return diff;
		} else {
			return new LagPolynomial();
		}
	}

	/**
	 * Create and return a new moving average lag polynomial.
	 *
	 * @param parameters
	 *            the moving average parameters of an ARIMA model.
	 * @return a new moving average lag polynomial.
	 */
	public static LagPolynomial movingAverage(double... parameters) {
		return new MovingAveragePolynomial(parameters);
	}

	/**
	 * Create and return a new autoregressive lag polynomial.
	 *
	 * @param parameters
	 *            the autoregressive parameters of an ARIMA model.
	 * @return a new autoregressive lag polynomial.
	 */
	public static LagPolynomial autoRegressive(double... parameters) {
		final double[] inverseParams = new double[parameters.length];
		for (int i = 0; i < inverseParams.length; i++) {
			inverseParams[i] = -parameters[i];
		}
		return new LagPolynomial(inverseParams);
	}

	/**
	 * Multiply this polynomial by another lag polynomial and return the result in a
	 * new lag polynomial.
	 *
	 * @param other
	 *            the polynomial to multiply this one with.
	 * @return the product of this polynomial with the given polynomial.
	 */
	public final LagPolynomial times(final LagPolynomial other) {
		final double[] newParams = new double[this.degree + other.degree + 1];
		for (int i = 0; i < coefficients.length; i++) {
			for (int j = 0; j < other.coefficients.length; j++) {
				newParams[i + j] += coefficients[i] * other.coefficients[j];
			}
		}
		return new LagPolynomial(slice(newParams, 1, newParams.length));
	}
	
	/**
     * Return a slice of the data between the given indices.
     *
     * @param data the data to slice.
     * @param from the starting index.
     * @param to   the ending index. The value at this index is excluded from the result.
     * @return a slice of the data between the given indices.
     */
    public static double[] slice(final double[] data, final int from, final int to) {
        final double[] sliced = new double[to - from];
        System.arraycopy(data, from, sliced, 0, to - from);
        return sliced;
    }

	/**
	 * Apply this lag polynomial to a time series at the given index.
	 *
	 * @param timeSeries
	 *            the time series containing the index to apply this lag polynomial
	 *            to.
	 * @param index
	 *            the index of the series to apply the lag polynomial at.
	 * @return the result of applying this lag polynomial to the given time series
	 *         at the given index.
	 */
	public final double apply(final TimeSeries timeSeries, final int index) {
		double value = 0.0;
		for (int i = 0; i < coefficients.length; i++) {
			value += this.coefficients[i] * LagOperator.apply(timeSeries, index, i);
		}
		return value;
	}

	/**
	 * Apply this lag polynomial to a time series at the given date-time.
	 *
	 * @param timeSeries
	 *            the time series containing the index to apply this lag polynomial
	 *            to.
	 * @param lag
	 *            the lag of the series to apply the lag polynomial at.
	 * @return the result of applying this lag polynomial to the given time series
	 *         at the given date-time.
	 */
	public final double applyLag(final TimeSeries timeSeries, final int lag) {
		double value = 0.0;
		for (int i = 0; i < coefficients.length; i++) {
			value += this.coefficients[i] * LagOperator.applyLag(timeSeries, lag, i);
		}
		return value;
	}

	/**
	 * Apply this lag polynomial to the series at the given index, then solve for
	 * the expected value of the series at that index. If this is a moving average
	 * polynomial, then the time series should be a series of residuals.
	 *
	 * @param timeSeries
	 *            the time series containing the index to apply the lag polynomial
	 *            to.
	 * @param index
	 *            the index of the series to apply the lag polynomial at.
	 * @return the result of applying this lag polynomial to the time series at the
	 *         given index and solving for the expected value of the series at that
	 *         index.
	 */
	public double solve(final TimeSeries timeSeries, final int index) {
		double value = 0.0;
		for (int i = 0; i < parameters.length; i++) {
			value -= parameters[i] * LagOperator.apply(timeSeries, index, i + 1);
		}
		return value;
	}

	/**
	 * Apply this lag polynomial to the series at the given date-time, then solve
	 * for the expected value of the series at that date-time. If this is a moving
	 * average polynomial, then the time series should be a series of residuals.
	 *
	 * @param timeSeries
	 *            the time series containing the date-time to apply the lag
	 *            polynomial to.
	 * @param lag
	 *            the lag of the series to apply the lag polynomial at.
	 * @return the result of applying this lag polynomial to the time series at the
	 *         given date-time and solving for the expected value of the series at
	 *         that date-time.
	 */
	public double solveLag(final TimeSeries timeSeries, int lag) {
		double value = 0.0;
		for (int i = 0; i < parameters.length; i++) {
			value -= parameters[i] * LagOperator.applyLag(timeSeries, lag, i + 1);
		}
		return value;
	}

	/**
	 * Apply this lag polynomial to the series at the given index, then solve for
	 * the expected value of the series at that index. If this is a moving average
	 * polynomial, then the time series should be a series of residuals.
	 *
	 * @param timeSeries
	 *            the time series containing the index to apply the lag polynomial
	 *            to.
	 * @param index
	 *            the index of the series to apply the lag polynomial at.
	 * @return the result of applying this lag polynomial to the time series at the
	 *         given index and solving for the expected value of the series at that
	 *         index.
	 */
	public double solve(final double[] timeSeries, final int index) {
		double value = 0.0;
		for (int i = 0; i < Math.min(parameters.length, timeSeries.length); i++) {
			value -= parameters[i] * LagOperator.apply(timeSeries, index, i + 1);
		}
		return value;
	}

	/**
	 * Return the parameters of this lag polynomial.
	 *
	 * @return the parameters of this lag polynomial.
	 */
	public final double[] parameters() {
		return this.parameters.clone();
	}

	/**
	 * Return the additive inverse of the parameters of this lag polynomial.
	 *
	 * @return the additive inverse of the parameters of this lag polynomial.
	 */
	public final double[] inverseParams() {
		final double[] invParams = new double[parameters.length];
		for (int i = 0; i < invParams.length; i++) {
			invParams[i] = -parameters[i];
		}
		return invParams;
	}

	/**
	 * Return the coefficients of this lag polynomial, which includes the constant
	 * term 1.
	 *
	 * @return the coefficients of this lag polynomial.
	 */
	final double[] coefficients() {
		return this.coefficients.clone();
	}

	@Override
	public String toString() {
		final double epsilon = Math.ulp(1.0);
		StringBuilder builder = new StringBuilder();
		builder.append("1");
		for (int i = 1; i < coefficients.length - 1; i++) {
			if (Math.abs(coefficients[i]) > epsilon) {

				if (coefficients[i] < 0) {
					builder.append(" - ");
				} else {
					builder.append(" + ");
				}
				if (Math.abs(coefficients[i] - 1.0) > epsilon) {
					builder.append(Double.toString(Math.abs(coefficients[i])));
				}
				builder.append("L");
				if (i > 1) {
					builder.append("^").append(i);
				}
			}
		}
		final int lastIndex = coefficients.length - 1;
		if (coefficients[lastIndex] < 0) {
			builder.append(" - ");
		} else {
			builder.append(" + ");
		}
		if (coefficients.length > 1) {
			if (coefficients[lastIndex] != 1.0) {
				builder.append(Double.toString(Math.abs(coefficients[lastIndex])));
			}
			builder.append("L");
		}
		if (coefficients.length > 2) {
			builder.append("^").append(lastIndex);
		}
		return builder.toString();
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + Arrays.hashCode(coefficients);
		result = prime * result + degree;
		result = prime * result + Arrays.hashCode(parameters);
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		LagPolynomial other = (LagPolynomial) obj;
		if (!Arrays.equals(coefficients, other.coefficients))
			return false;
		return degree == other.degree && Arrays.equals(parameters, other.parameters);
	}
}
