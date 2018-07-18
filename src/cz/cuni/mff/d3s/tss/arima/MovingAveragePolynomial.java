package cz.cuni.mff.d3s.tss.arima;

public class MovingAveragePolynomial extends LagPolynomial {
	/**
	 * Create a new moving average polynomial with the given parameters.
	 *
	 * @param parameters
	 *            the moving average parameters of the polynomial.
	 */
	MovingAveragePolynomial(final double... parameters) {
		super(parameters);
	}

	@Override
	public double solve(final TimeSeries residualSeries, final int index) {
		double value = 0.0;
		for (int i = 0; i < parameters.length; i++) {
			value += parameters[i] * LagOperator.apply(residualSeries, index, i + 1);
		}
		return value;
	}

	@Override
	public double solve(final double[] residualSeries, final int index) {
		double value = 0.0;
		for (int i = 0; i < Math.min(parameters.length, index); i++) {
			value += parameters[i] * LagOperator.apply(residualSeries, index, i + 1);
		}
		return value;
	}

	@Override
	public double solveLag(final TimeSeries timeSeries, int lag) {
		double value = 0.0;
		for (int i = 0; i < parameters.length; i++) {
			value += parameters[i] * LagOperator.applyLag(timeSeries, lag, i + 1);
		}
		return value;
	}
}
