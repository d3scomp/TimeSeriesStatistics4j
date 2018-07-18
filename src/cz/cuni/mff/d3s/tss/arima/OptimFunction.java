package cz.cuni.mff.d3s.tss.arima;

public class OptimFunction {

	private final TimeSeries observations;
	private final ArimaOrder order;
	private final ArimaParameters parameters;
	private final FittingStrategy fittingStrategy;
	private final int seasonalFrequency;
	private final Matrix externalRegressors;

	private static final double gradientTolerance = 1E-3;

	protected int functionEvaluations = 0;
	protected int gradientEvalutations = 0;

	public Vector gradientAt(Vector point) {
		gradientEvalutations++;
		return NumericalDerivatives.centralDifferenceGradient(this, point, gradientTolerance);
	}

	public Vector gradientAt(final Vector point, final double functionValue) {
		gradientEvalutations++;
		return NumericalDerivatives.forwardDifferenceGradient(this, point, gradientTolerance * gradientTolerance,
				functionValue);
	}

	/**
	 * The number of times this function has been evaluated.
	 *
	 * @return the number of times this function has been evaluated.
	 */
	public int functionEvaluations() {
		return this.functionEvaluations;
	}

	/**
	 * The number of times the gradient has been computed.
	 *
	 * @return the number of times the gradient has been computed.
	 */
	public int gradientEvaluations() {
		return this.gradientEvalutations;
	}

	public OptimFunction(TimeSeries observations, ArimaOrder order, ArimaParameters parameters,
			FittingStrategy fittingStrategy, Matrix externalRegressors, int seasonalFrequency) {
		this.observations = observations;
		this.order = order;
		this.parameters = parameters;
		this.fittingStrategy = fittingStrategy;
		this.externalRegressors = externalRegressors;
		this.seasonalFrequency = seasonalFrequency;
	}

	public final double at(final Vector point) {
		functionEvaluations++;

		final double[] params = point.elements();
		parameters.setAutoRegressivePars(slice(params, 0, order.p()));
		parameters.setMovingAveragePars(slice(params, order.p(), order.p() + order.q()));
		parameters
				.setSeasonalAutoRegressivePars(slice(params, order.p() + order.q(), order.p() + order.q() + order.P()));
		parameters.setSeasonalMovingAveragePars(
				slice(params, order.p() + order.q() + order.P(), order.p() + order.q() + order.P() + order.Q()));

		if (order.constant().include()) {
			parameters.setAndScaleMean(params[order.sumARMA()]);
		}
		if (order.drift().include()) {
			parameters.setAndScaleDrift(params[order.sumARMA() + order.constant().asInt()]);
		}
		final double[] arCoeffs = ArimaCoefficients.expandArCoefficients(parameters.getAutoRegressivePars(),
				parameters.getSeasonalAutoRegressivePars(), seasonalFrequency);
		final double[] maCoeffs = ArimaCoefficients.expandMaCoefficients(parameters.getMovingAveragePars(),
				parameters.getSeasonalMovingAveragePars(), seasonalFrequency);

		Vector regressionParameters = Vector.from(parameters.getRegressors(order));
		Vector regressionEffects = externalRegressors.times(regressionParameters);
		TimeSeries armaSeries = this.observations.minus(regressionEffects.elements());

		if (fittingStrategy == FittingStrategy.ML || fittingStrategy == FittingStrategy.CSSML) {
			double[] delta = getDelta(this.order, this.seasonalFrequency);
			ArimaKalmanFilter.KalmanOutput output = Arima.kalmanFit(armaSeries, arCoeffs, maCoeffs, delta);
			return 0.5 * (Math.log(output.sigma2()) + output.sumLog() / output.n());
		}

		TimeSeries differencedSeries = armaSeries.difference(1, order.d()).difference(seasonalFrequency, order.D());
		final ArimaModel info = Arima.fitCSS(differencedSeries, arCoeffs, maCoeffs, order.npar());
		return 0.5 * Math.log(info.sigma2);
	}

	private static double[] getDelta(ArimaOrder order, int observationFrequency) {
		LagPolynomial differencesPolynomial = LagPolynomial.differences(order.d());
		LagPolynomial seasonalDifferencesPolynomial = LagPolynomial.seasonalDifferences(observationFrequency,
				order.D());

		final LagPolynomial finalPolynomial = differencesPolynomial.times(seasonalDifferencesPolynomial);
		return scale(finalPolynomial.parameters(), -1.0);
	}

	/**
	 * Scale the original data by alpha and return the result in a new array.
	 *
	 * @param original
	 *            the data to be scaled.
	 * @param alpha
	 *            the scaling factor.
	 * @return the original data scaled by alpha.
	 */
	public static double[] scale(final double[] original, final double alpha) {
		final double[] scaled = new double[original.length];
		for (int i = 0; i < original.length; i++) {
			scaled[i] = original[i] * alpha;
		}
		return scaled;
	}

	/**
	 * Return a slice of the data between the given indices.
	 *
	 * @param data
	 *            the data to slice.
	 * @param from
	 *            the starting index.
	 * @param to
	 *            the ending index. The value at this index is excluded from the
	 *            result.
	 * @return a slice of the data between the given indices.
	 */
	public static double[] slice(final double[] data, final int from, final int to) {
		final double[] sliced = new double[to - from];
		System.arraycopy(data, from, sliced, 0, to - from);
		return sliced;
	}

}
