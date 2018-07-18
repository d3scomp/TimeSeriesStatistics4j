package cz.cuni.mff.d3s.tss.arima;

public class MultipleLinearRegressionPrediction {

	private final double estimate;
	private final double fitStandardError;
	private final DoublePair confidenceInterval;
	private final DoublePair predictionInterval;

	MultipleLinearRegressionPrediction(final double estimate, final double fitStandardError,
			final DoublePair confidenceInterval, final DoublePair predictionInterval) {
		this.estimate = estimate;
		this.fitStandardError = fitStandardError;
		this.confidenceInterval = confidenceInterval;
		this.predictionInterval = predictionInterval;
	}

	public double fitStandardError() {
		return this.fitStandardError;
	}

	public DoublePair confidenceInterval() {
		return this.confidenceInterval;
	}

	public DoublePair predictionInterval() {
		return this.predictionInterval;
	}

	public double estimate() {
		return this.estimate;
	}
}
