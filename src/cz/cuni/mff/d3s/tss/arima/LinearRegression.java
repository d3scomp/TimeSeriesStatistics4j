package cz.cuni.mff.d3s.tss.arima;

public class LinearRegression {
	/**
	 * Specifies whether a time series regression model has an intercept.
	 */
	enum Intercept {
		INCLUDE(1), EXCLUDE(0);

		private final int intercept;

		Intercept(final int intercept) {
			this.intercept = intercept;
		}

		boolean include() {
			return this == INCLUDE;
		}

		int asInt() {
			return this.intercept;
		}
	}

	/**
	 * Specifies whether a time series regression model has a time trend.
	 */
	enum TimeTrend {
		INCLUDE(1), EXCLUDE(0);

		private final int timeTrend;

		TimeTrend(final int timeTrend) {
			this.timeTrend = timeTrend;
		}

		boolean include() {
			return this == INCLUDE;
		}

		int asInt() {
			return this.timeTrend;
		}
	}

	/**
	 * Specifies whether a time series regression model has a seasonal component.
	 */
	enum Seasonal {
		INCLUDE(1), EXCLUDE(0);

		private final int seasonal;

		Seasonal(final int seasonal) {
			this.seasonal = seasonal;
		}

		boolean include() {
			return this == INCLUDE;
		}

		int asInt() {
			return this.seasonal;
		}
	}

	private final MultipleLinearRegression regression;
	private final TimeSeries timeSeries;
	private final int seasonalCycle;
	private final Intercept intercept;
	private final TimeTrend timeTrend;
	private final Seasonal seasonal;
	private final double[][] externalRegressors;

	LinearRegression(LinearRegressionBuilder timeSeriesRegressionBuilder) {
		this.timeSeries = timeSeriesRegressionBuilder.response();
		this.seasonalCycle = timeSeriesRegressionBuilder.seasonalCycle();
		this.externalRegressors = timeSeriesRegressionBuilder.externalRegressors();
		double[][] allPredictors = combine(timeSeriesRegressionBuilder.timeBasedPredictors(),
				timeSeriesRegressionBuilder.externalRegressors());
		MultipleLinearRegressionBuilder regressionBuilder = new MultipleLinearRegressionBuilder();
		regressionBuilder.hasIntercept(timeSeriesRegressionBuilder.intercept().include()).predictors(allPredictors)
				.response(timeSeries.asArray());
		this.regression = regressionBuilder.build();
		this.intercept = timeSeriesRegressionBuilder.intercept();
		this.timeTrend = timeSeriesRegressionBuilder.timeTrend();
		this.seasonal = timeSeriesRegressionBuilder.seasonal();
	}

	public double[][] predictors() {
		return copy(this.externalRegressors);
	}

	public double[][] copy(double[][] values) {
		double[][] copied = new double[values.length][];
		for (int i = 0; i < values.length; i++) {
			copied[i] = values[i].clone();
		}
		return copied;
	}

	public double[][] XtXInverse() {
		return this.regression.XtXInverse();
	}

	public double[][] designMatrix() {
		return this.regression.designMatrix();
	}

	public double[] response() {
		return regression.response();
	}

	public double[] beta() {
		return regression.beta();
	}

	public double[] standardErrors() {
		return regression.standardErrors();
	}

	public double[] fitted() {
		return regression.fitted();
	}

	public TimeSeriesRegressionForecast forecast(int steps, double alpha) {
		MultipleLinearRegressionPredictor predictor = MultipleLinearRegressionPredictor.from(this);
		Vector beta = Vector.from(this.beta());
		Matrix predictionMatrix = getPredictionMatrix(steps);
		TimeSeriesRegressionForecaster forecaster = new TimeSeriesRegressionForecaster(this.timeSeries, predictor, beta, predictionMatrix);
		return forecaster.forecast(steps, alpha);

	}

	public TimeSeries observations() {
		return this.timeSeriesResponse();
	}

	public TimeSeries fittedSeries() {
		return null;
	}

	public double sigma2() {
		return regression.sigma2();
	}

	public boolean hasIntercept() {
		return regression.hasIntercept();
	}

	public int seasonalCycle() {
		return this.seasonalCycle;
	}

	public TimeSeries timeSeriesResponse() {
		return this.timeSeries;
	}

	public Intercept intercept() {
		return this.intercept;
	}

	public TimeTrend timeTrend() {
		return this.timeTrend;
	}

	public Seasonal seasonal() {
		return this.seasonal;
	}

	public int seasonalFrequency() {
		return this.timeSeries.getTimePeriod();
	}

	private static double[] getIthSeasonalRegressor(int nrows, int startRow, int seasonalFrequency) {
		double[] regressor = new double[nrows];
		for (int j = 0; j < regressor.length - startRow; j += seasonalFrequency) {
			regressor[j + startRow] = 1.0;
		}
		return regressor;
	}

	// What we are doing here is equivalent to how R handles "factors" in linear
	// regression model.
	static double[][] getSeasonalRegressors(int nrows, int seasonalFrequency, int periodOffset) {
		int ncols = seasonalFrequency - 1;
		double[][] seasonalRegressors = new double[ncols][nrows];
		for (int i = 0; i < ncols; i++) {
			// Apparently, the "modulus" operator in Java is not actually a modulus
			// operator, but a remainder operator.
			// floorMod was added in Java 8 to give results one would expect when doing
			// modular arithmetic.
			int startRow = Math.floorMod(i + 1 - periodOffset, seasonalFrequency);
			seasonalRegressors[i] = getIthSeasonalRegressor(nrows, startRow, seasonalFrequency);
		}
		return seasonalRegressors;
	}

	private Matrix getPredictionMatrix(int steps) {
		int intercept = this.intercept().asInt();
		int timeTrend = this.timeTrend().asInt();
		int seasonal = this.seasonal().asInt();
		int seasonalFrequency = this.seasonalFrequency();
		int ncols = intercept + timeTrend + (seasonalFrequency - 1) * seasonal;

		double[][] designMatrix = new double[ncols][steps];
		if (this.intercept().include()) {
			designMatrix[0] = fill(steps, 1.0);
		}
		if (this.timeTrend().include()) {
			int startTime = this.response().length + 1;
			int endTime = startTime + steps;
			designMatrix[intercept] = exclusiveRange(startTime, endTime);
		}
		if (this.seasonal().include()) {
			int periodOffset = this.response().length % seasonalFrequency;
			double[][] seasonalMatrix = getSeasonalRegressors(steps, seasonalFrequency, periodOffset);
			for (int i = 0; i < seasonalMatrix.length; i++) {
				designMatrix[i + intercept + timeTrend] = seasonalMatrix[i];
			}
		}
		return Matrix.create(Matrix.Layout.BY_COLUMN, designMatrix);
	}
	
	private double[] exclusiveRange(int from, int to) {
		int size = to - from;
		double[] range = new double[size];
		for(int i = 0; i < size; i++) {
			range[i] = from + i;
		}
		return range;
	}
	
	/**
     * Create and return a new array of the given size with every value set to the given value.
     *
     * @param size  the number of elements of the new array.
     * @param value the value to fill every element of the array with.
     * @return a new array of the given size with every value set to the given value.
     */
    public static double[] fill(final int size, final double value) {
        final double[] filled = new double[size];
        for (int i = 0; i < filled.length; i++) {
            filled[i] = value;
        }
        return filled;
    }

	/**
	 * Create a new array by combining the elements of the input arrays in the order
	 * given.
	 *
	 * @param twoDArrays
	 *            the arrays to combine.
	 * @return a new array formed by combining the elements of the input arrays.
	 */
	public static double[][] combine(double[][]... twoDArrays) {
		int newArrayLength = 0;
		for (double[][] twoDArray : twoDArrays) {
			newArrayLength += twoDArray.length;
		}
		double[][] newArray = new double[newArrayLength][];
		int i = 0;
		for (double[][] twoDArray : twoDArrays) {
			for (double[] array : twoDArray) {
				newArray[i] = array.clone();
				i++;
			}
		}
		return newArray;
	}
}
