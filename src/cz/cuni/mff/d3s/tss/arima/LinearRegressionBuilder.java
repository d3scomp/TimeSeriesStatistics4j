package cz.cuni.mff.d3s.tss.arima;

public class LinearRegressionBuilder {

	private double[][] timeBasedPredictors = new double[0][0];
	private double[][] externalRegressors = new double[0][0];
	private TimeSeries response;
	private LinearRegression.Intercept intercept = LinearRegression.Intercept.INCLUDE;
	private LinearRegression.TimeTrend timeTrend = LinearRegression.TimeTrend.INCLUDE;
	private LinearRegression.Seasonal seasonal = LinearRegression.Seasonal.EXCLUDE;
	private int seasonalCycle = 0; // TODO: Parameterize

	/**
	 * Copy the attributes of the given regression object to this builder and return
	 * this builder.
	 *
	 * @param regression
	 *            the object to copy the attributes from.
	 * @return this builder.
	 */
	public final LinearRegressionBuilder from(LinearRegression regression) {
		this.externalRegressors = copy(regression.predictors());
		this.response = regression.timeSeriesResponse();
		this.intercept = regression.intercept();
		this.timeTrend = regression.timeTrend();
		this.seasonal = regression.seasonal();
		this.seasonalCycle = regression.seasonalCycle();
		return this;
	}
	
	public static double[][] copy(double[][] values) {
        double[][] copied = new double[values.length][];
        for (int i = 0; i < values.length; i++) {
            copied[i] = values[i].clone();
        }
        return copied;
    }

	/**
	 * Specify prediction variable data for the linear regression model. Note that
	 * if this method has already been called on this object, then the array of
	 * prediction variables will be <i>appended to</i> rather than overwritten. Each
	 * element of the two dimensional external regressors outer array is interpreted
	 * as a column vector of data for a single prediction variable.
	 *
	 * @param regressors
	 *            the external regressors to add to the regression model
	 *            specification.
	 * @return this builder.
	 */
	public LinearRegressionBuilder externalRegressors(double[]... regressors) {
		int currentCols = this.externalRegressors.length;
		int currentRows = 0;
		if (currentCols > 0) {
			currentRows = this.externalRegressors[0].length;
		} else if (regressors.length > 0) {
			currentRows = regressors[0].length;
		}
		double[][] newPredictors = new double[currentCols + regressors.length][currentRows];
		for (int i = 0; i < currentCols; i++) {
			System.arraycopy(this.externalRegressors[i], 0, newPredictors[i], 0, currentRows);
		}
		for (int i = 0; i < regressors.length; i++) {
			newPredictors[i + currentCols] = regressors[i].clone();
		}
		this.externalRegressors = newPredictors;
		return this;
	}

	/**
	 * Specify prediction variable data for the linear regression model. Note that
	 * if this method has already been called on this object, then the array of
	 * prediction variables will be <i>appended to</i> rather than overwritten. Each
	 * element of the two dimensional external predictors outer array is interpreted
	 * as a column vector of data for a single prediction variable.
	 *
	 * @param predictors
	 *            the external predictors to add to the regression model
	 *            specification.
	 * @return this builder.
	 */
	private LinearRegressionBuilder timeBasedPredictors(double[]... predictors) {
		int currentCols = this.timeBasedPredictors.length;
		int currentRows = 0;
		if (currentCols > 0) {
			currentRows = this.timeBasedPredictors[0].length;
		} else if (predictors.length > 0) {
			currentRows = predictors[0].length;
		}
		double[][] newPredictors = new double[currentCols + predictors.length][currentRows];
		for (int i = 0; i < currentCols; i++) {
			System.arraycopy(this.timeBasedPredictors[i], 0, newPredictors[i], 0, currentRows);
		}
		for (int i = 0; i < predictors.length; i++) {
			newPredictors[i + currentCols] = predictors[i].clone();
		}
		this.timeBasedPredictors = newPredictors;
		return this;
	}

	/**
	 * Specify prediction variable data for the linear regression model. Note that
	 * if this method has already been called on this object, then the matrix of
	 * prediction variables will be <i>appended to</i> rather than overwritten.
	 *
	 * @param regressors
	 *            the external regressors to add to the regression model
	 *            specification.
	 * @return this builder.
	 */
	public LinearRegressionBuilder externalRegressors(Matrix regressors) {
		externalRegressors(regressors.data2D(Matrix.Layout.BY_COLUMN));
		return this;
	}

	/**
	 * Specify the response, or dependent variable, in the form of a time series.
	 *
	 * @param response
	 *            the response, or dependent variable, in the form of a time series.
	 * @return this builder.
	 */
	public LinearRegressionBuilder response(TimeSeries response) {
		this.response = response;
		return this;
	}

	/**
	 * Specify whether to include an intercept in the regression model. The default
	 * is for an intercept to be included.
	 *
	 * @param intercept
	 *            whether or not to include an intercept in the model.
	 * @return this builder.
	 */
	public LinearRegressionBuilder hasIntercept(LinearRegression.Intercept intercept) {
		this.intercept = intercept;
		return this;
	}

	/**
	 * Specify whether to include a time trend in the regression model. The default
	 * is for a time trend to be included.
	 *
	 * @param timeTrend
	 *            whether or not to include a time trend in the model.
	 * @return this builder.
	 */
	public LinearRegressionBuilder timeTrend(LinearRegression.TimeTrend timeTrend) {
		this.timeTrend = timeTrend;
		return this;
	}

	/**
	 * Specify whether to include a seasonal component in the regression model. The
	 * default is for the seasonal component to be excluded.
	 *
	 * @param seasonal
	 *            whether or not to include a seasonal component in the model.
	 * @return this builder.
	 */
	public LinearRegressionBuilder seasonal(LinearRegression.Seasonal seasonal) {
		this.seasonal = seasonal;
		return this;
	}

	/**
	 * Specify the length of time it takes for the seasonal pattern to complete one
	 * cycle.
	 *
	 * @param seasonalCycle
	 *            the length of time it takes for the seasonal pattern to complete
	 *            one cycle. The default value for this property is one year.
	 * @return this builder.
	 */
	public LinearRegressionBuilder seasonalCycle(int seasonalCycle) {
		this.seasonalCycle = seasonalCycle;
		return this;
	}

	public LinearRegression build() {
		if (response == null) {
			throw new IllegalStateException(
					"A time series linear regression model " + "must have a non-null response variable.");
		}
		if (this.timeTrend.include()) {
			this.timeBasedPredictors(inclusiveRange(1, response.size()));
		}
		if (this.seasonal.include()) {
			int seasonalFrequency = seasonalCycle;
			int periodOffset = 0;
			double[][] seasonalRegressors = LinearRegression.getSeasonalRegressors(this.response.size(),
					seasonalFrequency, periodOffset);
			this.timeBasedPredictors(seasonalRegressors);
		}
		return new LinearRegression(this);
	}

	private double[] inclusiveRange(int from, int to) {
		int size = from - to;
		double[] range = new double[size];
		for(int i = 0; i < size; i++) {
			range[i] = (i+1) * from;
		}
		return range;
	}
	
	double[][] timeBasedPredictors() {
		return timeBasedPredictors;
	}

	double[][] externalRegressors() {
		return externalRegressors;
	}

	TimeSeries response() {
		return response;
	}

	LinearRegression.Intercept intercept() {
		return intercept;
	}

	LinearRegression.TimeTrend timeTrend() {
		return timeTrend;
	}

	LinearRegression.Seasonal seasonal() {
		return seasonal;
	}

	int seasonalCycle() {
		return seasonalCycle;
	}

}
