package cz.cuni.mff.d3s.tss.arima;

import java.util.ArrayList;
import java.util.List;

import cz.cuni.mff.d3s.tss.StudentsDistribution;
import cz.cuni.mff.d3s.tss.arima.ArimaKalmanFilter.KalmanOutput;

public class Arima {
	
    private static final double EPSILON = Math.ulp(1.0);
    private static final double DEFAULT_TOLERANCE = Math.sqrt(EPSILON);
    
    private final TimeSeries observations;
    private final TimeSeries differencedSeries;
    public final TimeSeries fittedSeries;
    private final TimeSeries residuals;
    private final ArimaOrder order;
    private final ArimaModel modelInfo;
    private final ArimaCoefficients coefficients;
    private final FittingStrategy fittingStrategy;

	private final int seasonalFrequency;
    private final double[] stdErrors;
    private final double[] arSarCoeffs;
    private final double[] maSmaCoeffs;

	public Arima(final TimeSeries observations, final ArimaOrder order, final int seasonalCycle,
            final FittingStrategy fittingStrategy, LinearRegression regression) {
		this.observations = observations;
        this.order = order;
        this.fittingStrategy = fittingStrategy;
        this.seasonalFrequency = seasonalCycle / observations.getTimePeriod();
        validateFreq(order, seasonalFrequency);
        this.differencedSeries = observations.difference(1, order.d()).difference(seasonalFrequency, order.D());

		final Vector initParams;
		final MatrixOneD initHessian;
		ArimaParameters parameters = ArimaParameters.initializePars(order.p(), order.q(), order.P(), order.Q());
		Matrix regressionMatrix = getRegressionMatrix(observations.size(), order);

		if (regression == null) {
			regression = getLinearRegression(differencedSeries, regressionMatrix);
		}

		if (order.constant().include()) {
			parameters.setMean(regression.beta()[0]);
			parameters.setMeanParScale(10 * regression.standardErrors()[0]);
		}
		if (order.drift().include()) {
			parameters.setDrift(regression.beta()[order.constant().asInt()]);
			parameters.setDriftParScale(10 * regression.standardErrors()[order.constant().asInt()]);
		}
		if (fittingStrategy == FittingStrategy.CSSML) {
			final FittingStrategy subStrategy = FittingStrategy.CSS;
			final Arima firstModel = new Arima(observations, order, seasonalCycle, subStrategy, regression);
			double meanParScale = parameters.getMeanParScale();
			double driftParScale = parameters.getDriftParScale();
			parameters = ArimaParameters.fromCoefficients(firstModel.coefficients());
			parameters.setMeanParScale(meanParScale);
			parameters.setDriftParScale(driftParScale);
			// parameters.setMean(firstModel.coefficients().mean());
			// parameters.setDrift(firstModel.coefficients().drift());
			initParams = Vector.from(parameters.getAllScaled(order));
			initHessian = getInitialHessian(firstModel);
		} else {
			initParams = Vector.from(parameters.getAllScaled(order));
			initHessian = getInitialHessian(initParams.size());
		}

		final OptimFunction function = new OptimFunction(observations, order, parameters,
				fittingStrategy, regressionMatrix, seasonalFrequency);
		final BFGS optimizer = new BFGS(function, initParams, DEFAULT_TOLERANCE, DEFAULT_TOLERANCE, initHessian);
		final Vector optimizedParams = optimizer.parameters();
		final MatrixOneD inverseHessian = optimizer.inverseHessian();

		this.stdErrors = sqrt(scale(inverseHessian.diagonal(), 1.0 / differencedSeries.size()));
		if (order.constant().include()) {
			this.stdErrors[order.sumARMA()] *= parameters.getMeanParScale();
		}
		if (order.drift().include()) {
			this.stdErrors[order.sumARMA() + order.constant().asInt()] *= parameters.getDriftParScale();
		}

		final double[] arCoeffs = getArCoeffs(optimizedParams);
		final double[] maCoeffs = getMaCoeffs(optimizedParams);
		final double[] sarCoeffs = getSarCoeffs(optimizedParams);
		final double[] smaCoeffs = getSmaCoeffs(optimizedParams);

		if (order.constant().include()) {
			parameters.setAndScaleMean(optimizedParams.at(order.sumARMA()));
		}
		if (order.drift().include()) {
			parameters.setAndScaleDrift(optimizedParams.at(order.sumARMA() + order.constant().asInt()));
		}
		this.coefficients = new ArimaCoefficients(arCoeffs, maCoeffs, sarCoeffs, smaCoeffs, order.d(), order.D(),
				parameters.getMean(), parameters.getDrift(), this.seasonalFrequency);
		this.arSarCoeffs = this.coefficients.getAllAutoRegressiveCoefficients();
		this.maSmaCoeffs = this.coefficients.getAllMovingAverageCoefficients();
		Vector regressionParameters = Vector.from(parameters.getRegressors(order));
		Vector regressionEffects = regressionMatrix.times(regressionParameters);
		TimeSeries armaSeries = this.observations.minus(regressionEffects.elements());
		TimeSeries differencedSeries = armaSeries.difference(1, order.d()).difference(seasonalFrequency, order.D());
		if (fittingStrategy == FittingStrategy.CSS) {
			this.modelInfo = fitCSS(differencedSeries, arSarCoeffs, maSmaCoeffs, order.npar());
			final double[] residuals = combine(new double[order.d() + order.D() * seasonalFrequency],
					modelInfo.getResiduals());
			this.fittedSeries = observations.minus(TimeSeries.from(residuals, observations.getObservationTimes()));
			this.residuals = observations.minus(this.fittedSeries);
		} else {
			double[] delta = getDelta(this.order, this.seasonalFrequency);
			this.modelInfo = fitML(armaSeries, arSarCoeffs, maSmaCoeffs, delta, order.npar());
			final double[] residuals = modelInfo.getResiduals();
			this.fittedSeries = observations.minus(TimeSeries.from(residuals, observations.getObservationTimes()));
			this.residuals = observations.minus(this.fittedSeries);
		}
	}
	
	/**
     * Take the square root of each element of the given array and return the result in a new array.
     *
     * @param data the data to take the square root of.
     * @return a new array containing the square root of each element.
     */
    public static double[] sqrt(final double... data) {
        final double[] sqrtData = new double[data.length];
        for (int i = 0; i < sqrtData.length; i++) {
            sqrtData[i] = Math.sqrt(data[i]);
        }
        return sqrtData;
    }
    
    /**
     * Scale the original data by alpha and return the result in a new array.
     *
     * @param original the data to be scaled.
     * @param alpha    the scaling factor.
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
	 * Fit an ARIMA model using conditional sum-of-squares.
	 *
	 * @param differencedSeries
	 *            the time series of observations to model.
	 * @param arCoeffs
	 *            the autoregressive coefficients of the model.
	 * @param maCoeffs
	 *            the moving-average coefficients of the model.
	 * @param npar
	 *            the order of the model to be fit.
	 * @return information about the fitted model.
	 */
	public static ArimaModel fitCSS(final TimeSeries differencedSeries, final double[] arCoeffs, final double[] maCoeffs,
			final int npar) {
		final int offset = arCoeffs.length;
		final int n = differencedSeries.size();

		final double[] fitted = new double[n];
		final double[] residuals = new double[n];

		for (int t = offset; t < fitted.length; t++) {
			// fitted[t] = mean;
			for (int i = 0; i < arCoeffs.length; i++) {
				if (Math.abs(arCoeffs[i]) > 0.0) {
					fitted[t] += arCoeffs[i] * differencedSeries.at(t - i - 1);
				}
			}
			for (int j = 0; j < Math.min(t, maCoeffs.length); j++) {
				if (Math.abs(maCoeffs[j]) > 0.0) {
					fitted[t] += maCoeffs[j] * residuals[t - j - 1];
				}
			}
			residuals[t] = differencedSeries.at(t) - fitted[t];
		}
		final int m = differencedSeries.size() - arCoeffs.length;
		final double sigma2 = sumOfSquared(residuals) / m;
		final double logLikelihood = (-n / 2.0) * (Math.log(2 * Math.PI * sigma2) + 1);
		return new ArimaModel(npar, sigma2, logLikelihood, residuals, fitted);
	}

	private static ArimaModel fitML(final TimeSeries observations, final double[] arCoeffs, final double[] maCoeffs,
			final double[] delta, int npar) {
		ArimaKalmanFilter.KalmanOutput output = kalmanFit(observations, arCoeffs, maCoeffs, delta);
		final double sigma2 = output.sigma2();
		final double logLikelihood = output.logLikelihood();
		final double[] residuals = output.residuals();
		final double[] fitted = differenceOf(observations.asArray(), residuals);
		npar += 1; // Add 1 for the variance estimate.
		return new ArimaModel(npar, sigma2, logLikelihood, residuals, fitted);
	}
	
	private MatrixOneD getInitialHessian(final Arima model) {
        double[] stdErrors = model.stdErrors;
        MatrixOneD.IdentityBuilder builder = new MatrixOneD.IdentityBuilder(stdErrors.length);
        for (int i = 0; i < stdErrors.length; i++) {
            builder.set(i, i, stdErrors[i] * stdErrors[i] * observations.size());
        }
        return builder.build();
    }
	
	private MatrixOneD getInitialHessian(final int n) {
        return MatrixOneD.identity(n);
	}

	public static KalmanOutput kalmanFit(final TimeSeries observations, final double[] arCoeffs, final double[] maCoeffs,
			final double[] delta) {
		ArimaStateSpace ss = new ArimaStateSpace(observations.asArray(), arCoeffs, maCoeffs, delta);
		ArimaKalmanFilter kalmanFilter = new ArimaKalmanFilter(ss);
		return kalmanFilter.output();
	}

	public static double sumOfSquared(final double... data) {
		return sumOf(squared(data));
	}
	
	public static double sumOfSquared(final List<Integer> data) {
		return sumOf(squared(data));
	}
	
	public static double sumOfMultiplied(final double[] data, final List<Integer> times) {
		return sumOf(multiplied(data, times));
	}

	static double[] squared(final double... data) {
		final double[] squared = new double[data.length];
		for (int i = 0; i < squared.length; i++) {
			squared[i] = data[i] * data[i];
		}
		return squared;
	}
	
	static List<Integer> squared(final List<Integer> data){
		final List<Integer> squared = new ArrayList<>(data);
		for(int i = 0; i < squared.size(); i++) {
			squared.set(i, data.get(i) * data.get(i));
		}
		return squared;
	}
	
	static double[] multiplied(final double[] data, final List<Integer> times) {
		final double[] multiplied = new double[data.length];
		for(int i = 0; i < data.length; i++) {
			multiplied[i] = data[i] * times.get(i);
		}
		return multiplied;
	}

	public static double sumOf(final double... data) {
		double sum = 0.0;
		for (double element : data) {
			sum += element;
		}
		return sum;
	}
	
	public static double sumOf(final List<Integer> data) {
		double sum = 0.0;
		for(Integer i : data) {
			sum += i;
		}
		return sum;
	}

	/**
	 * Take the element-by-element difference of the two arrays and return the
	 * result in a new array.
	 *
	 * @param left
	 *            the first array to take the difference with.
	 * @param right
	 *            the second array to take the difference with.
	 * @return the element-by-element difference of the two arrays.
	 */
	public static double[] differenceOf(final double[] left, final double[] right) {
		if (left.length != right.length) {
			throw new IllegalArgumentException("The data arrays must have the same length.");
		}
		final double[] difference = new double[left.length];
		for (int i = 0; i < left.length; i++) {
			difference[i] = left[i] - right[i];
		}
		return difference;
	}

	private void validateFreq(ArimaOrder order, int seasonalFrequency) {
		if (seasonalFrequency < 1) {
			String errorMessage = "The number of observations per seasonal cycle should be an integer"
					+ " greater than or equal to 1, but was " + seasonalFrequency;
			throw new IllegalArgumentException(errorMessage);
		}
		if (seasonalFrequency == 1) {
			int seasonalComponents = order.P() + order.Q() + order.D();
			if (seasonalComponents > 0) {
				String errorMessage = "There was a seasonal component in the model, but the number of "
						+ "observations per seasonal cycle was equal to 1.";
				throw new IllegalArgumentException(errorMessage);
			}
		}
	}

	private Matrix getRegressionMatrix(int size, ArimaOrder order) {
		double[][] matrix = new double[order.numRegressors()][size];
		if (order.constant().include()) {
			matrix[0] = fill(size, 1.0);
		}
		if (order.drift().include()) {
			matrix[order.constant().asInt()] = inclusiveRange(1, size);
		}
		return Matrix.create(Matrix.Layout.BY_COLUMN, matrix);
	}

	private double[] inclusiveRange(int from, int to) {
		int size = to - from;
		double[] range = new double[size + 1];
		for (int i = 0; i <= size; i++) {
			range[i] = from + i;
		}
		return range;
	}

	/**
	 * Create and return a new array of the given size with every value set to the
	 * given value.
	 *
	 * @param size
	 *            the number of elements of the new array.
	 * @param value
	 *            the value to fill every element of the array with.
	 * @return a new array of the given size with every value set to the given
	 *         value.
	 */
	public static double[] fill(final int size, final double value) {
		final double[] filled = new double[size];
		for (int i = 0; i < filled.length; i++) {
			filled[i] = value;
		}
		return filled;
	}

	private LinearRegression getLinearRegression(TimeSeries differencedSeries, Matrix designMatrix) {
		double[][] diffedMatrix = new double[designMatrix.ncol()][];
		double[][] designMatrixTwoD = designMatrix.data2D(Matrix.Layout.BY_COLUMN);
		for (int i = 0; i < diffedMatrix.length; i++) {
			diffedMatrix[i] = TimeSeries.difference(designMatrixTwoD[i], order.d());
		}
		for (int i = 0; i < diffedMatrix.length; i++) {
			diffedMatrix[i] = TimeSeries.difference(diffedMatrix[i], seasonalFrequency, order.D());
		}
		LinearRegressionBuilder regressionBuilder = new LinearRegressionBuilder();
		regressionBuilder.response(differencedSeries);
		regressionBuilder.hasIntercept(LinearRegression.Intercept.EXCLUDE);
		regressionBuilder.timeTrend(LinearRegression.TimeTrend.EXCLUDE);
		regressionBuilder.externalRegressors(Matrix.create(Matrix.Layout.BY_COLUMN, diffedMatrix));
		return regressionBuilder.build();
	}
	
	public ArimaCoefficients coefficients() {
        return this.coefficients;
    }
	

    private double[] getSarCoeffs(final Vector optimizedParams) {
        final double[] sarCoeffs = new double[order.P()];
        for (int i = 0; i < order.P(); i++) {
            sarCoeffs[i] = optimizedParams.at(i + order.p() + order.q());
        }
        return sarCoeffs;
    }

    private double[] getSmaCoeffs(final Vector optimizedParams) {
        final double[] smaCoeffs = new double[order.Q()];
        for (int i = 0; i < order.Q(); i++) {
            smaCoeffs[i] = optimizedParams.at(i + order.p() + order.q() + order.P());
        }
        return smaCoeffs;
    }

    private double[] getArCoeffs(final Vector optimizedParams) {
        final double[] arCoeffs = new double[order.p()];
        for (int i = 0; i < order.p(); i++) {
            arCoeffs[i] = optimizedParams.at(i);
        }
        return arCoeffs;
    }

    private double[] getMaCoeffs(final Vector optimizedParams) {
        final double[] arCoeffs = new double[order.q()];
        for (int i = 0; i < order.q(); i++) {
            arCoeffs[i] = optimizedParams.at(i + order.p());
        }
        return arCoeffs;
    }
    
    private static double[] getDelta(ArimaOrder order, int observationFrequency) {
        LagPolynomial differencesPolynomial = LagPolynomial.differences(order.d());
        LagPolynomial seasonalDifferencesPolynomial = LagPolynomial.seasonalDifferences(observationFrequency, order.D());

        final LagPolynomial finalPolynomial = differencesPolynomial.times(seasonalDifferencesPolynomial);
        return scale(finalPolynomial.parameters(), -1.0);
    }
    
    /**
     * Create a new array by combining the elements of the input arrays in the order given.
     *
     * @param arrays the arrays to combine.
     * @return a new array formed by combining the elements of the input arrays.
     */
    public static double[] combine(double[]... arrays) {
        int newArrayLength = 0;
        for (double[] array : arrays) {
            newArrayLength += array.length;
        }
        double[] newArray = new double[newArrayLength];
        newArrayLength = 0;
        for (double[] array : arrays) {
            System.arraycopy(array, 0, newArray, newArrayLength, array.length);
            newArrayLength += array.length;
        }
        return newArray;
    }
    
    /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
    
    public ArimaModel getFittedModel() {
    	return modelInfo;
    }
    
    public StudentsDistribution getMean() {
		int sampleCnt = (int) sumOf(fittedSeries.series);
		double mean = fittedSeries.getMean();
		double variance = computeMeanVariance(sampleCnt);

		return new StudentsDistribution(sampleCnt-1, mean, variance);
	}
    
    public int getSampleCnt() {
    	return fittedSeries.getSampleCnt();
    }
    
    public double getVariance() {
    	return modelInfo.sigma2;
    }
    
    public StudentsDistribution getLrb() {
		int sampleCnt = fittedSeries.getSampleCnt();
		double a = computeLraMean(sampleCnt);
		double b = computeLrbMean(sampleCnt);
		double e2 = computeEpsilonSquaredSum(a, b, sampleCnt);
		double variance = computeLrbVariance(sampleCnt, e2);

		// StudentsDistribution distribution = new StudentsDistribution(model.getSampleCnt() - 1, model.getMean(), model.getVariance());
		
		return new StudentsDistribution(sampleCnt-2, b, variance);
	}
    
    private double computeMeanVariance(int n) {
		if (n <= 1) {
			return Double.NaN;
		}

		double x2 = sumOfSquared(fittedSeries.series);
		double x = sumOf(fittedSeries.series);

		return (x2 - x*x/n) / ((n - 1) * n);
	}
    
    private double computeLraMean(int n) {
		if (n <= 0) {
			return Double.NaN;
		}

		double y = sumOf(fittedSeries.series);
		double x = sumOf(fittedSeries.getObservationTimes());
		double b = computeLrbMean(n);

		return (y - b * x) / n;
	}
    
    private double computeLrbMean(int n) {
		if (n <= 0) {
			return Double.NaN;
		}

		double x = sumOf(fittedSeries.getObservationTimes());
		double x2 = sumOfSquared(fittedSeries.getObservationTimes());
		double y = sumOf(fittedSeries.series);
		double xy = sumOfMultiplied(fittedSeries.series, fittedSeries.getObservationTimes());

		double nom = xy - x*y / n;
		double denom = x2 - x*x / n;

		return denom != 0 ? nom / denom : Double.NaN;
	}
    
    private double computeEpsilonSquaredSum(double a, double b, int n) {
		if (n <= 2) {
			return Double.NaN;
		}

		double x = sumOf(fittedSeries.getObservationTimes());
		double x2 = sumOfSquared(fittedSeries.getObservationTimes());
		double y = sumOf(fittedSeries.series);
		double y2 = sumOfSquared(fittedSeries.series);
		double xy = sumOfMultiplied(fittedSeries.series, fittedSeries.getObservationTimes());

		return y2 + n*a*a + b*b*x2 - 2*a*y - 2*b*xy + 2*a*b*x;
	}
    
    private double computeLrbVariance(int n, double e2) {
		if (n <= 0) {
			return Double.NaN;
		}

		double x = sumOf(fittedSeries.getObservationTimes());;
		double x2 = sumOfSquared(fittedSeries.getObservationTimes());

		double nom = e2 / (n - 2);
		double denom = x2 - x*x/n;
		return denom != 0 ? nom / denom : Double.NaN;
	}


}
