package cz.cuni.mff.d3s.tss.arima;

import java.util.ArrayList;
import java.util.List;

public class MultipleLinearRegressionPredictor {

	private final LinearRegression model;
	private final MatrixOneD XtXInverse;
	private final int degreesOfFreedom;

	private MultipleLinearRegressionPredictor(LinearRegression model) {
		this.model = model;
		this.XtXInverse = Matrix.create(model.XtXInverse());
		this.degreesOfFreedom = model.response().length - model.designMatrix().length;
	}

	/**
	 * Create a new predictor from the given multiple linear regression model.
	 *
	 * @param model
	 *            the regression model to base predictions off of.
	 * @return a new predictor.
	 */
	public static MultipleLinearRegressionPredictor from(LinearRegression model) {
		return new MultipleLinearRegressionPredictor(model);
	}

	private DoublePair getInterval(double sampleEstimate, double tValue, double standardError) {
		double lowerValue = sampleEstimate - tValue * standardError;
		double upperValue = sampleEstimate + tValue * standardError;
		return new DoublePair(lowerValue, upperValue);
	}

	public MultipleLinearRegressionPrediction predict(Vector observation, double alpha) {
		double estimate = estimate(predictorWithIntercept(observation));
		double seFit = standardErrorFit(predictorWithIntercept(observation));
		DoublePair confidenceInterval = confidenceInterval(alpha, predictorWithIntercept(observation), estimate);
		DoublePair predictionInterval = predictionInterval(alpha, predictorWithIntercept(observation), estimate);
		return new MultipleLinearRegressionPrediction(estimate, seFit, confidenceInterval, predictionInterval);
	}

	private MultipleLinearRegressionPrediction predictWithIntercept(Vector vector, double alpha) {
		double estimate = estimate(vector);
		double seFit = standardErrorFit(vector);
		DoublePair confidenceInterval = confidenceInterval(alpha, vector, estimate);
		DoublePair predictionInterval = predictionInterval(alpha, vector, estimate);
		return new MultipleLinearRegressionPrediction(estimate, seFit, confidenceInterval, predictionInterval);
	}

	private double estimate(Vector data) {
		return data.dotProduct(Vector.from(model.beta()));
	}

	private Vector predictorWithIntercept(Vector newData) {
		if (model.hasIntercept()) {
			return newData.push(1.0);
		} else {
			return newData;
		}
	}

	private double standardErrorFit(Vector predictor) {
		double product = QuadraticForm.multiply(predictor, XtXInverse);
		return Math.sqrt(model.sigma2() * product);
	}

	private DoublePair confidenceInterval(double alpha, Vector predictor, double estimate) {
		StudentsT T = new StudentsT(this.degreesOfFreedom);
		double tValue = T.quantile(1 - (alpha / 2.0));
		// send in predictor instead of newData since predict method also updates for
		// intercept.
		double seFit = standardErrorFit(predictor);
		return getInterval(estimate, tValue, seFit);
	}

	private DoublePair predictionInterval(double alpha, Vector predictor, double estimate) {
		StudentsT T = new StudentsT(this.degreesOfFreedom);
		double tValue = T.quantile(1 - (alpha / 2.0));
		double seFit = standardErrorFit(predictor);
		double standardError = Math.sqrt(model.sigma2() + seFit * seFit);
		return getInterval(estimate, tValue, standardError);
	}

	public List<MultipleLinearRegressionPrediction> predict(Matrix observations, double alpha) {
		List<MultipleLinearRegressionPrediction> predictions = new ArrayList<>(observations.nrow());
		for (int i = 0; i < observations.nrow(); i++) {
			predictions.add(predict(observations.getRow(i), alpha));
		}
		return predictions;
	}

	/**
	 * Predict a series of responses, one for each row in the design matrix.
	 *
	 * @param designMatrix
	 *            the design matrix for the prediction.
	 * @param alpha
	 *            the significance level.
	 * @return a list of predictions, one for each row in the design matrix.
	 */
	public List<MultipleLinearRegressionPrediction> predictDesignMatrix(Matrix designMatrix, double alpha) {
		List<MultipleLinearRegressionPrediction> predictions = new ArrayList<>(designMatrix.nrow());
		for (int i = 0; i < designMatrix.nrow(); i++) {
			predictions.add(predictWithIntercept(designMatrix.getRow(i), alpha));
		}
		return predictions;
	}

}
