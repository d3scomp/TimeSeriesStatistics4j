package cz.cuni.mff.d3s.tss.arima;

public class MultipleLinearRegressionBuilder {

	private double[][] predictors;
	private double[] response;
	private boolean hasIntercept = true;

	public double[][] getPredictors(){
		return predictors;
	}
	
	public double[] getResponse() {
		return response;
	}
	
	public boolean hasIntercept() {
		return hasIntercept;
	}
	
	public final MultipleLinearRegressionBuilder from(MultipleLinearRegression regression) {
		this.predictors = copy(regression.predictors());
		this.response = regression.response().clone();
		this.hasIntercept = regression.hasIntercept();
		return this;
	}
	
	public static double[][] copy(double[][] values) {
        double[][] copied = new double[values.length][];
        for (int i = 0; i < values.length; i++) {
            copied[i] = values[i].clone();
        }
        return copied;
    }

	
	public MultipleLinearRegressionBuilder predictors(double[]... predictors) {
		this.predictors = new double[predictors.length][];
		for (int i = 0; i < predictors.length; i++) {
			this.predictors[i] = predictors[i].clone();
		}
		return this;
	}

	
	public MultipleLinearRegressionBuilder response(double[] response) {
		this.response = response.clone();
		return this;
	}

	
	public MultipleLinearRegressionBuilder hasIntercept(boolean hasIntercept) {
		this.hasIntercept = hasIntercept;
		return this;
	}

	
	public MultipleLinearRegression build() {
		return new MultipleLinearRegression(this);
	}

}
