package cz.cuni.mff.d3s.tss.arima;

/**
 * The parameters of an ARIMA model. The main difference between this class and {@link ArimaCoefficients} is that
 * the coefficients represent fixed, unchanging quantities that are either known or have been estimated,
 * whereas the parameters represent the coefficients before they are known, or before they have been fully estimated.
 * For this reason, the ArimaCoefficients class is immutable while the variables of this class may be updated after
 * they have been initialized.
 */
public class ArimaParameters {

    private static final double EPSILON = Math.ulp(1.0);

    private double[] autoRegressivePars;
    private double[] movingAveragePars;
    private double[] seasonalAutoRegressivePars;
    private double[] seasonalMovingAveragePars;
    private double mean = 0.0;
    private double intercept = 0.0;
    private double drift = 0.0;
    private double meanParScale = 1.0;
    private double interceptParScale = 1.0;
    private double driftParScale = 1.0;

    ArimaParameters(int numAR, int numMA, int numSAR, int numSMA) {
        this.autoRegressivePars = new double[numAR];
        this.movingAveragePars = new double[numMA];
        this.seasonalAutoRegressivePars = new double[numSAR];
        this.seasonalMovingAveragePars = new double[numSMA];
    }

    public ArimaParameters(double[] arCoeffs, double[] maCoeffs, double[] seasonalARCoeffs, double[] seasonalMACoeffs) {
    	this.autoRegressivePars = arCoeffs;
    	this.movingAveragePars = maCoeffs;
    	this.seasonalAutoRegressivePars = seasonalARCoeffs;
    	this.seasonalMovingAveragePars = seasonalMACoeffs;
	}

	double getScaledMean() {
        return this.mean / this.meanParScale;
    }

    double getScaledIntercept() {
        return this.intercept / this.interceptParScale;
    }

    double getScaledDrift() {
        return this.drift / this.driftParScale;
    }

    void setAndScaleMean(final double meanFactor) {
        this.mean = meanFactor * this.meanParScale;
    }

    void setAndScaleIntercept(final double interceptFactor) {
        this.intercept = interceptFactor * this.interceptParScale;
    }

    void setAndScaleDrift(final double driftFactor) {
        this.drift = driftFactor * this.driftParScale;
    }

    double[] getRegressors(final ArimaOrder order) {
        double[] regressors = new double[order.npar() - order.sumARMA()];
        if (order.constant().include()) {
            regressors[0] = this.mean;
        }
        if (order.drift().include()) {
            regressors[order.constant().asInt()] = this.drift;
        }
        return regressors;
    }

    double[] getAll(ArimaOrder order) {
        double[] pars = new double[order.npar()];
        System.arraycopy(autoRegressivePars, 0, pars, 0, autoRegressivePars.length);
        System.arraycopy(movingAveragePars, 0, pars, order.p(), movingAveragePars.length);
        System.arraycopy(seasonalAutoRegressivePars, 0, pars, order.p() + order.q(),
                         seasonalAutoRegressivePars.length);
        System.arraycopy(seasonalMovingAveragePars, 0, pars, order.p() + order.q() + order.P(),
                         seasonalMovingAveragePars.length);
        if (order.constant().include()) {
            pars[order.sumARMA()] = this.mean;
        }
        if (order.drift().include()) {
            pars[order.sumARMA() + order.constant().asInt()] = this.drift;
        }
        return pars;
    }

    double[] getAllScaled(ArimaOrder order) {
        double[] pars = new double[order.npar()];
        System.arraycopy(autoRegressivePars, 0, pars, 0, autoRegressivePars.length);
        System.arraycopy(movingAveragePars, 0, pars, order.p(), movingAveragePars.length);
        System.arraycopy(seasonalAutoRegressivePars, 0, pars, order.p() + order.q(),
                         seasonalAutoRegressivePars.length);
        System.arraycopy(seasonalMovingAveragePars, 0, pars, order.p() + order.q() + order.P(),
                         seasonalMovingAveragePars.length);
        if (order.constant().include()) {
            pars[order.sumARMA()] = this.mean / (this.meanParScale + EPSILON);
        }
        if (order.drift().include()) {
            pars[order.sumARMA() + order.constant().asInt()] = this.drift / (this.driftParScale + EPSILON);
        }
        return pars;
    }

    static ArimaParameters fromOrder(ArimaOrder order) {
        return initializePars(order.p(), order.q(), order.P(), order.Q());
    }
    
    static ArimaParameters fromCoefficients(ArimaCoefficients coefficients) {
        ArimaParameters parameters = new ArimaParameters(coefficients.arCoeffs(),
                                                         coefficients.maCoeffs(),
                                                         coefficients.seasonalARCoeffs(),
                                                         coefficients.seasonalMACoeffs());
        parameters.setMean(coefficients.mean());
        parameters.setIntercept(coefficients.intercept());
        parameters.setDrift(coefficients.drift());
        return parameters;
    }

    private void setIntercept(double intercept) {
		this.intercept = intercept;
	}

	static ArimaParameters initializePars(int numAR, int numMA, int numSAR, int numSMA) {
        return new ArimaParameters(numAR, numMA, numSAR, numSMA);
    }

	public void setMean(double mean) {
		this.mean = mean;
	}

	public double getMean() {
		return mean;
	}

	public void setMeanParScale(double meanParScale) {
		this.meanParScale = meanParScale;
	}

	public void setDrift(double drift) {
		this.drift = drift;
	}

	public double getDrift() {
		return drift;
	}

	public void setDriftParScale(double driftParScale) {
		this.driftParScale = driftParScale;
	}

	public double getMeanParScale() {
		return meanParScale;
	}

	public double getDriftParScale() {
		return driftParScale;
	}

	public void setAutoRegressivePars(double[] autoRegressivePars) {
		this.autoRegressivePars = autoRegressivePars;
	}

	public void setMovingAveragePars(double[] movingAveragePars) {
		this.movingAveragePars = movingAveragePars;
	}

	public void setSeasonalAutoRegressivePars(double[] seasonalAutoRegressivePars) {
		this.seasonalAutoRegressivePars = seasonalAutoRegressivePars;
	}

	public void setSeasonalMovingAveragePars(double[] seasonalMovingAveragePars) {
		this.seasonalMovingAveragePars = seasonalMovingAveragePars;
	}

	public double[] getAutoRegressivePars() {
		return autoRegressivePars;
	}

	public double[] getSeasonalAutoRegressivePars() {
		return seasonalAutoRegressivePars;
	}

	public double[] getMovingAveragePars() {
		return movingAveragePars;
	}

	public double[] getSeasonalMovingAveragePars() {
		return seasonalMovingAveragePars;
	}
}
