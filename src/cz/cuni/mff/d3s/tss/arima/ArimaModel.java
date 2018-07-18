package cz.cuni.mff.d3s.tss.arima;

import cz.cuni.mff.d3s.tss.TimeSeries;


/**
 * A numerical description of an ARIMA model.
 *
 */
public class ArimaModel {
	final double sigma2;
    private final double logLikelihood;
    private final double aic;
    private final double[] residuals;
    private final double[] fitted;

    /**
     * Create new model information with the given data.
     *
     * @param npar          the number of parameters estimated in the model.
     * @param sigma2        an estimate of the model variance.
     * @param logLikelihood the natural logarithms of the likelihood of the model parameters.
     * @param residuals     the difference between the observations and the fitted values.
     * @param fitted        the values fitted by the model to the data.
     */
    ArimaModel(final int npar, final double sigma2, final double logLikelihood,
                     final double[] residuals, final double[] fitted) {
        this.sigma2 = sigma2;
        this.logLikelihood = logLikelihood;
        this.aic = 2 * npar - 2 * logLikelihood;
        this.residuals = residuals.clone();
        this.fitted = fitted.clone();
    }

	public double[] getResiduals() {
		return residuals;
	}
	
	public double[] getFitted() {
		return fitted;
	}
}
