package cz.cuni.mff.d3s.tss.arima;

/**
 * The strategy to be used for fitting an ARIMA model.
 *
 * @author Jacob Rachiele
 */
public enum FittingStrategy {

	CSS("conditional sum-of-squares"),
	ML("maximum likelihood"),
	CSSML("conditional sum-of-squares, then maximum likelihood");

	private final String description;

	FittingStrategy(final String description) {
		this.description = description;
	}

	@Override
	public String toString() {
		return this.description;
	}
}
