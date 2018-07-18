package cz.cuni.mff.d3s.tss.arima;


import smile.stat.distribution.TDistribution;

public class StudentsT {

	private final TDistribution dist;
	private final int df;

	/**
	 * Create a new Student's t distribution with the given degrees of freedom.
	 *
	 * @param df
	 *            the degrees of freedom for this distribution.
	 */
	public StudentsT(final int df) {
		this.dist = new TDistribution(df);
		this.df = df;
	}

	public double rand() {
		return this.dist.rand();
	}

	public double quantile(final double prob) {
		return this.dist.quantile(prob);
	}

}
