package cz.cuni.mff.d3s.tss;


public class StudentsDistribution {
	
	private final int df; // degrees of freedom
	private final double mean;
	private final double variance;
	private TTable.ALPHAS a;
	
	private static int DEFAULT_ALPHA_INDEX = 0;
	
	
	public StudentsDistribution(int df, double mean, double variance) {
		this.df = df;
		this.mean = mean;
		this.variance = variance;
		
		resetToDefaultAlpha();
	}
	
	public boolean isLessOrEqualTo(double threshold, TTable.ALPHAS confidence) {
		if (df < 1) {
			// accept hypothesis if there is not enough samples
			return true;
		}

		double dist = (mean - threshold) / Math.sqrt(variance);
		double icdfValue = -getICDF(confidence);

		return icdfValue >= dist;
	}
	
	public boolean isLessOrEqualTo(double threshold) {
		return isLessOrEqualTo(threshold, a);
	}
	
	public boolean isLessOrEqualTo(StudentsDistribution other, TTable.ALPHAS confidence) {
		double combinedMean = this.mean - other.mean;
		double combinedVariance = this.variance + other.variance;

		int combinedDf = (int) ((combinedVariance*combinedVariance)
				/ ((this.variance*this.variance) / (double) this.df
						+ (other.variance*other.variance) / (double) other.df));
		if (combinedDf < 1) {
			// accept hypothesis if there is not enough samples
			return true;
		}

		double icdfValue = -getICDF(confidence, combinedDf);

		return icdfValue >= combinedMean / Math.sqrt(combinedVariance);
	}
	
	public boolean isLessOrEqualTo(StudentsDistribution other) {
		return isLessOrEqualTo(other, a);
	}
	
	public boolean isGreaterOrEqualTo(double threshold, TTable.ALPHAS confidence) {
		if (df < 1) {
			// accept hypothesis if there is not enough samples
			return true;
		}

		double dist = (mean - threshold) / Math.sqrt(variance);
		double icdfValue = getICDF(confidence);
	
		return icdfValue <= dist;
	}
	
	public boolean isGreaterOrEqualTo(double threshold) {
		return isGreaterOrEqualTo(threshold, a);
	}
		
	public boolean isGreaterOrEqualTo(StudentsDistribution other, TTable.ALPHAS confidence) {
		double combinedMean = this.mean - other.mean;
		double combinedVariance = this.variance + other.variance;

		int combinedDf = (int) ((combinedVariance*combinedVariance)
				/ ((this.variance*this.variance) / (double) this.df
						+ (other.variance*other.variance) / (double) other.df));
		if (combinedDf < 1) {
			// accept hypothesis if there is not enough samples
			return true;
		}

		double icdfValue = getICDF(confidence, combinedDf);

		return icdfValue <= combinedMean / Math.sqrt(combinedVariance);
	}
	
	public boolean isGreaterOrEqualTo(StudentsDistribution other) {
		return isGreaterOrEqualTo(other, a);
	}

	public boolean isLessThan(double threshold, TTable.ALPHAS confidence) {
		return !isGreaterOrEqualTo(threshold, confidence);
	}

	public boolean isLessThan(double threshold) {
		return isLessThan(threshold, a);
	}
	
	public boolean isLessThan(StudentsDistribution other, TTable.ALPHAS confidence) {
		return !isGreaterOrEqualTo(other, confidence);
	}

	public boolean isLessThan(StudentsDistribution other) {
		return isLessThan(other, a);
	}
	
	public boolean isGreaterThan(double threshold, TTable.ALPHAS confidence) {
		return !isLessOrEqualTo(threshold, confidence);
	}
	
	public boolean isGreaterThan(double threshold) {
		return isGreaterThan(threshold, a);
	}
	
	public boolean isGreaterThan(StudentsDistribution other, TTable.ALPHAS confidence) {
		return !isLessOrEqualTo(other, confidence);
	}
	
	public boolean isGreaterThan(StudentsDistribution other) {
		return isGreaterThan(other, a);
	}
	
	public double getMean() {
		return mean;
	}
	
	public double getVariance() {
		return variance;
	}

	public void setAlpha(TTable.ALPHAS a) {
		this.a = a;
	}
	
	public void resetToDefaultAlpha() {
		a = TTable.ALPHAS.values()[DEFAULT_ALPHA_INDEX];
	}
	
	public static void setDefaultAlpha(int index) {
		DEFAULT_ALPHA_INDEX = index;
	}
	
	public TTable.ALPHAS getAlpha() {
		return a;
	}

	
	private double getICDF(TTable.ALPHAS confidence, int dfValue) {
		/* Though declared as one-dimensional, icdfA (a table with critical values for a particular alpha) is essentially a two dimensional table. 
		   The rows are indexed by major_idx, the columns are indexed by minor_idx. 
		   The first row corresponds to critical values for degrees of freedom 1, 2, 3, ..., minor_count
		   The second row starts where the previous row left off and goes by minor_step, i.e.  minor_count + minor_step, minor_count + minor_step * 2, ..., minor_count + minor_step * minor_count
		   The same pattern repets for all subsequent rows. The minor_step is computed as 2^(major_idx * boost), which means that the minor_step grows dramatically with every row.
		   As the result the algorithm below needs only a couple of iterations to get to degrees of freedom in order of milions. With current settings, it needs 6 iterations to reach df=1e9
		   Once it finds the critical value, it returns it. If the particular degree of freedom is not present, it interpolates the critical value
		   between to closest higer and lower degree of freedom.
		*/

		final double[] icdfA = TTable.icdf[confidence.ordinal()];
		int base = 0;
		int major_idx = -1;
		int base_incr = 0;
		int minor_step = 1;
		while (base + base_incr < dfValue) {
			base += base_incr;
			major_idx += 1;
			minor_step = (1 << major_idx * TTable.boost);
			base_incr = minor_step * TTable.minor_count;
		}
		dfValue -= base;
		if (dfValue % minor_step == 0) {
			int minor_idx = dfValue / minor_step - 1;
			int idx = major_idx * TTable.minor_count + minor_idx;
			return icdfA[idx];
		} else {
			int minor_idx = dfValue / minor_step;
			int df_high = base + (minor_idx + 1) * minor_step;
			int df_low = df_high - minor_step;
			int idx_high = major_idx * TTable.minor_count + minor_idx;
			int idx_low = idx_high - 1;
			double delta = (double) dfValue / (df_high - df_low);
			return icdfA[idx_high] * delta + icdfA[idx_low] * (1 - delta);
		}
	}
	
	private double getICDF(TTable.ALPHAS confidence) { 
		return getICDF(confidence, df);
	}

}
