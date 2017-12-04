package cz.cuni.mff.d3s.tss;

public class TimeSeries {
	

	/**
	 * WindowCnt - The number of windows to keep the history of
	 */
	private final int WindowCnt;

	/**
	 * WindowSize - The time period to accumulate the samples within together. (in milliseconds)
	 */
	private final int WindowSize;
	
	private int time; // time in ms
	private int index;
	private int sampleCounts[];
	private double sampleSum[]; // sum(x_i)
	private double sampleSquaresSum[]; // sum(x_i^2)
	private double timeSum[]; // sum(t_i)
	private double timeSquaresSum[]; // sum(t_i^2)
	private double sampleTimeSum[]; // sum(x_i*t_i)

	private int totalSampleCounts;
	private double totalSampleSum;
	private double totalSampleSquaresSum;
	private double totalTimeSum;
	private double totalTimeSquaresSum;
	private double totalSampleTimeSum;

	
	public TimeSeries(int windowCnt, int windowSize) {
		this.WindowCnt = windowCnt;
		this.WindowSize = windowSize;
		
		time = 0;
		index = 0;

		sampleCounts = new int[WindowCnt];
		sampleSum = new double[WindowCnt];
		sampleSquaresSum = new double[WindowCnt];
		timeSum = new double[WindowCnt];
		timeSquaresSum = new double[WindowCnt];
		sampleTimeSum = new double[WindowCnt];
		
		totalSampleCounts = 0;
		totalTimeSquaresSum = 0;
		totalTimeSum = 0;
		totalSampleSum = 0;
		totalSampleTimeSum = 0;
		totalSampleSquaresSum = 0;

		for (int i = 0; i < windowCnt; ++i) {
			sampleCounts[i] = 0;
			sampleSum[i] = 0;
			sampleSquaresSum[i] = 0;
		}
	}

	public void addSample(double sample, int sampleTime) { // time in ms
		while ((time + WindowSize) < sampleTime) { // While here to skip potential unfilled subWindows
			// If the sample_time belongs to the next time SubWindow
			time += WindowSize; // Move to next time window and initialize it

			index = nextIndex();

			totalSampleCounts -= sampleCounts[index];	
			sampleCounts[index] = 0;

			totalSampleSum -= sampleSum[index];
			sampleSum[index] = 0;

			totalSampleSquaresSum -= sampleSquaresSum[index];
			sampleSquaresSum[index] = 0;

			totalTimeSum -= timeSum[index];
			timeSum[index] = 0;

			totalTimeSquaresSum -= timeSquaresSum[index];
			timeSquaresSum[index] = 0;

			totalSampleTimeSum -= sampleTimeSum[index];
			sampleTimeSum[index] = 0;
						
		}

		totalSampleCounts += 1;
		sampleCounts[index] += 1;

		totalSampleSum += sample;
		sampleSum[index] += sample;

		double sample2 = sample * sample;
		totalSampleSquaresSum += sample2;
		sampleSquaresSum[index] += sample2;

		double sampleTimeD = sampleTime;

		totalTimeSum += sampleTimeD;
		timeSum[index] += sampleTimeD;
		
		double sampleTime2 = sampleTimeD * sampleTimeD;
		totalTimeSquaresSum += sampleTime2;
		timeSquaresSum[index] += sampleTime2;

		double sampleSampleTime = sample * sampleTime;
		totalSampleTimeSum += sampleSampleTime;
		sampleTimeSum[index] += sampleSampleTime;
	}

	public StudentsDistribution getMean() {
		int sampleCnt = computeSampleCnt();
		double mean = computeMean(sampleCnt);
		double variance = computeMeanVariance(sampleCnt);

		return new StudentsDistribution(sampleCnt-1, mean, variance);
	}

	public StudentsDistribution getLra() {
		int sampleCnt = computeSampleCnt();
		double a = computeLraMean(sampleCnt);
		double b = computeLrbMean(sampleCnt);
		double e2 = computeEpsilonSquaredSum(a, b, sampleCnt);
		double varianceB = computeLrbVariance(sampleCnt, e2);
		double varianceA = computeLraVariance(sampleCnt, varianceB);

		return new StudentsDistribution(sampleCnt-2, a, varianceA);
	}

	public StudentsDistribution getLrb() {
		int sampleCnt = computeSampleCnt();
		double a = computeLraMean(sampleCnt);
		double b = computeLrbMean(sampleCnt);
		double e2 = computeEpsilonSquaredSum(a, b, sampleCnt);
		double variance = computeLrbVariance(sampleCnt, e2);

		return new StudentsDistribution(sampleCnt-2, b, variance);
	}

	public StudentsDistribution getLr(double x) {
		int sampleCnt = computeSampleCnt();
		double a = computeLraMean(sampleCnt);
		double b = computeLrbMean(sampleCnt);
		double mean = computeLrMean(a, b, x);
		double e2 = computeEpsilonSquaredSum(a, b, sampleCnt);
		double variance = computeLrVariance(sampleCnt, e2, x);

		return new StudentsDistribution(sampleCnt-2, mean, variance);
	}


	private int nextIndex() {
		return (index + 1) % WindowCnt;
	}

	private int computeSampleCnt() {
		return totalSampleCounts;
	}

	private double computeMean(int n) {
		if (n <= 0) {
			return Double.NaN;
		}

		double m = totalSampleSum;
		return m / n;
	}

	private double computeMeanVariance(int n) {
		if (n <= 1) {
			return Double.NaN;
		}

		double x2 = totalSampleSquaresSum;
		double x = totalSampleSum;

		return (x2 - x*x/n) / ((n - 1) * n);
	}

	private double computeLraMean(int n) {
		if (n <= 0) {
			return Double.NaN;
		}

		double y = totalSampleSum;
		double x = totalTimeSum;
		double b = computeLrbMean(n);

		return (y - b * x) / n;
	}

	private double computeLraVariance(int n, double varianceB) {
		if (n <= 0) {
			return Double.NaN;
		}

		double x2 = totalTimeSquaresSum;

		return varianceB * x2 / n;
	}

	private double computeLrbMean(int n) {
		if (n <= 0) {
			return Double.NaN;
		}

		double x = totalTimeSum;
		double x2 = totalTimeSquaresSum;
		double y = totalSampleSum;
		double xy = totalSampleTimeSum;

		double nom = xy - x*y / n;
		double denom = x2 - x*x / n;

		return denom != 0 ? nom / denom : Double.NaN;
	}

	/*
	 * e2 = computeEpsilonSquaredSum(...)
	 */
	private double computeLrbVariance(int n, double e2) {
		if (n <= 0) {
			return Double.NaN;
		}

		double x = totalTimeSum;
		double x2 = totalTimeSquaresSum;

		double nom = e2 / (n - 2);
		double denom = x2 - x*x/n;
		return denom != 0 ? nom / denom : Double.NaN;
	}

	private double computeLrMean(double a, double b, double x) {
		return a + b * x;
	}

	/*
	 * point is the point of interest
	 */
	private double computeLrVariance(int n, double e2, double point) {
		// Check that division by zero won't occur
		if (n <= 2) {
			return Double.NaN;
		}

		double x = totalTimeSum;
		double x2 = totalTimeSquaresSum;
		double pAvgX = point - x/n;
		double denom = x2 - x*x/n;

		return denom != 0 ? (e2 / (n-2)) * (1.0 / n + pAvgX * pAvgX / denom) : Double.NaN;
	}

	/*private double computeSum(double samples[]) {
		double s = 0;
		for (int i = 0; i < WindowCnt; ++i) {
			s += samples[i];
		}
		return s;
	}

	private int computeSum(int samples[]) {
		int s = 0;
		for (int i = 0; i < WindowCnt; ++i) {
			s += samples[i];
		}
		return s;
	}*/

	private double computeEpsilonSquaredSum(double a, double b, int n) {
		if (n <= 2) {
			return Double.NaN;
		}

		double x = totalTimeSum;
		double x2 = totalTimeSquaresSum;
		double y = totalSampleSum;
		double y2 = totalSampleSquaresSum;
		double xy = totalSampleTimeSum;

		return y2 + n*a*a + b*b*x2 - 2*a*y - 2*b*xy + 2*a*b*x;
	}

}
