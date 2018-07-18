package cz.cuni.mff.d3s.tss.arima;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TimeSeries {
	private final int n;
	private final double mean;
	public final double[] series;
	private int timePeriod;
	private final List<Integer> observationTimes;
	private final Map<Integer, Integer> dateTimeIndex;
	private final double[] dataSet;
	
	private TimeSeries(final double[] series, final List<Integer> times) {
		this.dataSet = series;
		this.series = series.clone();
		this.n = series.length;
		this.mean = meanOf(dataSet);
		Map<Integer, Integer> dateTimeIndex = new HashMap<>(series.length);
		for (int i = 0; i < series.length; i++) {
			dateTimeIndex.put(times.get(i), i);
		}
		this.observationTimes = Collections.unmodifiableList(times);
		this.dateTimeIndex = Collections.unmodifiableMap(dateTimeIndex);
	}

	/**
     * Create a new time series with the given time period, the time of first observation, and the observation data.
     *
     * @param timePeriod the period of time between observations.
     * @param startTime  the time of the first observation.
     * @param series     the observation data.
     * @return a new time series from the supplied data.
     */
    public static TimeSeries from(final int startTime,
                                  final double... series) {
    	ArrayList<Integer> times = new ArrayList<>();
    	for(int i = 0; i < series.length; i++) {
    		times.add(startTime + i);
    	}
        return new TimeSeries(series, times);
    }
    
    /**
     * Create a new time series from the given data without regard to when the observations were made. Use this
     * constructor if the dates and times associated with the observations do not matter.
     *
     * @param series the observation data.
     * @return a new time series from the supplied data.
     *
     */
    public static TimeSeries from(final double[] series, List<Integer> times) {
        return new TimeSeries(series, times);
    }
    
	/**
	 * Difference the given series the given number of times at the given lag.
	 *
	 * @param series
	 *            the series to difference.
	 * @param lag
	 *            the lag at which to take differences.
	 * @param times
	 *            the number of times to difference the series at the given lag.
	 *            Note that this argument may equal 0, in which case a copy of the
	 *            original series is returned.
	 * @return a new time series differenced the given number of times at the given
	 *         lag.
	 *
	 * @throws IllegalArgumentException
	 *             if lag is less than 1.
	 * @throws IllegalArgumentException
	 *             if times is less than 0.
	 * @throws IllegalArgumentException
	 *             if the product of lag and times is greater than the length of the
	 *             series.
	 */
	public static double[] difference(final double[] series, final int lag, final int times) {
		validate(lag);
		validate(series, lag, times);
		if (times == 0) {
			return series.clone();
		}
		double[] diffed = differenceArray(series, lag);
		for (int i = 1; i < times; i++) {
			diffed = differenceArray(diffed, lag);
		}
		return diffed;
	}

	/**
	 * Difference the given series the given number of times at lag 1.
	 *
	 * @param series
	 *            the series to difference.
	 * @param times
	 *            the number of times to difference the series.
	 * @return a new time series differenced the given number of times at lag 1.
	 *
	 * @throws IllegalArgumentException
	 *             if times is less than 0.
	 * @throws IllegalArgumentException
	 *             if times is greater than the length of the series.
	 */
	public static double[] difference(final double[] series, final int times) {
		return difference(series, 1, times);
	}

	private static double[] differenceArray(final double[] series, final int lag) {
		double[] differenced = new double[series.length - lag];
		for (int i = 0; i < differenced.length; i++) {
			differenced[i] = series[i + lag] - series[i];
		}
		return differenced;
	}

	private static void validate(double[] series, int lag, int times) {
		if (times < 0) {
			throw new IllegalArgumentException("The value of times must be non-negative " + "but was " + times);
		}
		if (times * lag > series.length) {
			throw new IllegalArgumentException(
					"The product of lag and times " + "must be less than or equal to the length of the series, but "
							+ times + " * " + lag + " = " + times * lag + " is greater than " + series.length);
		}
	}

	private static void validate(double[] series, int lag) {
		if (lag < 1) {
			throw new IllegalArgumentException("The lag must be positive, but was " + lag);
		}
		if (lag > series.length) {
			throw new IllegalArgumentException("The lag must be less than or equal to the length of the series, "
					+ "but " + lag + " is greater than " + series.length);
		}
	}

	private static void validate(int lag) {
		if (lag < 1) {
			throw new IllegalArgumentException("The lag must be positive, but was " + lag);
		}
	}

	/**
	 * Retrieve the value of the time series at the given index. Indexing begins at
	 * 0.
	 *
	 * @param index
	 *            the index of the value to return.
	 * @return the value of the time series at the given index.
	 */
	public final double at(final int index) {
		if (index < 0 || index >= this.series.length) {
			throw new IndexOutOfBoundsException("No observation available at index: " + index);
		}
		return this.series[index];
	}

	/**
	 * Retrieve the value of the time series at the given date-time.
	 *
	 * @param dateTime
	 *            the date-time of the value to return.
	 * @return the value of the time series at the given date-time.
	 *
	 * @throws IllegalArgumentException
	 *             if there is no observation at the given date-time.
	 */
	public final double atTime(final int dateTime) {
		if (!dateTimeIndex.containsKey(dateTime)) {
			throw new IllegalArgumentException("No observation available at date-time: " + dateTime);
		}
		return this.series[dateTimeIndex.get(dateTime)];
	}

	/**
	 * The correlation of this series with itself at lag k.
	 *
	 * @param k
	 *            the lag to compute the autocorrelation at.
	 * @return the correlation of this series with itself at lag k.
	 *
	 * @throws IllegalArgumentException
	 *             if k is less than 0.
	 */
	public final double autoCorrelationAtLag(final int k) {
		validateLag(k);
		final double variance = autoCovarianceAtLag(0);
		return autoCovarianceAtLag(k) / variance;
	}

	/**
	 * Every correlation coefficient of this series with itself up to the given lag.
	 *
	 * @param k
	 *            the maximum lag to compute the autocorrelation at.
	 * @return every correlation coefficient of this series with itself up to the
	 *         given lag.
	 *
	 * @throws IllegalArgumentException
	 *             if k is less than 0.
	 */
	public final double[] autoCorrelationUpToLag(final int k) {
		validateLag(k);
		final double[] autoCorrelation = new double[Math.min(k + 1, n)];
		for (int i = 0; i < Math.min(k + 1, n); i++) {
			autoCorrelation[i] = autoCorrelationAtLag(i);
		}
		return autoCorrelation;
	}

	/**
	 * The covariance of this series with itself at lag k.
	 *
	 * @param k
	 *            the lag to compute the autocovariance at.
	 * @return the covariance of this series with itself at lag k.
	 *
	 * @throws IllegalArgumentException
	 *             if k is less than 0.
	 */
	public final double autoCovarianceAtLag(final int k) {
		validateLag(k);
		double sumOfProductOfDeviations = 0.0;
		for (int t = 0; t < n - k; t++) {
			sumOfProductOfDeviations += (series[t] - mean) * (series[t + k] - mean);
		}
		return sumOfProductOfDeviations / n;
	}

	/**
	 * Every covariance measure of this series with itself up to the given lag.
	 *
	 * @param k
	 *            the maximum lag to compute the autocovariance at.
	 * @return every covariance measure of this series with itself up to the given
	 *         lag.
	 *
	 * @throws IllegalArgumentException
	 *             if k is less than 0.
	 */
	public final double[] autoCovarianceUpToLag(final int k) {
		validateLag(k);
		final double[] acv = new double[Math.min(k + 1, n)];
		for (int i = 0; i < Math.min(k + 1, n); i++) {
			acv[i] = autoCovarianceAtLag(i);
		}
		return acv;
	}

	private void validateLag(int k) {
		if (k < 0) {
			throw new IllegalArgumentException("The lag, k, must be non-negative, but was " + k);
		}
	}

	/**
	 * Compute a moving average of order m.
	 *
	 * @param m
	 *            the order of the moving average.
	 * @return a new time series with the smoothed observations.
	 */
	public final TimeSeries movingAverage(final int m) {
		final int c = m % 2;
		final int k = (m - c) / 2;
		final double[] average;
		average = new double[this.n - m + 1];
		double sum;
		for (int t = 0; t < average.length; t++) {
			sum = 0;
			for (int j = -k; j < k + c; j++) {
				sum += series[t + k + j];
			}
			average[t] = sum / m;
		}
		final List<Integer> times = this.observationTimes.subList(k + c - 1, n - k);
		return new TimeSeries(average, times);
	}

	/**
	 * Return a moving average of order m if m is odd and of order 2 &times; m if m
	 * is even.
	 *
	 * @param m
	 *            the order of the moving average.
	 * @return a centered moving average of order m.
	 */
	public final TimeSeries centeredMovingAverage(final int m) {
		if (m % 2 == 1)
			return movingAverage(m);
		TimeSeries firstAverage = movingAverage(m);
		final int k = m / 2;
		final List<Integer> times = this.observationTimes.subList(k, n - k);
		return new TimeSeries(firstAverage.movingAverage(2).series, times);
	}

	/**
	 * Difference this series the given number of times at the given lag.
	 *
	 * @param lag
	 *            the lag at which to take differences.
	 * @param times
	 *            the number of times to difference the series at the given lag.
	 * @return a new time series differenced the given number of times at the given
	 *         lag.
	 *
	 * @throws IllegalArgumentException
	 *             if lag is less than 1.
	 * @throws IllegalArgumentException
	 *             if times is less than 0.
	 * @throws IllegalArgumentException
	 *             if the product of lag and times is greater than the length of the
	 *             series.
	 */
	public final TimeSeries difference(final int lag, final int times) {
		validate(this.series, lag, times);
		if (times > 0) {
			TimeSeries diffed = difference(lag);
			for (int i = 1; i < times; i++) {
				diffed = diffed.difference(lag);
			}
			return diffed;
		}
		return this;
	}

	/**
	 * Difference this time series at the given lag and return the result as a new
	 * time series.
	 *
	 * @param lag
	 *            the lag at which to take differences.
	 * @return a new time series differenced at the given lag.
	 *
	 * @throws IllegalArgumentException
	 *             if lag is less than 1.
	 * @throws IllegalArgumentException
	 *             if lag is greater than the size of this series.
	 */
	public final TimeSeries difference(final int lag) {
		validate(this.series, lag);
		double[] diffed = differenceArray(series, lag);
		final List<Integer> obsTimes = this.observationTimes.subList(lag, n);
		return new TimeSeries(diffed, obsTimes);
	}

	/**
	 * Subtract the given series from this time series and return the result as a
	 * new time series. Note that if the other series is empty, then this series is
	 * returned. However, in all other cases in which the two series differ in size,
	 * an IllegalArgumentException is thrown.
	 *
	 * @param otherSeries
	 *            the series to subtract from this one.
	 * @return The difference between this series and the given series.
	 *
	 * @throws IllegalArgumentException
	 *             if the other series is non-empty and the two series differ in
	 *             size.
	 */
	public final TimeSeries minus(final TimeSeries otherSeries) {
		if (otherSeries.series.length == 0) {
			return this;
		}
		if (otherSeries.series.length != this.series.length) {
			throw new IllegalArgumentException("The two series must have the same length.");
		}
		final double[] subtracted = new double[this.series.length];
		for (int t = 0; t < subtracted.length; t++) {
			subtracted[t] = this.series[t] - otherSeries.series[t];
		}
		return new TimeSeries(subtracted, observationTimes);
	}

	/**
	 * Subtract the given series from this series and return the result as a new
	 * time series. Note that if the other series is empty, then this series is
	 * returned. However, in all other cases in which the two series differ in size,
	 * an IllegalArgumentException is thrown.
	 *
	 * @param otherSeries
	 *            the series to subtract from this one.
	 * @return The difference between this series and the given series.
	 *
	 * @throws IllegalArgumentException
	 *             if the other series is non-empty and the two series differ in
	 *             size.
	 */
	public final TimeSeries minus(final double[] otherSeries) {
		if (otherSeries.length == 0) {
			return this;
		}
		if (otherSeries.length != this.series.length) {
			throw new IllegalArgumentException("The two series must have the same length.");
		}
		final double[] subtracted = new double[this.series.length];
		for (int t = 0; t < subtracted.length; t++) {
			subtracted[t] = this.series[t] - otherSeries[t];
		}
		return new TimeSeries(subtracted, observationTimes);
	}

	public static double meanOf(final double... data) {
		final double sum = sumOf(data);
		return sum / data.length;
	}

	public static double sumOf(final double... data) {
		double sum = 0.0;
		for (double element : data) {
			sum += element;
		}
		return sum;
	}
	
	public int getTimePeriod() {
		return timePeriod;
	}
	
	public void setTimePeriod(int period) {
		timePeriod = period;
	}
	
	public List<Integer> getObservationTimes() {
		return observationTimes;
	}
	
	public int startTime() {
		return observationTimes.get(0);
	}

	public int size() {
		return series.length; 
	}
	
	 /**
     * Retrieve the time series of observations.
     *
     * @return the time series of observations.
     */
    public final double[] asArray() {
        return this.series.clone();
    }

	public Map<Integer, Integer> dateTimeIndex() {
		return dateTimeIndex;
	}
	
	public double getMean() {
		return mean;
	}
	
	public int getSampleCnt() {
		return n;
	}
}
