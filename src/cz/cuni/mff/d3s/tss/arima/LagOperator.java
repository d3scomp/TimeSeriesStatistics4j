package cz.cuni.mff.d3s.tss.arima;

public class LagOperator {

	private LagOperator() {
	}

	/**
	 * Apply the lag operator once at the given index.
	 *
	 * @param series
	 *            the series to apply the lag operator to.
	 * @param index
	 *            the index to apply the lag operator at.
	 * @return the value of the series at lag 1 from the given index.
	 */
	public static double apply(final TimeSeries series, final int index) {
		return series.at(index - 1);
	}

	/**
	 * Apply the lag operator once at the given date-time.
	 *
	 * @param series
	 *            the series to apply the lag operator to.
	 * @param lag
	 *            the lag to apply the lag operator at.
	 * @return the value of the series at lag 1 from the given date-time.
	 */
	public static double applyLag(final TimeSeries series, final int lag) {
		return series.at(series.dateTimeIndex().get(lag) - 1);
	}

	/**
	 * Apply the lag operator the given number of times at the given index.
	 *
	 * @param series
	 *            the series to apply the lag operator to.
	 * @param index
	 *            the index to apply the lag operator at.
	 * @param times
	 *            the number of times to apply the lag operator.
	 * @return the value of the series at the given number of lags from the given
	 *         index.
	 */
	public static double apply(final TimeSeries series, final int index, final int times) {
		return series.at(index - times);
	}

	/**
	 * Apply the lag operator the given number of times at the given date-time.
	 *
	 * @param series
	 *            the series to apply the lag operator to.
	 * @param lag
	 *            the lag to apply the lag operator at.
	 * @param times
	 *            the number of times to apply the lag operator.
	 * @return the value of the series at the given number of lags from the given
	 *         date-time.
	 */
	public static double applyLag(final TimeSeries series, final int lag, final int times) {
		return series.at(series.dateTimeIndex().get(lag) - times);
	}

	/**
	 * Apply the lag operator the given number of times at the given index.
	 *
	 * @param series
	 *            the series to apply the lag operator to.
	 * @param index
	 *            the index to apply the lag operator at.
	 * @param times
	 *            the number of times to apply the lag operator.
	 * @return the value of the series at the given number of lags from the given
	 *         index.
	 */
	public static double apply(final double[] series, final int index, final int times) {
		return series[index - times];
	}
}
