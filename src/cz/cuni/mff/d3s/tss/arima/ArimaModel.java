package cz.cuni.mff.d3s.tss.arima;

import com.github.signaflo.timeseries.forecast.Forecast;
import com.github.signaflo.timeseries.model.arima.Arima;
import com.github.signaflo.timeseries.TimeSeries;

import cz.cuni.mff.d3s.tss.TTable;


public class ArimaModel {

	public static double[] getArimaForecast(double[] samples, int forecast_length, ArimaOrder order, TTable.ALPHAS confidence) {
		com.github.signaflo.timeseries.TimeSeries series = com.github.signaflo.timeseries.TimeSeries.from(samples);
		
		Arima model = Arima.model(series, order.getOrder());
		Forecast forecast = model.forecast(forecast_length, confidence.getValue());
		TimeSeries forecastSeries = forecast.pointEstimates();
		
		return forecastSeries.asArray();
	}
	
	public static double[] getArimaForecastLowerBound(double[] samples, int forecast_length, ArimaOrder order, TTable.ALPHAS confidence) {
		com.github.signaflo.timeseries.TimeSeries series = com.github.signaflo.timeseries.TimeSeries.from(samples);
		
		Arima model = Arima.model(series, order.getOrder());
		Forecast forecast = model.forecast(forecast_length, confidence.getValue());
		TimeSeries forecastSeries = forecast.lowerPredictionInterval();
		
		return forecastSeries.asArray();
	}
	
	public static double[] getArimaForecastUpperBound(double[] samples, int forecast_length, ArimaOrder order, TTable.ALPHAS confidence) {
		com.github.signaflo.timeseries.TimeSeries series = com.github.signaflo.timeseries.TimeSeries.from(samples);
		
		Arima model = Arima.model(series, order.getOrder());
		Forecast forecast = model.forecast(forecast_length, confidence.getValue());
		TimeSeries forecastSeries = forecast.upperPredictionInterval();
		
		return forecastSeries.asArray();
	}
}
