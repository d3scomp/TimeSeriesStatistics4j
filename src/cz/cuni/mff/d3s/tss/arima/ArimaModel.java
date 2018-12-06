package cz.cuni.mff.d3s.tss.arima;

import com.github.signaflo.timeseries.TimeSeries;
import com.github.signaflo.timeseries.forecast.Forecast;
import com.github.signaflo.timeseries.model.arima.Arima;


public class ArimaModel {

	private final Arima model;
	
	public ArimaModel(double[] samples, int seasonalCycle, ArimaOrder order) {
		com.github.signaflo.timeseries.TimeSeries series = 
				com.github.signaflo.timeseries.TimeSeries.from(samples);
		model = Arima.model(series, order.getOrder());
	}
		
	public double[] getForecast(int forecastLength) {
		Forecast forecast = model.forecast(forecastLength);
		TimeSeries forecastSeries = forecast.pointEstimates();
		
		return forecastSeries.asArray();
	}
	
	public double getForecastValue(int timeOffset) {
		return getForecast(timeOffset+1)[timeOffset];
	}
	
	public double[] getForecastLowerBound(int forecastLength) {
		Forecast forecast = model.forecast(forecastLength);
		TimeSeries forecastSeries = forecast.lowerPredictionInterval();
		
		return forecastSeries.asArray();
	}
	
	public double getForecastLowerBoundValue(int timeOffset) {
		return getForecastLowerBound(timeOffset+1)[timeOffset];
	}
	
	public double[] getForecastUpperBound(int forecastLength) {
		Forecast forecast = model.forecast(forecastLength);
		TimeSeries forecastSeries = forecast.upperPredictionInterval();
		
		return forecastSeries.asArray();
	}
	
	public double getForecastUpperBoundValue(int timeOffset) {
		return getForecastUpperBound(timeOffset+1)[timeOffset];
	}
	
	public boolean isForecastValueAbove(int timeOffset, double threshold) {
		return getForecastLowerBoundValue(timeOffset) > threshold;
	}

	public boolean isForecastValueBelow(int timeOffset, double threshold) {
		return getForecastUpperBoundValue(timeOffset) < threshold;
	}
}
