package cz.cuni.mff.d3s.tss.arima;

public class TimeSeriesRegressionForecast {

    private final TimeSeries pointForecast;
    private final TimeSeries lowerPredictionInterval;
    private final TimeSeries upperPredictionInterval;

    TimeSeriesRegressionForecast(TimeSeries pointForecast, TimeSeries lowerPredictionInterval,
                                 TimeSeries upperPredictionInterval) {
        this.pointForecast = pointForecast;
        this.lowerPredictionInterval = lowerPredictionInterval;
        this.upperPredictionInterval = upperPredictionInterval;
    }

    public TimeSeries upperPredictionInterval() {
        return this.upperPredictionInterval;
    }

    public TimeSeries lowerPredictionInterval() {
        return this.lowerPredictionInterval;
    }

    public TimeSeries pointEstimates() {
        return this.pointForecast;
    }
}
