package cz.cuni.mff.d3s.tss.arima;

import java.util.List;

public class TimeSeriesRegressionForecaster {

    private final TimeSeries timeSeries;
    private final MultipleLinearRegressionPredictor predictor;
    private final Vector beta;
    private final Matrix predictionMatrix;

    TimeSeriesRegressionForecaster(TimeSeries timeSeries, MultipleLinearRegressionPredictor predictor,
                                   Vector beta, Matrix predictionMatrix) {
        this.timeSeries = timeSeries;
        this.predictor = predictor;
        this.beta = beta;
        this.predictionMatrix = predictionMatrix;
    }

    public TimeSeries computePointForecasts(int steps) {
        double[] forecasts = predictionMatrix.times(beta).elements();
        int sampleEnd = this.timeSeries.getObservationTimes().get(timeSeries.size() - 1);
        int startTime = sampleEnd + 1;
        return TimeSeries.from(startTime, forecasts);
    }

    public TimeSeriesRegressionForecast forecast(int steps, double alpha) {
        final TimeSeries forecast = computePointForecasts(steps);
        List<MultipleLinearRegressionPrediction> predictions = this.predictor.predictDesignMatrix(predictionMatrix, alpha);
        final TimeSeries lowerBounds = computeLowerPredictionBounds(predictions, forecast, steps);
        final TimeSeries upperBounds = computeUpperPredictionBounds(predictions, forecast, steps);
        return new TimeSeriesRegressionForecast(forecast, lowerBounds, upperBounds);
    }

    private TimeSeries computeLowerPredictionBounds(List<MultipleLinearRegressionPrediction> predictions,
                                            TimeSeries forecast, int steps) {
        double[] bounds = new double[steps];
        for (int i = 0; i < steps; i++) {
            bounds[i] = predictions.get(i).predictionInterval().first();
        }
        return TimeSeries.from(forecast.startTime(), bounds);
    }

    private TimeSeries computeUpperPredictionBounds(List<MultipleLinearRegressionPrediction> predictions,
                                            TimeSeries forecast, int steps) {
        double[] bounds = new double[steps];
        for (int i = 0; i < steps; i++) {
            bounds[i] = predictions.get(i).predictionInterval().second();
        }
        return TimeSeries.from(forecast.startTime(), bounds);
    }

    public TimeSeries computeLowerPredictionBounds(TimeSeries forecast, int steps, double alpha) {
        List<MultipleLinearRegressionPrediction> predictions = this.predictor.predictDesignMatrix(predictionMatrix, alpha);
        double[] bounds = new double[steps];
        for (int i = 0; i < steps; i++) {
            bounds[i] = predictions.get(i).predictionInterval().first();
        }
        return TimeSeries.from(forecast.startTime(), bounds);
    }

    public TimeSeries computeUpperPredictionBounds(TimeSeries forecast, int steps, double alpha) {
        List<MultipleLinearRegressionPrediction> predictions = this.predictor.predictDesignMatrix(predictionMatrix, alpha);
        double[] bounds = new double[steps];
        for (int i = 0; i < steps; i++) {
            bounds[i] = predictions.get(i).predictionInterval().second();
        }
        return TimeSeries.from(forecast.startTime(), bounds);
    }
}
