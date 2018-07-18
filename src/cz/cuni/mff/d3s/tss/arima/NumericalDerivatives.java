package cz.cuni.mff.d3s.tss.arima;

public class NumericalDerivatives {

    private NumericalDerivatives() {
    }

    public static Vector forwardDifferenceGradient(final OptimFunction f, final Vector point, final double h) {
        double[] newPoints = point.elements().clone();
        final double[] partials = new double[point.size()];
        final double functionValue = f.at(point);
        for (int i = 0; i < partials.length; i++) {
            newPoints[i] = point.at(i) + h;
            partials[i] = (f.at(Vector.from(newPoints)) - functionValue) / h;
            newPoints = point.elements().clone();
        }
        return Vector.from(partials);
    }

    public static Vector forwardDifferenceGradient(final OptimFunction f, final Vector point, final double h,
                                                          final double functionValue) {
        double[] newPoints = point.elements().clone();
        final double[] partials = new double[point.size()];
        for (int i = 0; i < partials.length; i++) {
            newPoints[i] = point.at(i) + h;
            partials[i] = (f.at(Vector.from(newPoints)) - functionValue) / h;
            newPoints = point.elements().clone();
        }
        return Vector.from(partials);
    }

    public static Vector centralDifferenceGradient(final OptimFunction f, final Vector point, final double h) {
        double[] forwardPoints = point.elements().clone();
        double[] backwardPoints = point.elements().clone();
        final double[] partials = new double[point.size()];
        for (int i = 0; i < partials.length; i++) {
            forwardPoints[i] = point.at(i) + h;
            backwardPoints[i] = point.at(i) - h;
            partials[i] = (f.at(Vector.from(forwardPoints)) - f.at(Vector.from(backwardPoints))) / (2 * h);
            forwardPoints = point.elements().clone();
            backwardPoints = point.elements().clone();
        }
        return Vector.from(partials);
    }
}
