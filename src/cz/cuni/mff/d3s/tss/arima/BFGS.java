package cz.cuni.mff.d3s.tss.arima;

/**
 * An implementation of the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm for unconstrained
 * nonlinear optimization. This class is immutable and thread-safe.
 *
 * @author Jacob Rachiele
 */
public final class BFGS {

    private static final double C1 = 1E-4;
    private static final double STEP_REDUCTION_FACTOR = 0.2;
    //private static final double c2 = 0.9;

    private final MatrixOneD identity;
    private Vector iterate; // The point at which to evaluate the target function.
    private double functionValue; // The latest value of the target function.
    private double rho; // Defined as 1 divided by the dot product of y and s.
    private Vector s; // The difference between successive iterates.
    private Vector y; // The difference between successive gradients.
    private MatrixOneD H; // The inverse Hessian approximation.

    /**
     * Create a new BFGS object and run the algorithm with the supplied information.
     *
     * @param f                       the function to be minimized.
     * @param startingPoint           the initial guess of the minimum.
     * @param gradientTolerance       the tolerance for the norm of the gradient of the function.
     * @param functionChangeTolerance the tolerance for the change in function value.
     */
    public BFGS(final OptimFunction f, final Vector startingPoint, final double gradientTolerance,
                final double functionChangeTolerance) {
        this(f, startingPoint, gradientTolerance, functionChangeTolerance, MatrixOneD.identity(startingPoint.size()));
    }

    /**
     * Create a new BFGS object and run the algorithm with the supplied information.
     *
     * @param f                       the function to be minimized.
     * @param startingPoint           the initial guess of the minimum.
     * @param gradientNormTolerance   the tolerance for the norm of the gradient of the function.
     * @param relativeChangeTolerance the tolerance for the change in function value.
     * @param initialHessian          The initial guess for the inverse Hessian approximation.
     */
    public BFGS(final OptimFunction f, final Vector startingPoint, final double gradientNormTolerance,
                final double relativeChangeTolerance, final MatrixOneD initialHessian) {
        this.identity = MatrixOneD.identity(startingPoint.size());
        this.H = initialHessian;
        this.iterate = startingPoint;
        int k = 0;
        double priorFunctionValue;
        functionValue = f.at(startingPoint);
        Vector gradient = f.gradientAt(startingPoint, functionValue);
        int maxIterations = 100;
        if (gradient.size() > 0) {
            double relativeChange;
            double relativeChangeDenominator;
            double stepSize;
            double slopeAt0;
            double yDotS;
            Vector nextIterate;
            Vector nextGradient;
            Vector searchDirection;
            double gradientNorm = gradient.norm();
            boolean stop = gradientNorm < gradientNormTolerance || !Double.isFinite(gradientNorm);
            int iterationsSinceIdentityReset = 0;
            while (!stop) {
                if (iterationsSinceIdentityReset > 2 * iterate.size()) {
                    H = identity;
                    iterationsSinceIdentityReset = 0;
                }
                iterationsSinceIdentityReset++;
                searchDirection = (H.times(gradient).scaledBy(-1.0));
                slopeAt0 = searchDirection.dotProduct(gradient);
                if (slopeAt0 > 0) {
                    H = this.identity;
                    searchDirection = (H.times(gradient).scaledBy(-1.0));
                    slopeAt0 = searchDirection.dotProduct(gradient);
                }
//        try {
//          stepSize = updateStepSize(functionValue);
//        } catch (NaNStepLengthException | ViolatedTheoremAssumptionsException e) {
//          stop = true;
//          continue;
//        }
                stepSize = 1.0;
                s = searchDirection.scaledBy(stepSize);
                nextIterate = iterate.plus(s);
                priorFunctionValue = functionValue;
                functionValue = f.at(nextIterate);
                final int maxStepReductions = 25;
                int stepReductions = 0;
                while (!(Double.isFinite(functionValue) &&
                         functionValue < priorFunctionValue + C1 * stepSize * slopeAt0) && !stop) {
                    relativeChangeDenominator = Math.max(Math.abs(priorFunctionValue), Math.abs(nextIterate.norm()));
                    relativeChange = Math.abs((priorFunctionValue - functionValue) / relativeChangeDenominator);
                    if (relativeChange <= relativeChangeTolerance) {
                        stop = true;
                    } else if (stepReductions > maxStepReductions) {
                        stop = true;
                    } else {
                        stepReductions++;
                        stepSize *= STEP_REDUCTION_FACTOR;
                        s = searchDirection.scaledBy(stepSize);
                        nextIterate = iterate.plus(s);
                        functionValue = f.at(nextIterate);
                    }
                }
                nextGradient = f.gradientAt(nextIterate, functionValue);
                if (!stop) {
                    relativeChangeDenominator = Math.max(Math.abs(priorFunctionValue), Math.abs(nextIterate.norm()));
                    //Hamming, Numerical Methods, 2nd edition, pg. 22
                    relativeChange = Math.abs((priorFunctionValue - functionValue) / relativeChangeDenominator);
                    if (relativeChange <= relativeChangeTolerance || nextGradient.norm() < gradientNormTolerance) {
                        stop = true;
                    }
                }
                y = nextGradient.minus(gradient);
                yDotS = y.dotProduct(s);
                if (yDotS > 0) {
                    rho = 1 / yDotS;
                    H = updateHessian();
                } else if (!stop) {
                    H = identity;
                    iterationsSinceIdentityReset = 0;
                }
                iterate = nextIterate;
                gradient = nextGradient;
                k += 1;
                if (k > maxIterations) {
                    stop = true;
                }
            }
        }
    }

//  private double updateStepSize(double functionValue) {
//    int maxAttempts = 10;
//    final double slope0 = gradient.dotProduct(searchDirection);
//    if (slope0 > 0) {
//      System.out.println("The slope at step size 0 is positive");
//    }
//    double stepSize = 1.0;
//    Vector nextIterate = iterate.plus(searchDirection.scaledBy(stepSize));
//    s = nextIterate.minus(iterate);
//    double priorFunctionValue = functionValue;
//    functionValue = f.at(nextIterate);
//    int k = 1;
//    while (!(Double.isFinite(functionValue) && functionValue < priorFunctionValue + C1 * stepSize * slope0) && k <
// maxAttempts) {
//      stepSize *= 0.2;
//      nextIterate = iterate.plus(searchDirection.scaledBy(stepSize));
//      s = nextIterate.minus(iterate);
//      functionValue = f.at(nextIterate);
//      k++;
//    }
//    return stepSize;
////    final QuasiNewtonLineFunction lineFunction = new QuasiNewtonLineFunction(this.f, iterate, searchDirection);
////    StrongWolfeLineSearch lineSearch = StrongWolfeLineSearch.newBuilder(lineFunction, functionValue, slope0).C1(C1)
////        .c2(c2).alphaMax(50).alpha0(1.0).build();
////    return lineSearch.search();
//  }

    private MatrixOneD updateHessian() {
        MatrixOneD a = identity.minus(s.outerProduct(y).scaledBy(rho));
        MatrixOneD b = identity.minus(y.outerProduct(s).scaledBy(rho));
        MatrixOneD c = s.outerProduct(s).scaledBy(rho);
        return a.times(H).times(b).plus(c);
    }

    /**
     * Return the final value of the target function.
     *
     * @return the final value of the target function.
     */
    public double functionValue() {
        return this.functionValue;
    }

    /**
     * Return the final, optimized input parameters.
     *
     * @return the final, optimized input parameters.
     */
    public Vector parameters() {
        return this.iterate;
    }

    /**
     * Return the final approximation to the inverse Hessian.
     *
     * @return the final approximation to the inverse Hessian.
     */
    public MatrixOneD inverseHessian() {
        return this.H;
    }
}
