package cz.cuni.mff.d3s.tss.arima;

import org.ejml.alg.dense.mult.MatrixVectorMult;
import org.ejml.data.D1Matrix64F;
import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.LinearSolverFactory;
import org.ejml.interfaces.decomposition.QRDecomposition;
import org.ejml.interfaces.linsol.LinearSolver;
import org.ejml.ops.CommonOps;

public class MultipleLinearRegression {
	private final double[][] predictors;
    private final double[][] XtXInv;
    private final double[] response;
    private final double[] beta;
    private final double[] standardErrors;
    private final double[] residuals;
    private final double[] fitted;
    private final boolean hasIntercept;
    private final double sigma2;

    MultipleLinearRegression(MultipleLinearRegressionBuilder multipleLinearRegressionBuilder) {
        this.predictors = multipleLinearRegressionBuilder.getPredictors();
        this.response = multipleLinearRegressionBuilder.getResponse();
        this.hasIntercept = multipleLinearRegressionBuilder.hasIntercept();
        MatrixFormulation matrixFormulation = new MatrixFormulation();
        this.XtXInv = getXtXInverse(matrixFormulation);
        this.beta = matrixFormulation.getBetaEstimates();
        this.standardErrors = matrixFormulation.getBetaStandardErrors(this.beta.length);
        this.fitted = matrixFormulation.computeFittedValues();
        this.residuals = matrixFormulation.getResiduals();
        this.sigma2 = matrixFormulation.getSigma2();
    }

	private double[][] getXtXInverse(MatrixFormulation matrixFormulation) {
        DenseMatrix64F XtXInvMatrix = matrixFormulation.XtXInv.copy();
        int dim = XtXInvMatrix.getNumCols();
        double[][] XtXInvArray = new double[dim][dim];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                XtXInvArray[i][j] = XtXInvMatrix.get(i, j);
            }
        }
        return XtXInvArray;
    }

    public double[][] predictors() {
        double[][] copy = new double[this.predictors.length][];
        for (int i = 0; i < copy.length; i++) {
            copy[i] = this.predictors[i].clone();
        }
        return copy;
    }

    public double[][] designMatrix() {
        if (this.hasIntercept) {
            double[][] copy = new double[this.predictors.length + 1][];
            copy[0] = fill(response.length, 1.0);
            for (int i = 1; i < copy.length; i++) {
                copy[i] = this.predictors[i - 1].clone();
            }
            return copy;
        }
        return predictors();
    }
    
    /**
     * Create and return a new array of the given size with every value set to the given value.
     *
     * @param size  the number of elements of the new array.
     * @param value the value to fill every element of the array with.
     * @return a new array of the given size with every value set to the given value.
     */
    public static double[] fill(final int size, final double value) {
        final double[] filled = new double[size];
        for (int i = 0; i < filled.length; i++) {
            filled[i] = value;
        }
        return filled;
    }

    public double[][] XtXInverse() {
        double[][] copy = new double[this.XtXInv.length][];
        for (int i = 0; i < copy.length; i++) {
            copy[i] = this.XtXInv[i].clone();
        }
        return copy;
    }

    public double[] beta() {
        return beta.clone();
    }

    public double[] standardErrors() {
        return this.standardErrors.clone();
    }

    public double sigma2() {
        return this.sigma2;
    }

    public double[] response() {
        return this.response.clone();
    }

    public double[] fitted() {
        return fitted.clone();
    }

    public double[] residuals() {
        return residuals.clone();
    }

    public boolean hasIntercept() {
        return this.hasIntercept;
    }

    /**
     * Create a new linear regression model from this one, using the given boolean to determine whether
     * to fit an intercept or not.
     *
     * @param hasIntercept whether or not the new regression should have an intercept.
     * @return a new linear regression model using the given boolean to determine whether to fit an intercept.
     */
    MultipleLinearRegression withHasIntercept(boolean hasIntercept) {
        return new MultipleLinearRegressionBuilder().from(this).hasIntercept(hasIntercept).build();
    }

    /**
     * Create a new linear regression model from this one, replacing the current response with the provided one.
     *
     * @param response the response variable of the new regression.
     * @return a new linear regression model with the given response variable in place of the current one.
     */
    MultipleLinearRegression withResponse(double[] response) {
        return new MultipleLinearRegressionBuilder().from(this).response(response).build();
    }

    /**
     * Create a new linear regression model from this one, with the given predictors fully replacing the current ones.
     *
     * @param predictors The new array of prediction variables to use for the regression.
     * @return a new linear regression model using the given predictors in place of the current ones.
     */
    MultipleLinearRegression withPredictors(double[]... predictors) {
        return new MultipleLinearRegressionBuilder().from(this).predictors(predictors).build();
    }

    private class MatrixFormulation {

        private final DenseMatrix64F X; // The design matrix.
        private final DenseMatrix64F Xt; // The transpose of X.
        private final DenseMatrix64F XtXInv; // The inverse of Xt times X.
        private final DenseMatrix64F b; // The parameter estimate vector.
        private final DenseMatrix64F y; // The response vector.
        private final double[] fitted;
        private final double[] residuals;
        private final double sigma2;
        private final DenseMatrix64F covarianceMatrix;

        private MatrixFormulation() {
            int numRows = response.length;
            int numCols = predictors.length + ((hasIntercept) ? 1 : 0);
            this.X = createMatrixA(numRows, numCols);
            this.Xt = new DenseMatrix64F(numCols, numRows);
            CommonOps.transpose(X, Xt);
            this.XtXInv = new DenseMatrix64F(numCols, numCols);
            this.b = new DenseMatrix64F(numCols, 1);
            this.y = new DenseMatrix64F(numRows, 1);
            solveSystem(numRows, numCols);
            this.fitted = computeFittedValues();
            this.residuals = computeResiduals();
            this.sigma2 = estimateSigma2(numCols);
            this.covarianceMatrix = new DenseMatrix64F(numCols, numCols);
            CommonOps.scale(sigma2, XtXInv, covarianceMatrix);
        }

        private void solveSystem(int numRows, int numCols) {
            LinearSolver<DenseMatrix64F> qrSolver = LinearSolverFactory.qr(numRows, numCols);
            QRDecomposition<DenseMatrix64F> decomposition = qrSolver.getDecomposition();
            qrSolver.setA(X);
            y.setData(response);
            qrSolver.solve(this.y, this.b);
            DenseMatrix64F R = decomposition.getR(null, true);
            LinearSolver<DenseMatrix64F> linearSolver = LinearSolverFactory.linear(numCols);
            linearSolver.setA(R);
            DenseMatrix64F Rinverse = new DenseMatrix64F(numCols, numCols);
            linearSolver.invert(Rinverse); // stores solver's solution inside of Rinverse.
            CommonOps.multOuter(Rinverse, this.XtXInv);
        }

        private DenseMatrix64F createMatrixA(int numRows, int numCols) {
            double[] data;
            if (hasIntercept) {
                data = fill(numRows, 1.0);
            } else {
                data = new double[0];
            }
            for (double[] predictor : predictors) {
                data = combine(data, predictor.clone());
            }
            boolean isRowMajor = false;
            return new DenseMatrix64F(numRows, numCols, isRowMajor, data);
        }
        
        /**
         * Create a new array by combining the elements of the input arrays in the order given.
         *
         * @param arrays the arrays to combine.
         * @return a new array formed by combining the elements of the input arrays.
         */
        public double[] combine(double[]... arrays) {
            int newArrayLength = 0;
            for (double[] array : arrays) {
                newArrayLength += array.length;
            }
            double[] newArray = new double[newArrayLength];
            newArrayLength = 0;
            for (double[] array : arrays) {
                System.arraycopy(array, 0, newArray, newArrayLength, array.length);
                newArrayLength += array.length;
            }
            return newArray;
        }

        private double[] computeFittedValues() {
            D1Matrix64F fitted = new DenseMatrix64F(response.length, 1);
            MatrixVectorMult.mult(X, b, fitted);
            return fitted.getData();
        }

        private double[] computeResiduals() {
            double[] residuals = new double[fitted.length];
            for (int i = 0; i < residuals.length; i++) {
                residuals[i] = (response[i] - fitted[i]);
            }
            return residuals;
        }

        private double[] getResiduals() {
            return this.residuals.clone();
        }

        private double estimateSigma2(int df) {
            double ssq = sumOfSquared(this.residuals.clone());
            return ssq / (this.residuals.length - df);
        }
        
        public double sumOfSquared(final double... data) {
            return sumOf(squared(data));
        }
        
        public double[] squared(final double... data) {
            final double[] squared = new double[data.length];
            for (int i = 0; i < squared.length; i++) {
                squared[i] = data[i] * data[i];
            }
            return squared;
        }
        public double sumOf(final double... data) {
            double sum = 0.0;
            for (double element : data) {
                sum += element;
            }
            return sum;
        }

        private double[] getBetaStandardErrors(int numCols) {
            DenseMatrix64F diag = new DenseMatrix64F(numCols, 1);
            CommonOps.extractDiag(this.covarianceMatrix, diag);
            return sqrt(diag.getData());
        }

        private double[] getBetaEstimates() {
            return b.getData().clone();
        }

        private double getSigma2() {
            return this.sigma2;
        }
        

        /**
         * Take the square root of each element of the given array and return the result in a new array.
         *
         * @param data the data to take the square root of.
         * @return a new array containing the square root of each element.
         */
        public double[] sqrt(final double... data) {
            final double[] sqrtData = new double[data.length];
            for (int i = 0; i < sqrtData.length; i++) {
                sqrtData[i] = Math.sqrt(data[i]);
            }
            return sqrtData;
        }
    }
}
