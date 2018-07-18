package cz.cuni.mff.d3s.tss.arima;

import java.util.Arrays;

public class MatrixOneD {
	
	/**
     * Specifies the layout of the two-dimensional array representation of a matrix. In other words, this
     * specifies whether the outer part of the two-dimensional array is a sequence of row vectors or a sequence
     * of column vectors.
     */
    enum Layout {
        BY_ROW, BY_COLUMN
    }
    
	private final int nrow;
	private final int ncol;
	private final double[] data;
	private final Layout layout;

	/**
	 * Create a new matrix with the supplied data and dimensions. The data is
	 * assumed to be in row-major order.
	 *
	 * @param nrow
	 *            the number of columns for the matrix.
	 * @param ncol
	 *            the number of columns for the matrix.
	 * @param data
	 *            the data in row-major order.
	 */
	MatrixOneD(final int nrow, final int ncol, final double... data) {
		if (nrow * ncol != data.length) {
			throw new IllegalArgumentException("The dimensions do not match the amount of data provided. "
					+ "There were " + data.length + " data points provided but the number of columns and columns "
					+ "were " + nrow + " and " + ncol + " respectively.");
		}
		this.nrow = nrow;
		this.ncol = ncol;
		this.data = data.clone();
		this.layout = Layout.BY_ROW;
	}

	/**
	 * Create a new matrix with the given dimensions filled with the supplied value.
	 *
	 * @param nrow
	 *            the number of columns for the matrix.
	 * @param ncol
	 *            the number of columns for the matrix.
	 * @param value
	 *            the data point to fill the matrix with.
	 */
	MatrixOneD(final int nrow, final int ncol, final double value) {
		this.nrow = nrow;
		this.ncol = ncol;
		this.data = new double[nrow * ncol];
		for (int i = 0; i < data.length; i++) {
			this.data[i] = value;
		}
		this.layout = Layout.BY_ROW;
	}

	/**
	 * Create a new matrix from the given two-dimensional array of data.
	 *
	 * @param layout
	 *            the layout of the elements in the supplied two dimensional array.
	 * @param matrixData
	 *            the two-dimensional array of data constituting the matrix.
	 */
	MatrixOneD(Layout layout, final double[]... matrixData) {
		this.layout = layout;
		if (matrixData.length == 0) {
			// throw new IllegalArgumentException("The matrix data cannot be empty.");
			this.ncol = 0;
			this.nrow = 0;
			this.data = new double[0];
		} else if (layout == Layout.BY_COLUMN) {
			this.ncol = matrixData.length;
			this.nrow = matrixData[0].length;
			this.data = new double[ncol * nrow];
			for (int i = 0; i < nrow; i++) {
				for (int j = 0; j < ncol; j++) {
					this.data[i * ncol + j] = matrixData[j][i];
				}
			}
		} else {
			this.nrow = matrixData.length;
			this.ncol = matrixData[0].length;
			this.data = new double[nrow * ncol];
			for (int i = 0; i < nrow; i++) {
				System.arraycopy(matrixData[i], 0, this.data, i * ncol, ncol);
			}

		}
	}

	public double get(int i, int j) {
		return this.data[i * ncol + j];
	}

	public int nrow() {
		return this.nrow;
	}

	public int ncol() {
		return this.ncol;
	}

	public MatrixOneD plus(final MatrixOneD other) {
		if (this.nrow != other.nrow() || this.ncol != other.ncol()) {
			throw new IllegalArgumentException(
					"The dimensions of this matrix must equal the dimensions of the other matrix. "
							+ "This matrix has dimension (" + this.nrow + ", " + this.ncol
							+ ") and the other matrix has dimension (" + other.nrow() + ", " + other.ncol() + ")");
		}
		final double[] sum = new double[nrow * ncol];
		final double[] otherData = other.data();
		for (int i = 0; i < nrow; i++) {
			for (int j = 0; j < ncol; j++) {
				sum[i * ncol + j] = this.data[i * ncol + j] + otherData[i * ncol + j];
			}
		}
		return new MatrixOneD(this.nrow, this.ncol, sum);
	}

	public MatrixOneD times(final MatrixOneD other) {
		if (this.ncol != other.nrow()) {
			throw new IllegalArgumentException(
					"The columns of this matrix must equal the columns of the other matrix. " + "This matrix has "
							+ this.ncol + " columns and the other matrix has " + other.nrow() + " columns.");
		}
		final double[] product = new double[this.nrow * other.ncol()];
		final double[] otherData = other.data();
		for (int i = 0; i < this.nrow; i++) {
			for (int j = 0; j < other.ncol(); j++) {
				for (int k = 0; k < this.ncol; k++) {
					product[i * this.nrow + j] += this.data[i * this.ncol + k] * otherData[j + k * other.ncol()];
				}
			}
		}
		return new MatrixOneD(this.nrow, other.ncol(), product);
	}

	public Vector times(final Vector vector) {
		double[] elements = vector.elements();
		if (this.ncol != elements.length) {
			throw new IllegalArgumentException("The columns of this matrix must equal the columns of the vector. "
					+ "This matrix has " + this.ncol + " columns and the vector has " + elements.length + " columns.");
		}
		final double[] product = new double[this.nrow];
		for (int i = 0; i < this.nrow; i++) {
			for (int k = 0; k < this.ncol; k++) {
				product[i] += this.data[i * this.ncol + k] * elements[k];
			}
		}
		return new Vector(product);
	}

	public MatrixOneD scaledBy(final double c) {
		final double[] scaled = new double[this.data.length];
		for (int i = 0; i < this.data.length; i++) {
			scaled[i] = this.data[i] * c;
		}
		return new MatrixOneD(this.nrow, this.ncol, scaled);
	}

	public MatrixOneD minus(final MatrixOneD other) {
		if (this.nrow != other.nrow() || this.ncol != other.ncol()) {
			throw new IllegalArgumentException(
					"The dimensions of this matrix must equal the dimensions of the other matrix. "
							+ "This matrix has dimension (" + this.nrow + ", " + this.ncol
							+ ") and the other matrix has dimension (" + other.nrow() + ", " + other.ncol() + ")");
		}
		final double[] minus = new double[nrow * ncol];
		final double[] otherData = other.data();
		for (int i = 0; i < nrow; i++) {
			for (int j = 0; j < ncol; j++) {
				minus[i * ncol + j] = this.data[i * ncol + j] - otherData[i * ncol + j];
			}
		}
		return new MatrixOneD(this.nrow, this.ncol, minus);
	}

	public boolean isSquare() {
		return this.nrow == this.ncol;
	}

	public MatrixOneD transpose() {
		final double[] transposedData = new double[this.data.length];
		for (int i = 0; i < this.nrow; i++) {
			for (int j = 0; j < this.ncol; j++) {
				transposedData[i + j * this.nrow] = this.data[j + i * ncol];
			}
		}
		return new MatrixOneD(this.ncol, this.nrow, transposedData);
	}

	public Vector getRow(int i) {
		double[] row = new double[this.ncol];
		int offset = this.ncol * i;
		System.arraycopy(this.data, offset, row, 0, row.length);
		return Vector.from(row);
	}

	public Vector getColumn(int j) {
		double[] col = new double[this.nrow];
		for (int i = 0; i < col.length; i++) {
			col[i] = this.data[i * this.ncol + j];
		}
		return Vector.from(col);
	}

	public MatrixOneD pushColumn(Vector newData) {
		if (newData.size() != this.nrow) {
			throw new IllegalArgumentException(
					"The number of elements of the new column must match the " + "number of columns of the matrix.");
		}
		double[][] thisData = data2D(Layout.BY_COLUMN);
		double[][] newMatrix = new double[this.ncol + 1][];
		newMatrix[0] = newData.elements();
		for (int i = 1; i < newMatrix.length; i++) {
			newMatrix[i] = thisData[i - 1].clone();
		}
		return new MatrixOneD(Layout.BY_COLUMN, newMatrix);
	}

	public MatrixOneD pushRow(Vector newData) {
		if (newData.size() != this.ncol) {
			throw new IllegalArgumentException(
					"The number of elements of the new row must match the " + "number of columns of the matrix.");
		}
		double[] newMatrix = new double[newData.size() + this.data.length];
		System.arraycopy(newData.elements(), 0, newMatrix, 0, newData.size());
		System.arraycopy(this.data, 0, newMatrix, newData.size(), this.data.length);
		return new MatrixOneD(this.nrow + 1, this.ncol, newMatrix);
	}

	public double[] diagonal() {
		final double[] diag = new double[Math.min(nrow, ncol)];
		for (int i = 0; i < diag.length; i++) {
			diag[i] = data[ncol * i + i];
		}
		return diag;
	}

	public double[] data() {
		return this.data.clone();
	}

	public double[][] data2D(Layout layout) {
		if (layout == Layout.BY_ROW) {
			return data2DRowMajor();
		}
		return data2DColumnMajor();
	}

	public double[][] data2D() {
		if (this.layout == Layout.BY_ROW) {
			return data2DRowMajor();
		}
		return data2DColumnMajor();
	}

	private double[][] data2DRowMajor() {
		final double[][] twoD = new double[this.nrow][this.ncol];
		for (int i = 0; i < nrow; i++) {
			System.arraycopy(this.data, i * ncol, twoD[i], 0, ncol);
		}
		return twoD;
	}

	private double[][] data2DColumnMajor() {
		final double[][] twoD = new double[this.ncol][this.nrow];
		for (int i = 0; i < ncol; i++) {
			for (int j = 0; j < nrow; j++) {
				twoD[i][j] = this.data[i + j * ncol];
			}
		}
		return twoD;
	}

	public MatrixOneD getSymmetricPart() {
		return this.plus(this.transpose()).scaledBy(0.5);
	}

	@Override
	public String toString() {
		String newLine = System.lineSeparator();
		StringBuilder representation = new StringBuilder(newLine);
		double[][] twoD = data2D(Layout.BY_ROW);
		for (int i = 0; i < this.nrow; i++) {
			representation.append(Arrays.toString(twoD[i])).append(newLine);
		}
		return representation.toString();
	}

	@Override
	public boolean equals(Object o) {
		if (this == o)
			return true;
		if (o == null || getClass() != o.getClass())
			return false;

		MatrixOneD matrix = (MatrixOneD) o;
		return nrow == matrix.nrow && ncol == matrix.ncol && Arrays.equals(data, matrix.data);
	}

	@Override
	public int hashCode() {
		int result = nrow;
		result = 31 * result + ncol;
		result = 31 * result + Arrays.hashCode(data);
		return result;
	}
	
	/**
     * Create a new identity matrix with the given dimension.
     *
     * @param n the dimension of the identity matrix.
     * @return a new identity matrix with the given dimension.
     */
    static MatrixOneD identity(final int n) {
        final double[] data = new double[n * n];
        for (int i = 0; i < n; i++) {
            data[i * n + i] = 1.0;
        }
        return new MatrixOneD(n, n, data);
    }
	
	/**
     * A class that allows one to start with an identity matrix, then set specific elements before creating
     * an immutable matrix.
     *
     * @author Jacob Rachiele
     */
    static final class IdentityBuilder {

        final int n;
        final double[] data;

        /**
         * Create a new builder with the given dimension.
         *
         * @param n the dimension of the matrix.
         */
        IdentityBuilder(final int n) {
            this.n = n;
            this.data = new double[n * n];
            for (int i = 0; i < n; i++) {
                this.data[i * n + i] = 1.0;
            }
        }

        /**
         * Set the matrix at the given coordinates to the provided value and return the builder.
         *
         * @param i     the row to set the value at.
         * @param j     the column to set the value at.
         * @param value the value to set.
         * @return the builder with the value set at the given coordinates.
         */
        public IdentityBuilder set(final int i, final int j, final double value) {
            ZeroBuilder.validateRow(i, this.n);
            ZeroBuilder.validateColumn(j, this.n);
            this.data[i * n + j] = value;
            return this;
        }

        /**
         * Create a new matrix using the data in this builder.
         *
         * @return a new matrix from this builder.
         */
        public MatrixOneD build() {
            return new MatrixOneD(n, n, data);
        }

        public IdentityBuilder setRow(final int i, final Vector row) {
            ZeroBuilder.validateRow(i, this.n);
            ZeroBuilder.validateSize(this.n, row);
            System.arraycopy(row.elements(), 0, this.data, i * n, n);
            return this;
        }

        public IdentityBuilder setColumn(final int j, final Vector column) {
            ZeroBuilder.validateColumn(j, this.n);
            ZeroBuilder.validateSize(this.n, column);
            for (int i = 0; i < this.n; i++) {
                this.data[i * n + j] = column.at(i);
            }
            return this;
        }

    }

    static final class ZeroBuilder {

        final int m;
        final int n;
        final double[] data;

        /**
         * Create a new builder with the given dimensions.
         *
         * @param m the number of rows of the matrix.
         * @param n the number of columns of the matrix.
         */
        ZeroBuilder(final int m, final int n) {
            this.m = m;
            this.n = n;
            this.data = new double[m * n];
        }

        public ZeroBuilder set(final int i, final int j, final double value) {
            validateRow(i, this.m);
            validateColumn(j, this.n);
            this.data[i * n + j] = value;
            return this;
        }

        public ZeroBuilder setRow(final int i, final Vector row) {
            validateRow(i, this.m);
            validateSize(this.n, row);
            System.arraycopy(row.elements(), 0, this.data, i * n, n);
            return this;
        }

        public ZeroBuilder setColumn(final int j, final Vector column) {
            validateColumn(j, this.n);
            validateSize(this.m, column);
            for (int i = 0; i < this.m; i++) {
                this.data[i * n + j] = column.at(i);
            }
            return this;
        }

        public MatrixOneD build() {
            return new MatrixOneD(m, n, data);
        }

        static void validateRow(int i, int m) {
            if (i >= m) {
                throw new IllegalArgumentException("The row index must be less than the number of rows, " +
                                                   "but the index, " + i + ", is greater than or equal to " + m);
            }
            if (i < 0) {
                throw new IllegalArgumentException("The row index must be greater than or equal to zero, " +
                                                   "but was " + i);
            }
        }

        static void validateColumn(int i, int n) {
            if (i >= n) {
                throw new IllegalArgumentException("The column index must be less than the number of columns, " +
                                                   "but the index, " + i + ", is greater than or equal to " + n);
            }
            if (i < 0) {
                throw new IllegalArgumentException("The column index must be greater than or equal to zero, " +
                                                   "but was " + i);
            }
        }

        static void validateSize(int n, Vector vector) {
            if (vector.size() != n) {
                throw new IllegalArgumentException("The vector must have " + n + " elements, " +
                                                   "but had " + vector.size());
            }
        }
    }
}
