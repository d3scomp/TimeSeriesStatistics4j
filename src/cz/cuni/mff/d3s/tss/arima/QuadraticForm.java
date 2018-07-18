package cz.cuni.mff.d3s.tss.arima;

public class QuadraticForm {

	private final Vector x;
	private final MatrixOneD A;

	public QuadraticForm(Vector x, MatrixOneD A) {
		validateArguments(x, A);
		this.x = x;
		this.A = A;
	}

	/**
	 * Compute <em>x</em><sup>T</sup><em>A</em><em>x</em> and return the resulting
	 * value.
	 *
	 * @param x
	 *            the vector component of the quadratic form.
	 * @param A
	 *            the matrix component of the quadratic form.
	 * @return the result of <em>x</em><sup>T</sup><em>A</em><em>x</em>.
	 */
	public static double multiply(Vector x, MatrixOneD A) {
		validateArguments(x, A);
		int n = x.size();
		double result = 0.0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				result += x.at(i) * x.at(j) * A.get(i, j);
			}
		}
		return result;
	}

	private static void validateArguments(Vector x, MatrixOneD A) {
		if (!A.isSquare()) {
			throw new IllegalArgumentException("The matrix must be square.");
		}
		if (x.size() != A.nrow()) {
			throw new IllegalArgumentException(
					"The number of matrix columns must be the same" + " as the size of the vector.");
		}
	}

	/**
	 * Compute <em>x</em><sup>T</sup><em>A</em><em>x</em> and return the resulting
	 * value.
	 *
	 * @return the result of <em>x</em><sup>T</sup><em>A</em><em>x</em>.
	 */
	public double multiply() {
		return multiply(this.x, this.A);
	}

	@Override
	public boolean equals(Object o) {
		if (this == o)
			return true;
		if (o == null || getClass() != o.getClass())
			return false;

		QuadraticForm that = (QuadraticForm) o;

		if (!x.equals(that.x))
			return false;
		return A.equals(that.A);
	}

	@Override
	public int hashCode() {
		int result = x.hashCode();
		result = 31 * result + A.hashCode();
		return result;
	}
}
