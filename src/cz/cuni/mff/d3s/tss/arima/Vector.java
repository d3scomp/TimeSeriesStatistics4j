package cz.cuni.mff.d3s.tss.arima;

import java.util.Arrays;

public class Vector {
	private final double[] elements;

	/**
	 * Create a new vector using the provided elements.
	 *
	 * @param elements
	 *            the elements of the new vector.
	 */
	Vector(double... elements) {
		this.elements = elements.clone();
	}

	public double[] elements() {
		return this.elements.clone();
	}

	public double at(final int i) {
		return this.elements[i];
	}

	public int size() {
		return this.elements.length;
	}

	public Vector plus(final Vector other) {
		if (other.elements().length == 0) {
			return this;
		}
		final double[] summed = new double[this.size()];
		for (int i = 0; i < summed.length; i++) {
			summed[i] = this.elements[i] + other.at(i);
		}
		return new Vector(summed);
	}

	public Vector minus(final Vector other) {
		final double[] differenced = new double[this.size()];
		for (int i = 0; i < differenced.length; i++) {
			differenced[i] = this.elements[i] - other.at(i);
		}
		return new Vector(differenced);
	}

	public Vector minus(final double scalar) {
		final double[] differenced = new double[this.size()];
		for (int i = 0; i < differenced.length; i++) {
			differenced[i] = this.elements[i] - scalar;
		}
		return new Vector(differenced);
	}

	public Vector scaledBy(final double alpha) {
		final double[] scaled = new double[this.size()];
		for (int i = 0; i < scaled.length; i++) {
			scaled[i] = alpha * this.elements[i];
		}
		return new Vector(scaled);
	}

	public double dotProduct(final Vector other) {
		if (other.elements().length == 0) {
			throw new IllegalArgumentException("The dot product is undefined for zero length vectors");
		}
		double product = 0.0;
		for (int i = 0; i < elements.length; i++) {
			product += this.elements[i] * other.at(i);
		}
		return product;
	}

	Vector axpy(final Vector other, final double alpha) {
		final double[] result = new double[this.size()];
		for (int i = 0; i < result.length; i++) {
			result[i] = alpha * this.elements[i] + other.elements[i];
		}
		return new Vector(result);
	}

	public double norm() {
		return Math.sqrt(dotProduct(this));
	}

	public double sum() {
		double sum = 0.0;
		for (double element : elements) {
			sum += element;
		}
		return sum;
	}

	public double sumOfSquares() {
		final double[] squared = new double[elements.length];
        for (int i = 0; i < squared.length; i++) {
            squared[i] = elements[i] * elements[i];
        }
        double sum = 0.0;
		for (double element : squared) {
			sum += element;
		}
		return sum;
	}

	public Vector push(double value) {
		double[] newElements = new double[this.elements.length + 1];
		newElements[0] = value;
		System.arraycopy(this.elements, 0, newElements, 1, this.elements.length);
		return new Vector(newElements);
	}

	/**
     * Create a new vector from the given elements.
     *
     * @param elements the elements of the new vector.
     * @return a new vector with the given elements.
     */
    static Vector from(double... elements) {
        return new Vector(elements);
    }
    
    public MatrixOneD outerProduct(final Vector other) {
        double[] otherElements = other.elements();
        double[] product = new double[elements.length * otherElements.length];
        for (int i = 0; i < elements.length; i++) {
            for (int j = 0; j < otherElements.length; j++) {
                product[i * otherElements.length + j] = elements[i] * otherElements[j];
            }
        }
        return Matrix.create(elements.length, otherElements.length, product);

    }
    
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + Arrays.hashCode(elements);
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Vector other = (Vector) obj;
		return Arrays.equals(elements, other.elements);
	}

	@Override
	public String toString() {
		return "elements: " + Arrays.toString(elements);
	}
}
