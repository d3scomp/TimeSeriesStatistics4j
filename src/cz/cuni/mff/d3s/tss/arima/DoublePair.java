package cz.cuni.mff.d3s.tss.arima;

public class DoublePair implements Comparable<DoublePair> {

	private final double first;
	private final double second;

	public DoublePair(double first, double second) {
		this.first = first;
		this.second = second;
	}

	public double first() {
		return this.first;
	}

	public double second() {
		return this.second;
	}

	/**
	 * Compare this pair to another pair for lexicographic ordering. The algorithm
	 * was adapted from the top answer <a target="_blank" href=
	 * "https://stackoverflow.com/questions/5292303/how-does-tuple-comparison-work-in-python">here.</a>
	 *
	 * @param otherPair
	 *            the pair to compare this one to.
	 * @return an integer value satisfying the {@link Comparable#compareTo(Object)}
	 *         contract.
	 */
	@Override
	public int compareTo(DoublePair otherPair) {
		int result = Double.compare(this.first, otherPair.first);
		if (result != 0) {
			return result;
		}
		return Double.compare(this.second, otherPair.second);
	}

	@Override
	public boolean equals(Object o) {
		if (this == o)
			return true;
		if (o == null || getClass() != o.getClass())
			return false;

		DoublePair that = (DoublePair) o;

		if (Double.compare(that.first, first) != 0)
			return false;
		return Double.compare(that.second, second) == 0;
	}

	@Override
	public int hashCode() {
		int result;
		long temp;
		temp = Double.doubleToLongBits(first);
		result = (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(second);
		result = 31 * result + (int) (temp ^ (temp >>> 32));
		return result;
	}

	@Override
	public String toString() {
		return "(" + Double.toString(first) + ", " + Double.toString(second) + ")";
	}
}
