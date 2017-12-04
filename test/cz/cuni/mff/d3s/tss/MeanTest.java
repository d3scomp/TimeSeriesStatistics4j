package cz.cuni.mff.d3s.tss;

import java.util.ArrayList;
import java.util.List;

public class MeanTest {

	public static void main(String[] args) {
		
		List<Integer> times = new ArrayList<>();
		List<Double> values = new ArrayList<>();
		
		CsvLoader.loadValues("MeanTest.csv", times, values);
		
		TimeSeries s = new TimeSeries(10, 1000);
		for (int i = 0; i < times.size(); i++) {
			s.addSample(values.get(i), times.get(i));
		}
		
		StudentsDistribution m = s.getMean();
		System.out.println("Real mean: 2  Real variance: 0.0009");
		System.out.println("Computed mean: " + m.getMean() + " Computed variance: " + m.getVariance());

		m.setAlpha(TTable.ALPHAS.ALPHA_0_05);

		System.out.println("alpha = " + m.getAlpha());
		System.out.println();
		System.out.println("m <= 1.94  ... " + m.isLessOrEqualTo(1.94) + "  (should be false)");
		System.out.println("m <= 1.96  ... " + m.isLessOrEqualTo(1.96) + "  (should be true)");
		System.out.println("m >= 2.06  ... " + m.isGreaterOrEqualTo(2.06) + "  (should be false)");
		System.out.println("m >= 2.04  ... " + m.isGreaterOrEqualTo(2.04) + "  (should be true)");
		System.out.println();
		System.out.println("m > 1.94  ... " + m.isGreaterThan(1.94) + "  ");
		System.out.println("m > 1.96  ... " + m.isGreaterThan(1.96) + "  ");
		System.out.println("m < 2.06  ... " + m.isLessThan(2.06) + "  ");
		System.out.println("m < 2.04  ... " + m.isLessThan(2.04) + "  ");

	}

}
