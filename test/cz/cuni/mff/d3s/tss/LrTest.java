package cz.cuni.mff.d3s.tss;

import java.util.ArrayList;
import java.util.List;

public class LrTest {
	
	
	public static void main(String[] args) {

		List<Integer> times = new ArrayList<>();
		List<Double> values = new ArrayList<>();
		
		CsvLoader.loadValues("LrTest.csv", times, values);
		
		TimeSeries s = new TimeSeries(20, 3000);
		for (int i = 0; i < times.size(); i++) {
			s.addSample(values.get(i), times.get(i));
		}
		
		StudentsDistribution a = s.getLra();
		System.out.println("alpha - Real mean: 2");
		System.out.println("alpha - Computed mean: " + a.getMean() + " Computed variance: " + a.getVariance());

		StudentsDistribution b = s.getLrb();
		System.out.println("beta - Real mean: 0.001");
		System.out.println("beta - Computed mean: " + b.getMean() + " Computed variance: " + b.getVariance());

		StudentsDistribution x5 = s.getLr(5000);
		System.out.println("x=5000 - Computed mean: " + x5.getMean() + " Computed variance: " + x5.getVariance());

		StudentsDistribution x10 = s.getLr(10000);
		System.out.println("x=10000 - Computed mean: " + x10.getMean() + " Computed variance: " + x10.getVariance());

		StudentsDistribution x60 = s.getLr(60000);
		System.out.println("x=60000 - Computed mean: " + x60.getMean() + " Computed variance: " + x60.getVariance());


	}
	
	
}
