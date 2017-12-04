package cz.cuni.mff.d3s.tss;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.List;
import java.util.Scanner;

public class CsvLoader {

	public static void loadValues(String fileName, List<Integer> times, List<Double> values) {
		File file = new File(fileName);
        try(Scanner inputStream = new Scanner(file)){
        	// Read times
            String line = inputStream.next();
            String[] entries = line.split(",");
            for(String entry : entries) {
            	times.add(Integer.parseInt(entry));
            }
        	// Read values
            line = inputStream.next();
            entries = line.split(",");
            for(String entry : entries) {
            	values.add(Double.parseDouble(entry));
            }
        
        }catch (FileNotFoundException e) {
        	System.err.println(e.getMessage());;
        }
	}
}
