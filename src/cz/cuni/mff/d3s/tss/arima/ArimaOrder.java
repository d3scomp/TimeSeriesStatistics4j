package cz.cuni.mff.d3s.tss.arima;


public class ArimaOrder {
	
	private com.github.signaflo.timeseries.model.arima.ArimaOrder order;
	
	public ArimaOrder(int p, int d, int q) {
		order = com.github.signaflo.timeseries.model.arima.ArimaOrder.order(p, d, q);
	}
	
	com.github.signaflo.timeseries.model.arima.ArimaOrder getOrder(){
		return order;
	}
}
