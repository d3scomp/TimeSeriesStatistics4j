package cz.cuni.mff.d3s.tss.arima;

/**
 * The order of an ARIMA model, consisting of the number of autoregressive and moving average parameters, along with
 * the degree of differencing and flags indicating whether or not the model includes a constant and/or drift term.
 * This class is immutable and thread-safe.
 *
 * @author Jacob Rachiele
 */
public class ArimaOrder {

	/**
     * An indicator for whether an ARIMA model has a constant term.
     */
    public enum Constant { // TODO: change to yes/no
        INCLUDE(1), EXCLUDE(0);

        private final int constant;

        Constant(int constant) {
            this.constant = constant;
        }

        public int asInt() {
            return constant;
        }

        public boolean include() {
            return this == INCLUDE;
        }
    }
    
    /**
     * An indicator for whether an ARIMA model has a drift term.
     */
    enum Drift { // TODO: change to yes/no
        INCLUDE(1), EXCLUDE(0);

        private final int drift;

        Drift(int drift) {
            this.drift = drift;
        }

        public int asInt() {
            return drift;
        }

        public boolean include() {
            return this == INCLUDE;
        }
    }
	
    private final int p;
    private final int d;
    private final int q;
    private final int P;
    private final int D;
    private final int Q;
    private final Constant constant;
    private final Drift drift;
    private final int sumARMA;
    private final int npar;
    private final int numRegressors;

    ArimaOrder(final int p, final int d, final int q, final int P, final int D, final int Q,
               final Constant constant, final Drift drift) {
        this.p = p;
        this.d = d;
        this.q = q;
        this.P = P;
        this.D = D;
        this.Q = Q;
        this.constant = constant;
        this.drift = drift;
        this.sumARMA = this.p + this.q + this.P + this.Q;
        this.numRegressors = constant.asInt() + drift.asInt();
        this.npar = sumARMA + numRegressors;
    }

    /**
     * Create and return a new non-seasonal model order with the given number of coefficients. A constant will be fit
     * only if d is equal to 0.
     *
     * @param p the number of non-seasonal autoregressive coefficients.
     * @param d the degree of non-seasonal differencing.
     * @param q the number of non-seasonal moving-average coefficients.
     * @return a new ARIMA model order.
     */
    public static ArimaOrder order(final int p, final int d, final int q) {
        return order(p, d, q, 0, 0, 0);
    }

    /**
     * Create and return a new non-seasonal model order with the given number of coefficients and indication of
     * whether or not to fit a constant.
     *
     * @param p        the number of non-seasonal autoregressive coefficients.
     * @param d        the degree of non-seasonal differencing.
     * @param q        the number of non-seasonal moving-average coefficients.
     * @param constant determines whether or not a constant is fitted with the model.
     *                 If d = 0, then the constant is interpreted as the model mean.
     *                 If d = 1, then the constant is interpreted as a drift term. Finally, if
     *                 the constant is set to be included but d &gt; 1, then the constant term will
     *                 be ignored and a warning logged.
     * @return         a new ARIMA model order.
     */
    public static ArimaOrder order(final int p, final int d, final int q, final Constant constant) {
        return order(p, d, q, 0, 0, 0, constant);
    }

    /**
     * Create and return a new non-seasonal model order with the given number of coefficients and indication of
     * whether or not to fit a drift term. A constant representing the model mean will be fit only if both
     * the degree of differencing is zero and no drift term is included. Note that if the degree of differencing
     * is greater than 1, then the drift parameter is ignored.
     *
     * @param p        the number of non-seasonal autoregressive coefficients.
     * @param d        the degree of non-seasonal differencing.
     * @param q        the number of non-seasonal moving-average coefficients.
     * @param drift    determines whether or not a drift term is fitted with the model.
     * @return         a new ARIMA model order.
     */
    public static ArimaOrder order(final int p, final int d, final int q, final Drift drift) {
        return order(p, d, q, 0, 0, 0, drift);
    }

    /**
     * Create and return a new non-seasonal model order with the given number of coefficients and indication of
     * whether or not to fit a constant and/or drift term. If the degree of differencing is greater than 0, then
     * only a drift or constant term may be included.
     *
     * @param p        the number of non-seasonal autoregressive coefficients.
     * @param d        the degree of non-seasonal differencing.
     * @param q        the number of non-seasonal moving-average coefficients.
     * @param constant determines whether or not a constant is fitted with the model.
     * @param drift    determines whether or not a drift term is fitted with the model.
     * @return a new ARIMA model order.
     *
     * @throws IllegalArgumentException if the degree of differencing is greater than zero
     *                                  and both a drift term and constant are set to be included.
     */
    public static ArimaOrder order(final int p, final int d, final int q, final Constant constant,
                                   final Drift drift) {
        return order(p, d, q, 0, 0, 0, constant, drift);
    }

    /**
     * Create a new model order using the provided number of autoregressive and moving-average parameters,
     * as well as the degrees of differencing. A constant will be fit only if both d and D are equal to 0.
     *
     * @param p the number of non-seasonal autoregressive coefficients.
     * @param d the degree of non-seasonal differencing.
     * @param q the number of non-seasonal moving-average coefficients.
     * @param P the number of seasonal autoregressive coefficients.
     * @param D the degree of seasonal differencing.
     * @param Q the number of seasonal moving-average coefficients.
     * @return a new ARIMA model order.
     */
    public static ArimaOrder order(final int p, final int d, final int q, final int P, final int D, final int Q) {
        final Constant constant;
        if ((d + D) == 0) {
            /* A constant term will be automatically fit to the model
               since the degree of differencing is 0. */
            constant = Constant.INCLUDE;
        } else {
            /* A constant term will not be fit since the degree of differencing is greater than 0. */
            constant = Constant.EXCLUDE;
        }
        return new ArimaOrder(p, d, q, P, D, Q, constant, Drift.EXCLUDE);
    }

    /**
     * Create a new ArimaOrder using the provided number of autoregressive and moving-average parameters,
     * as well as the degrees of differencing and indication of whether or not to fit a constant.
     *
     * @param p        the number of non-seasonal autoregressive coefficients.
     * @param d        the degree of non-seasonal differencing.
     * @param q        the number of non-seasonal moving-average coefficients.
     * @param P        the number of seasonal autoregressive coefficients.
     * @param D        the degree of seasonal differencing.
     * @param Q        the number of seasonal moving-average coefficients.
     * @param constant determines whether or not a constant is fitted with the model.
     *                 If d = 0, then the constant is interpreted as the model mean.
     *                 If d = 1, then the constant is interpreted as a drift term. Finally, if
     *                 the constant is set to be included but d &gt; 1, then the constant term will
     *                 be ignored and a warning logged.
     * @return a new ARIMA model order.
     */
    public static ArimaOrder order(final int p, final int d, final int q, final int P, final int D, final int Q,
                                   final Constant constant) {
        Drift driftTerm = Drift.EXCLUDE;
        Constant updatedConstant = constant;
        if ((d + D) == 1 && constant.include()) {
            /* The constant will be interpreted as a drift term
               since the degree of differencing is equal to 1. */
            driftTerm = Drift.INCLUDE;
            updatedConstant = Constant.EXCLUDE;
        }
        else if ((d + D) > 1 && constant.include()) {
            /* The constant term will be ignored since the degree of differencing is greater than 1. */
            driftTerm = Drift.EXCLUDE;
            updatedConstant = Constant.EXCLUDE;
        }
        return new ArimaOrder(p, d, q, P, D, Q, updatedConstant, driftTerm);
    }

    /**
     * Create a new model order using the provided number of autoregressive and moving-average parameters, as well
     * as the degrees of differencing and indication of whether or not to fit a drift term. A constant will be included
     * only if both d and D are equal to 0.
     *
     * @param p        the number of non-seasonal autoregressive coefficients.
     * @param d        the degree of non-seasonal differencing.
     * @param q        the number of non-seasonal moving-average coefficients.
     * @param P        the number of seasonal autoregressive coefficients.
     * @param D        the degree of seasonal differencing.
     * @param Q        the number of seasonal moving-average coefficients.
     * @param drift    determines whether or not a drift term is fitted with the model.
     *                 This parameter is ignored if the degree of differencing is greater than 1.
     * @return a new ARIMA model order.
     */
    public static ArimaOrder order(final int p, final int d, final int q, final int P, final int D, final int Q,
                                   final Drift drift) {
        Constant constant = ((d + D) > 0)? Constant.EXCLUDE : Constant.INCLUDE;
        Drift actualDrift;
        if ((d + D) > 1 && drift.include()) {
            /* Drift term will be excluded since the degree of differencing is greater than 1. */
            actualDrift = Drift.EXCLUDE;
        } else {
            actualDrift = drift;
        }
        return new ArimaOrder(p, d, q, P, D, Q, constant, actualDrift);
    }

    /**
     * Create a new model order using the provided number of autoregressive and moving-average parameters, as well as
     * the degrees of differencing and indication of whether or not to fit a constant and/or a drift term.
     *
     * @param p        the number of non-seasonal autoregressive coefficients.
     * @param d        the degree of non-seasonal differencing.
     * @param q        the number of non-seasonal moving-average coefficients.
     * @param P        the number of seasonal autoregressive coefficients.
     * @param D        the degree of seasonal differencing.
     * @param Q        the number of seasonal moving-average coefficients.
     * @param constant determines whether or not a constant is fitted with the model.
     * @param drift    determines whether or not a drift term is fitted with the model.
     * @return a new ARIMA model order.
     *
     * @throws IllegalArgumentException if the degree of differencing (seasonal or non-seasonal) is greater than zero
     *                                  and both a drift term and constant are included.
     */
    public static ArimaOrder order(final int p, final int d, final int q, final int P, final int D, final int Q,
                                   final Constant constant, final Drift drift) {
        if (((d + D) > 0) && constant.include() && drift.include()) {
            throw new IllegalArgumentException("Arima model cannot be fit with both a constant and a" +
                                               " drift term when the degree of differencing is greater" +
                                               " than zero.");
        }
        return new ArimaOrder(p, d, q, P, D, Q, constant, drift);
    }

    int p() {
        return p;
    }

    int q() {
        return q;
    }

    int P() {
        return P;
    }

    int Q() {
        return Q;
    }

    int d() {
        return d;
    }

    int D() {
        return D;
    }

    Constant constant() {
        return this.constant;
    }

    Drift drift() {
        return this.drift;
    }

    // This returns the total number of nonseasonal and seasonal ARMA parameters.
    int sumARMA() {
        return this.sumARMA;
    }

    int npar() {
        return this.npar;
    }

    int numRegressors() {
        return this.numRegressors;
    }

    @Override
    public String toString() {
        boolean isSeasonal = P > 0 || Q > 0 || D > 0;
        StringBuilder builder = new StringBuilder();
        if (isSeasonal) {
            builder.append("Seasonal ");
        }
        builder.append("ARIMA (").append(p).append(", ").append(d).append(", ").append(q);
        if (isSeasonal) {
            builder.append(") x (").append(P).append(", ").append(D).append(", ").append(Q);
        }
        builder.append(") with").append((constant == Constant.INCLUDE) ? " a constant" : " no constant");
        builder.append((drift == Drift.INCLUDE) ? " and a drift term" : "");
        return builder.toString();
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + D;
        result = prime * result + P;
        result = prime * result + Q;
        result = prime * result + constant.asInt();
        result = prime * result + drift.asInt();
        result = prime * result + d;
        result = prime * result + p;
        result = prime * result + q;
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null) return false;
        if (getClass() != obj.getClass()) return false;
        ArimaOrder other = (ArimaOrder) obj;
        if (D != other.D) return false;
        if (P != other.P) return false;
        if (Q != other.Q) return false;
        if (constant != other.constant) return false;
        if (drift != other.drift) return false;
        if (d != other.d) return false;
        return p == other.p && q == other.q;
    }
}
