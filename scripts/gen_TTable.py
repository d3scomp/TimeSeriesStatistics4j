#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Generates files with tabulated quantiles Student's t-distribution for alphas given on the command line.

Usage: gen_ttable <alpha 1> ... <alpha n>

If no alphas are given on the command line, it generates the table for: 0.05, 0.005, 0.00625, 0.001, 0.0125, 0.025

Example: gen_TTable 0.0125 0.025 0.05

"""

import numpy as np
from scipy.stats import t
import sys
import re


minorCount = 100
boost = 3


def getLimits(dfMax):
	base = 0
	for majorIdx in xrange(0, sys.maxint):
		minorStep = (1 << majorIdx * boost)
		base += minorStep * minorCount

		if base >= dfMax:
			break

	return (base)


dfMax = getLimits(1e9)


def getICDFTable(alpha):
	icdf = np.zeros((1, minorCount))
	base = 0
	for majorIdx in xrange(0, sys.maxint):
		minorStep = (1 << majorIdx * boost)
		for minorIdx in xrange(0, minorCount):
			df = (minorIdx + 1) * minorStep + base
			icdf[majorIdx, minorIdx] = t.ppf(alpha, df)

		base += minorStep * minorCount

		if base >= dfMax:
			break

		icdf.resize((majorIdx + 2, minorCount))
		
	return icdf.reshape((-1))

print 'Generation started ...'
	
alphas = [float(alphaStr) for alphaStr in sys.argv[1:]]
if len(alphas) == 0:
	alphas = [0.05, 0.005, 0.00625, 0.001, 0.0125, 0.025]

dotToUnderscore = re.compile('\.')

enumAlphas = []
for alpha in alphas:
	enumAlphas.append('ALPHA_{}({})'.format(dotToUnderscore.sub('_', str(alpha)), alpha))

out = open('../src/cz/cuni/mff/d3s/tss/TTable.java', 'w')
out.write('''/*
 * TTable.java
 * Generated by gen_TTable.py
 *
 * Quantiles of the Student's t-distribution
 */

package cz.cuni.mff.d3s.tss;


public class TTable {{

	public enum ALPHAS {{
		{0};
	
		private final double value;
	
		ALPHAS(double value) {{
			this.value = value;
		}}
		
		public double getValue() {{
			return value;
		}}
	}};


	public static final int minor_count = {1};
	public static final int boost = {2};
	public static final int df_max = {3};

	public static final double icdf[][] = {{
{4}
	}};
}}
'''.format(
	',\n\t\t'.join(enumAlphas),
	minorCount, boost, dfMax, 
	',\n\n'.join( ['\t\t\t{' + ','.join([str(val) for val in getICDFTable(alpha)]) + '}' for alpha in alphas] )
))


out.close()

print 'done.'
