/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:33:05 GMT 2023
 */

package org.apache.commons.math.stat.regression;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.stat.regression.SimpleRegression;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SimpleRegression_ESTest extends SimpleRegression_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SimpleRegression simpleRegression0 = new SimpleRegression();
      simpleRegression0.getIntercept();
      assertEquals(Double.NaN, simpleRegression0.getSlope(), 0.01);
      assertEquals(Double.NaN, simpleRegression0.getSumSquaredErrors(), 0.01);
      assertEquals(0L, simpleRegression0.getN());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SimpleRegression simpleRegression0 = new SimpleRegression();
      // Undeclared exception!
      try { 
        simpleRegression0.getSignificance();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // degrees of freedom must be positive.
         //
         verifyException("org.apache.commons.math.distribution.TDistributionImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      SimpleRegression simpleRegression0 = new SimpleRegression();
      simpleRegression0.getR();
      assertEquals(Double.NaN, simpleRegression0.getTotalSumSquares(), 0.01);
      assertEquals(0L, simpleRegression0.getN());
      assertEquals(Double.NaN, simpleRegression0.getSumSquaredErrors(), 0.01);
      assertEquals(Double.NaN, simpleRegression0.getRegressionSumSquares(), 0.01);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SimpleRegression simpleRegression0 = new SimpleRegression();
      simpleRegression0.predict(Double.NaN);
      assertEquals(Double.NaN, simpleRegression0.getSumSquaredErrors(), 0.01);
      assertEquals(Double.NaN, simpleRegression0.getRegressionSumSquares(), 0.01);
      assertEquals(0L, simpleRegression0.getN());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SimpleRegression simpleRegression0 = new SimpleRegression();
      double double0 = simpleRegression0.getRegressionSumSquares();
      assertEquals(Double.NaN, double0, 0.01);
      assertEquals(0L, simpleRegression0.getN());
      assertEquals(Double.NaN, simpleRegression0.getSumSquaredErrors(), 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SimpleRegression simpleRegression0 = new SimpleRegression();
      simpleRegression0.clear();
      assertEquals(Double.NaN, simpleRegression0.getSumSquaredErrors(), 0.01);
      assertEquals(0L, simpleRegression0.getN());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleRegression simpleRegression0 = new SimpleRegression();
      long long0 = simpleRegression0.getN();
      assertEquals(Double.NaN, simpleRegression0.getSumSquaredErrors(), 0.01);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SimpleRegression simpleRegression0 = new SimpleRegression();
      simpleRegression0.getInterceptStdErr();
      assertEquals(0L, simpleRegression0.getN());
      assertEquals(Double.NaN, simpleRegression0.getSumSquaredErrors(), 0.01);
      assertEquals(Double.NaN, simpleRegression0.getMeanSquareError(), 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SimpleRegression simpleRegression0 = new SimpleRegression();
      // Undeclared exception!
      try { 
        simpleRegression0.getSlopeConfidenceInterval();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // degrees of freedom must be positive.
         //
         verifyException("org.apache.commons.math.distribution.TDistributionImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleRegression simpleRegression0 = new SimpleRegression();
      double[][] doubleArray0 = new double[4][0];
      double[] doubleArray1 = new double[5];
      doubleArray0[0] = doubleArray1;
      doubleArray0[1] = doubleArray0[0];
      doubleArray0[2] = doubleArray1;
      doubleArray0[3] = doubleArray0[1];
      simpleRegression0.addData(doubleArray0);
      double double0 = simpleRegression0.getR();
      assertEquals(4L, simpleRegression0.getN());
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleRegression simpleRegression0 = new SimpleRegression();
      double[][] doubleArray0 = new double[4][0];
      double[] doubleArray1 = new double[5];
      doubleArray0[0] = doubleArray1;
      doubleArray0[1] = doubleArray1;
      doubleArray0[2] = doubleArray0[0];
      doubleArray0[3] = doubleArray1;
      simpleRegression0.addData(doubleArray0);
      double double0 = simpleRegression0.getSlopeStdErr();
      assertEquals(4L, simpleRegression0.getN());
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleRegression simpleRegression0 = new SimpleRegression();
      simpleRegression0.addData((-2980.0), 2762.35601);
      double[][] doubleArray0 = new double[4][0];
      double[] doubleArray1 = new double[5];
      doubleArray0[0] = doubleArray1;
      doubleArray0[1] = doubleArray1;
      doubleArray0[2] = doubleArray0[1];
      doubleArray0[3] = doubleArray0[0];
      simpleRegression0.addData(doubleArray0);
      double double0 = simpleRegression0.getR();
      assertEquals(5L, simpleRegression0.getN());
      assertEquals((-0.9999999999999998), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SimpleRegression simpleRegression0 = new SimpleRegression();
      // Undeclared exception!
      try { 
        simpleRegression0.getSlopeConfidenceInterval(1001.885420847);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.stat.regression.SimpleRegression", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SimpleRegression simpleRegression0 = new SimpleRegression();
      // Undeclared exception!
      try { 
        simpleRegression0.getSlopeConfidenceInterval(0.0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.stat.regression.SimpleRegression", e);
      }
  }
}