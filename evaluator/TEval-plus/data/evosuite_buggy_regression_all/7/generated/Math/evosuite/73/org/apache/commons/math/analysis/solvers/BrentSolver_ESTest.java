/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:58:38 GMT 2023
 */

package org.apache.commons.math.analysis.solvers;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.analysis.UnivariateRealFunction;
import org.apache.commons.math.analysis.polynomials.PolynomialFunction;
import org.apache.commons.math.analysis.solvers.BrentSolver;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BrentSolver_ESTest extends BrentSolver_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      BrentSolver brentSolver0 = new BrentSolver(polynomialFunction0);
      assertEquals(1.0E-6, brentSolver0.getAbsoluteAccuracy(), 0.01);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      // Undeclared exception!
      try { 
        brentSolver0.solve(0.0, 0.0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // endpoints do not specify an interval: [0, 0]
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      // Undeclared exception!
      try { 
        brentSolver0.solve(0.0, 0.0, 0.0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // invalid interval, initial value parameters:  lower=0, initial=0, upper=0
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      double[] doubleArray0 = new double[4];
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-2044.32825894), 1641.47, 0.0);
      assertEquals(0, brentSolver0.getIterationCount());
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      double[] doubleArray0 = new double[9];
      doubleArray0[1] = 1.5;
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, 0.0, Double.POSITIVE_INFINITY, 1.5);
      assertEquals(0.0, double0, 0.01);
      assertEquals(0, brentSolver0.getIterationCount());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      doubleArray0[1] = 2435.4560677006243;
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      BrentSolver brentSolver0 = new BrentSolver();
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-42.60105791), 1.5, 0.5);
      assertEquals(2, brentSolver0.getIterationCount());
      assertEquals(1.1102230246251565E-16, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      double[] doubleArray0 = new double[4];
      doubleArray0[0] = (-2052.08224);
      doubleArray0[1] = (-2052.08224);
      doubleArray0[2] = (-2052.08224);
      doubleArray0[3] = (-2052.08224);
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-2052.08224), (-1.0), (-1.0000000000285132));
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      doubleArray0[1] = (-3778.6);
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      BrentSolver brentSolver0 = new BrentSolver();
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-3778.6), 1641.47, (-2880.50820853));
      assertEquals(2, brentSolver0.getIterationCount());
      assertEquals((-2.2737367544323206E-13), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      double[] doubleArray0 = new double[4];
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-2052.08224), 1.1598035877661097E22);
      assertEquals((-2052.08224), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      double[] doubleArray0 = new double[4];
      doubleArray0[0] = (-2052.08224);
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      // Undeclared exception!
      try { 
        brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, 2.0, 1150.6);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // function values at endpoints do not have different signs.  Endpoints: [2, 1,150.6], Values: [-2,052.082, -2,052.082]
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      doubleArray0[4] = (-751.8950408450742);
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      BrentSolver brentSolver0 = new BrentSolver();
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, 1.0E-15, Double.POSITIVE_INFINITY);
      assertEquals(1.0E-15, double0, 0.01);
      assertEquals(0, brentSolver0.getIterationCount());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      doubleArray0[2] = 2435.4560677006243;
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      BrentSolver brentSolver0 = new BrentSolver();
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-2344.6691509), (-3.2386407945807193E-10));
      assertEquals((-3.2386407945807193E-10), double0, 0.01);
      assertEquals(0, brentSolver0.getIterationCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      double[] doubleArray0 = new double[4];
      doubleArray0[3] = Double.NaN;
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, Double.NaN, Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      double[] doubleArray0 = new double[4];
      doubleArray0[3] = (-2052.08224);
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      try { 
        brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-2052.08224), 1.1598035877661097E22);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Maximal number of iterations (100) exceeded
         //
         verifyException("org.apache.commons.math.analysis.solvers.BrentSolver", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      double[] doubleArray0 = new double[9];
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      brentSolver0.setFunctionValueAccuracy((-8.963314354116288E-4));
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-8.963314354116288E-4), 3240.427, 1968.3690581067542);
      assertEquals(31, brentSolver0.getIterationCount());
      assertEquals(2816.4076859304832, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      double[] doubleArray0 = new double[4];
      doubleArray0[3] = (-2052.08224);
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      UnivariateRealFunction univariateRealFunction0 = polynomialFunction0.derivative();
      double double0 = brentSolver0.solve(univariateRealFunction0, (-1154.25904052642), 5.000001692678779E-7, (-1.0000000000285132));
      assertEquals(20, brentSolver0.getIterationCount());
      assertEquals(1.6926787795898916E-13, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      double[] doubleArray0 = new double[4];
      doubleArray0[3] = (-2052.08224);
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-701.35), 1.0);
      assertEquals(79, brentSolver0.getIterationCount());
      assertEquals(3.6327112520639414E-7, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      double[] doubleArray0 = new double[4];
      brentSolver0.setRelativeAccuracy(Double.NaN);
      doubleArray0[3] = Double.NaN;
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      try { 
        brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-2044.32825894), 1641.47, 0.0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Maximal number of iterations (100) exceeded
         //
         verifyException("org.apache.commons.math.analysis.solvers.BrentSolver", e);
      }
  }
}