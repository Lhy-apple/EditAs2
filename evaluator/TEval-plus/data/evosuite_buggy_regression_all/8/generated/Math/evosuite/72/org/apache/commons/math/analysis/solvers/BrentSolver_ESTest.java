/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:09:13 GMT 2023
 */

package org.apache.commons.math.analysis.solvers;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.analysis.UnivariateRealFunction;
import org.apache.commons.math.analysis.polynomials.PolynomialFunction;
import org.apache.commons.math.analysis.polynomials.PolynomialFunctionLagrangeForm;
import org.apache.commons.math.analysis.solvers.BrentSolver;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BrentSolver_ESTest extends BrentSolver_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      // Undeclared exception!
      try { 
        brentSolver0.solve((-15.435371678165954), (-15.435371678165954));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // endpoints do not specify an interval: [-15.435, -15.435]
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      // Undeclared exception!
      try { 
        brentSolver0.solve(2.0, 2.0, 2.0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // invalid interval, initial value parameters:  lower=2, initial=2, upper=2
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      doubleArray0[0] = 1010.03;
      doubleArray0[1] = Double.NaN;
      doubleArray0[2] = 0.5;
      PolynomialFunctionLagrangeForm polynomialFunctionLagrangeForm0 = new PolynomialFunctionLagrangeForm(doubleArray0, doubleArray0);
      BrentSolver brentSolver0 = new BrentSolver(polynomialFunctionLagrangeForm0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunctionLagrangeForm0, 1.2515482073992253E-6, 0.5, 4.811111547861021E-6);
      assertEquals(1, brentSolver0.getIterationCount());
      assertEquals(0.4999995, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      double[] doubleArray0 = new double[11];
      BrentSolver brentSolver0 = new BrentSolver();
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, 0.0, 974.870437280648, 1.4568171);
      assertEquals(1.4568171, double0, 0.01);
      assertEquals(0, brentSolver0.getIterationCount());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      double[] doubleArray0 = new double[11];
      doubleArray0[6] = 9.998575067686546E-15;
      BrentSolver brentSolver0 = new BrentSolver();
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, 0.0, 974.870437280648, 1.4568171);
      assertEquals(0.0, double0, 0.01);
      assertEquals(0, brentSolver0.getIterationCount());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      doubleArray0[3] = 2.729403302570096E15;
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      BrentSolver brentSolver0 = new BrentSolver();
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-576.597901476929), 2.729403302570096E15, 1931.7454987726);
      assertEquals(93, brentSolver0.getIterationCount());
      assertEquals((-1.0100124592789657E-7), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      double[] doubleArray0 = new double[9];
      doubleArray0[3] = 1.0E-14;
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-125.04972526569759), 1.0E-14, (-1.0));
      assertEquals(0, brentSolver0.getIterationCount());
      assertEquals(1.0E-56, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      doubleArray0[3] = 2.729403302570095E15;
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      BrentSolver brentSolver0 = new BrentSolver();
      try { 
        brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-1574.0258488998622), 2.729403302570095E15, (-83.943795537));
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Maximal number of iterations (100) exceeded
         //
         verifyException("org.apache.commons.math.analysis.solvers.BrentSolver", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      double[] doubleArray0 = new double[11];
      doubleArray0[1] = 9.998575067686546E-15;
      BrentSolver brentSolver0 = new BrentSolver();
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      UnivariateRealFunction univariateRealFunction0 = polynomialFunction0.derivative();
      // Undeclared exception!
      try { 
        brentSolver0.solve(univariateRealFunction0, 0.0, 3530.742972691976, 9.998575067686546E-15);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // function values at endpoints do not have different signs.  Endpoints: [0, 3,530.743], Values: [0, 0]
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      BrentSolver brentSolver0 = new BrentSolver();
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-1881.42264), 2459.06089);
      assertEquals((-1881.42264), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      double[] doubleArray0 = new double[15];
      doubleArray0[1] = 9.998575067686546E-15;
      BrentSolver brentSolver0 = new BrentSolver();
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, 9.998575067686546E-15, 1524.6413474852989);
      assertEquals(9.998575067686546E-15, double0, 0.01);
      assertEquals(0, brentSolver0.getIterationCount());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      double[] doubleArray0 = new double[12];
      doubleArray0[1] = 9.998575067686546E-15;
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      BrentSolver brentSolver0 = new BrentSolver(polynomialFunction0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-2967.866822), (-3.5107628564764965E-6));
      assertEquals(0, brentSolver0.getIterationCount());
      assertEquals((-3.5107628564764965E-6), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      double[] doubleArray0 = new double[18];
      doubleArray0[3] = 2.7294033025700945E15;
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      BrentSolver brentSolver0 = new BrentSolver();
      // Undeclared exception!
      try { 
        brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, 227.51496288184634, 2.7294033025700945E15);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // function values at endpoints do not have different signs.  Endpoints: [227.515, 2,729,403,302,570,094.5], Values: [32,143,828,679,207,393,000,000, 55,497,171,710,286,320,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000]
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      double[] doubleArray0 = new double[8];
      doubleArray0[3] = 2.7294033025701225E15;
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      BrentSolver brentSolver0 = new BrentSolver();
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, Double.NaN, 2.7294033025701225E15);
      assertEquals(2.7294033025701225E15, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      double[] doubleArray0 = new double[15];
      doubleArray0[1] = (-0.41864476657801364);
      BrentSolver brentSolver0 = new BrentSolver();
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-0.41864476657801364), 1487.9077104564317);
      assertEquals(1, brentSolver0.getIterationCount());
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      double[] doubleArray0 = new double[15];
      doubleArray0[1] = (-0.41864476657801364);
      BrentSolver brentSolver0 = new BrentSolver();
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      brentSolver0.setFunctionValueAccuracy((-0.41864476657801364));
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-0.41864476657801364), 1487.9077104564317);
      assertEquals(20, brentSolver0.getIterationCount());
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      doubleArray0[3] = Double.POSITIVE_INFINITY;
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      BrentSolver brentSolver0 = new BrentSolver();
      brentSolver0.setAbsoluteAccuracy(Double.NaN);
      try { 
        brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-1340.6430467690761), 3.7652962974407355E-12);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Maximal number of iterations (100) exceeded
         //
         verifyException("org.apache.commons.math.analysis.solvers.BrentSolver", e);
      }
  }
}