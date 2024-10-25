/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:33:50 GMT 2023
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
      double[] doubleArray0 = new double[4];
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      BrentSolver brentSolver0 = new BrentSolver(polynomialFunction0);
      assertEquals(1.0E-14, brentSolver0.getRelativeAccuracy(), 0.01);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      // Undeclared exception!
      try { 
        brentSolver0.solve(518.43454147883, 518.43454147883);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // endpoints do not specify an interval: [518.435, 518.435]
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      // Undeclared exception!
      try { 
        brentSolver0.solve((-2191.3794), (-2191.3794), (-2191.3794));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // invalid interval, initial value parameters:  lower=-2,191.379, initial=-2,191.379, upper=-2,191.379
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      double[] doubleArray0 = new double[7];
      doubleArray0[1] = (-0.31250328355524437);
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-2740.3224304), (-0.31250328355524437), (-2.158501899962518));
      assertEquals(21, brentSolver0.getIterationCount());
      assertEquals((-0.31250328355524437), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      double[] doubleArray0 = new double[12];
      BrentSolver brentSolver0 = new BrentSolver();
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, 0.0, 5079.0, 2307.604374796);
      assertEquals(0, brentSolver0.getIterationCount());
      assertEquals(2307.604374796, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      double[] doubleArray0 = new double[8];
      doubleArray0[2] = (-2741.199773159562);
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, 1.0E-15, 1018.11256119241, 1.8459987164447555);
      assertEquals((-2.7411997731595624E-27), double0, 0.01);
      assertEquals(0, brentSolver0.getIterationCount());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      double[] doubleArray0 = new double[12];
      doubleArray0[3] = (-2191.3794);
      BrentSolver brentSolver0 = new BrentSolver();
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-2191.3794), 5079.0, 2307.604374796);
      assertEquals(91, brentSolver0.getIterationCount());
      assertEquals((-5.490504071436738E-7), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      double[] doubleArray0 = new double[8];
      doubleArray0[2] = (-0.31250328355524437);
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-1.0), 1.0E-15, (-0.31250328355524437));
      assertEquals((-3.1250328355524442E-31), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      doubleArray0[1] = (-0.31250328355524437);
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      BrentSolver brentSolver0 = new BrentSolver();
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-0.3125032835552706), 649.0088380892, (-0.31250328355524437));
      assertEquals(1, brentSolver0.getIterationCount());
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      BrentSolver brentSolver0 = new BrentSolver();
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-2176.8544295212187), Double.POSITIVE_INFINITY);
      assertEquals((-2176.8544295212187), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      double[] doubleArray0 = new double[6];
      doubleArray0[2] = (-2760.82989326675);
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-3003.78368), 1.0E-15);
      assertEquals(1.0E-15, double0, 0.01);
      assertEquals(0, brentSolver0.getIterationCount());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      double[] doubleArray0 = new double[7];
      doubleArray0[2] = (-0.31250328355524437);
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, 1.0E-15, 1.5);
      assertEquals(1.0E-15, double0, 0.01);
      assertEquals(0, brentSolver0.getIterationCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      doubleArray0[2] = (-2176.8544295212187);
      BrentSolver brentSolver0 = new BrentSolver();
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      // Undeclared exception!
      try { 
        brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-2176.8544295212187), Double.POSITIVE_INFINITY);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // function values at endpoints do not have different signs.  Endpoints: [-2,176.854, \u221E], Values: [-10,315,449,652.219, -\u221E]
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      doubleArray0[3] = (-2176.8544295212187);
      BrentSolver brentSolver0 = new BrentSolver();
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      try { 
        brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-2176.8544295212187), Double.POSITIVE_INFINITY);
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
      double[] doubleArray0 = new double[10];
      doubleArray0[6] = (-0.31250328355524437);
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, 1.8651746813699288E-14, Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BrentSolver brentSolver0 = new BrentSolver();
      double[] doubleArray0 = new double[8];
      doubleArray0[0] = (-1258.7555391338467);
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      double double0 = brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-1258.7555391338467), 1.0E-6, (-2.158501899962518));
      assertEquals(22, brentSolver0.getIterationCount());
      assertEquals((-0.7194998097389692), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      double[] doubleArray0 = new double[12];
      doubleArray0[3] = (-2191.3794);
      BrentSolver brentSolver0 = new BrentSolver();
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      brentSolver0.setAbsoluteAccuracy(Double.NaN);
      try { 
        brentSolver0.solve((UnivariateRealFunction) polynomialFunction0, (-2191.3794), 5079.0, 2307.604374796);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Maximal number of iterations (100) exceeded
         //
         verifyException("org.apache.commons.math.analysis.solvers.BrentSolver", e);
      }
  }
}
