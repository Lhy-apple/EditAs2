/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:24:06 GMT 2023
 */

package org.apache.commons.math.stat.descriptive.moment;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.stat.descriptive.moment.SecondMoment;
import org.apache.commons.math.stat.descriptive.moment.Variance;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Variance_ESTest extends Variance_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Variance variance0 = new Variance();
      Variance variance1 = new Variance(variance0);
      assertTrue(variance1.isBiasCorrected());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SecondMoment secondMoment0 = new SecondMoment();
      Variance variance0 = new Variance(secondMoment0);
      double[] doubleArray0 = new double[3];
      variance0.incrementAll(doubleArray0);
      assertTrue(variance0.isBiasCorrected());
      assertEquals(0L, variance0.getN());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Variance variance0 = new Variance();
      variance0.getN();
      assertTrue(variance0.isBiasCorrected());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SecondMoment secondMoment0 = new SecondMoment();
      Variance variance0 = new Variance(false, secondMoment0);
      assertFalse(variance0.isBiasCorrected());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Variance variance0 = new Variance();
      variance0.setBiasCorrected(true);
      assertTrue(variance0.isBiasCorrected());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Variance variance0 = new Variance();
      Variance variance1 = variance0.copy();
      assertNotSame(variance1, variance0);
      assertTrue(variance1.isBiasCorrected());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Variance variance0 = new Variance();
      boolean boolean0 = variance0.isBiasCorrected();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Variance variance0 = new Variance();
      double[] doubleArray0 = new double[15];
      variance0.incrementAll(doubleArray0);
      double double0 = variance0.getResult();
      assertEquals(15L, variance0.getN());
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Variance variance0 = new Variance();
      double double0 = variance0.getResult();
      assertEquals(Double.NaN, double0, 0.01);
      assertTrue(variance0.isBiasCorrected());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Variance variance0 = new Variance();
      double[] doubleArray0 = new double[1];
      variance0.incrementAll(doubleArray0);
      double double0 = variance0.getResult();
      assertEquals(1L, variance0.getN());
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Variance variance0 = new Variance(false);
      double[] doubleArray0 = new double[2];
      variance0.incrementAll(doubleArray0);
      double double0 = variance0.getResult();
      assertEquals(2L, variance0.getN());
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      SecondMoment secondMoment0 = new SecondMoment();
      Variance variance0 = new Variance(secondMoment0);
      double double0 = variance0.evaluate(doubleArray0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Variance variance0 = new Variance(false);
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = 991.4417343851474;
      double double0 = variance0.evaluate(doubleArray0, doubleArray0);
      assertFalse(variance0.isBiasCorrected());
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Variance variance0 = new Variance();
      // Undeclared exception!
      try { 
        variance0.evaluate();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // input array
         //
         verifyException("org.apache.commons.math.stat.descriptive.moment.Variance", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Variance variance0 = new Variance();
      double[] doubleArray0 = new double[0];
      double double0 = variance0.evaluate(doubleArray0);
      assertTrue(variance0.isBiasCorrected());
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      Variance variance0 = new Variance();
      double double0 = variance0.evaluate(doubleArray0);
      assertEquals(0.0, double0, 0.01);
      assertTrue(variance0.isBiasCorrected());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Variance variance0 = new Variance();
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = 991.2864846;
      double double0 = variance0.evaluate(doubleArray0, doubleArray0);
      assertTrue(variance0.isBiasCorrected());
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Variance variance0 = new Variance();
      double[] doubleArray0 = new double[0];
      double double0 = variance0.evaluate(doubleArray0, 992.5916693234842);
      assertTrue(variance0.isBiasCorrected());
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Variance variance0 = new Variance();
      double[] doubleArray0 = new double[1];
      double double0 = variance0.evaluate(doubleArray0, 0.0);
      assertTrue(variance0.isBiasCorrected());
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Variance variance0 = new Variance(false);
      double[] doubleArray0 = new double[2];
      double double0 = variance0.evaluate(doubleArray0);
      assertFalse(variance0.isBiasCorrected());
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Variance variance0 = new Variance();
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = 3854.70993808299;
      double double0 = variance0.evaluate(doubleArray0, doubleArray0, 3854.70993808299);
      assertEquals(0.0, double0, 0.01);
      assertTrue(variance0.isBiasCorrected());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Variance variance0 = new Variance();
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = 991.2864846;
      double double0 = variance0.evaluate(doubleArray0, doubleArray0);
      assertEquals(0.0, double0, 0.01);
  }
}