/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:47:44 GMT 2023
 */

package org.apache.commons.math3.distribution;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.distribution.FDistribution;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class FDistribution_ESTest extends FDistribution_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      FDistribution fDistribution0 = new FDistribution(497.586570762909, 497.586570762909);
      assertEquals(Double.POSITIVE_INFINITY, fDistribution0.getSupportUpperBound(), 0.01);
      
      double double0 = fDistribution0.sample();
      assertEquals(1.0040356218630404, fDistribution0.getNumericalMean(), 0.01);
      assertEquals(1.1233900324381083, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      FDistribution fDistribution0 = new FDistribution(0.5, 0.5);
      boolean boolean0 = fDistribution0.isSupportUpperBoundInclusive();
      assertFalse(boolean0);
      assertEquals(0.5, fDistribution0.getNumeratorDegreesOfFreedom(), 0.01);
      assertEquals(0.5, fDistribution0.getDenominatorDegreesOfFreedom(), 0.01);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      FDistribution fDistribution0 = new FDistribution(1.1379799629071911E-50, 1.1379799629071911E-50, 1.1379799629071911E-50);
      boolean boolean0 = fDistribution0.isSupportLowerBoundInclusive();
      assertTrue(boolean0);
      assertEquals(1.1379799629071911E-50, fDistribution0.getDenominatorDegreesOfFreedom(), 0.01);
      assertEquals(1.1379799629071911E-50, fDistribution0.getNumeratorDegreesOfFreedom(), 0.01);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      FDistribution fDistribution0 = new FDistribution(0.5, 0.5);
      double double0 = fDistribution0.density(2.300227165222168);
      assertEquals(0.0397386237959564, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      FDistribution fDistribution0 = null;
      try {
        fDistribution0 = new FDistribution(0.0, 0.0, 0.32058215141296387);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // degrees of freedom (0)
         //
         verifyException("org.apache.commons.math3.distribution.FDistribution", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      FDistribution fDistribution0 = null;
      try {
        fDistribution0 = new FDistribution(3502.3403212034978, (-1.0), 3502.3403212034978);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // degrees of freedom (-1)
         //
         verifyException("org.apache.commons.math3.distribution.FDistribution", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      FDistribution fDistribution0 = new FDistribution(0.5, 0.5);
      double double0 = fDistribution0.getNumericalVariance();
      assertEquals(Double.POSITIVE_INFINITY, fDistribution0.getSupportUpperBound(), 0.01);
      assertEquals(Double.NaN, double0, 0.01);
      
      double double1 = fDistribution0.sample();
      assertEquals(Double.NaN, fDistribution0.getNumericalMean(), 0.01);
      assertTrue(fDistribution0.isSupportConnected());
      assertEquals(941.0262453603775, double1, 0.01);
      assertEquals(0.5, fDistribution0.getDenominatorDegreesOfFreedom(), 0.01);
      assertEquals(0.5, fDistribution0.getNumeratorDegreesOfFreedom(), 0.01);
  }
}