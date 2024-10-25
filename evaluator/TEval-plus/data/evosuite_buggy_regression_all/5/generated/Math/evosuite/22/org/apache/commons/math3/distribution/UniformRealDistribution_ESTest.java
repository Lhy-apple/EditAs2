/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:27:52 GMT 2023
 */

package org.apache.commons.math3.distribution;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class UniformRealDistribution_ESTest extends UniformRealDistribution_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      UniformRealDistribution uniformRealDistribution0 = new UniformRealDistribution();
      double double0 = uniformRealDistribution0.sample();
      assertEquals(0.9026297667469598, double0, 0.01);
      assertEquals(0.0, uniformRealDistribution0.getSupportLowerBound(), 0.01);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      UniformRealDistribution uniformRealDistribution0 = new UniformRealDistribution();
      double double0 = uniformRealDistribution0.inverseCumulativeProbability(0.9026297667469598);
      assertEquals(0.08333333333333333, uniformRealDistribution0.getNumericalVariance(), 0.01);
      assertEquals(0.9026297667469598, double0, 0.01);
      assertEquals(0.0, uniformRealDistribution0.getSupportLowerBound(), 0.01);
      assertEquals(1.0, uniformRealDistribution0.getSupportUpperBound(), 0.01);
      assertEquals(0.5, uniformRealDistribution0.getNumericalMean(), 0.01);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      UniformRealDistribution uniformRealDistribution0 = new UniformRealDistribution();
      boolean boolean0 = uniformRealDistribution0.isSupportUpperBoundInclusive();
      assertFalse(boolean0);
      assertEquals(0.0, uniformRealDistribution0.getSupportLowerBound(), 0.01);
      assertEquals(0.08333333333333333, uniformRealDistribution0.getNumericalVariance(), 0.01);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      UniformRealDistribution uniformRealDistribution0 = new UniformRealDistribution();
      boolean boolean0 = uniformRealDistribution0.isSupportLowerBoundInclusive();
      assertEquals(0.0, uniformRealDistribution0.getSupportLowerBound(), 0.01);
      assertTrue(boolean0);
      assertEquals(0.5, uniformRealDistribution0.getNumericalMean(), 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      UniformRealDistribution uniformRealDistribution0 = null;
      try {
        uniformRealDistribution0 = new UniformRealDistribution(0.0, 0.0, 0.0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // lower bound (0) must be strictly less than upper bound (0)
         //
         verifyException("org.apache.commons.math3.distribution.UniformRealDistribution", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      UniformRealDistribution uniformRealDistribution0 = new UniformRealDistribution();
      double double0 = uniformRealDistribution0.density((-2.0));
      assertEquals(0.0, double0, 0.01);
      assertEquals(0.08333333333333333, uniformRealDistribution0.getNumericalVariance(), 0.01);
      assertEquals(0.0, uniformRealDistribution0.getSupportLowerBound(), 0.01);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      UniformRealDistribution uniformRealDistribution0 = new UniformRealDistribution();
      double double0 = uniformRealDistribution0.density(0.0);
      assertEquals(0.0, uniformRealDistribution0.getSupportLowerBound(), 0.01);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      UniformRealDistribution uniformRealDistribution0 = new UniformRealDistribution();
      double double0 = uniformRealDistribution0.density(1600.50964804964);
      assertEquals(0.0, double0, 0.01);
      assertEquals(0.5, uniformRealDistribution0.getNumericalMean(), 0.01);
      assertEquals(0.0, uniformRealDistribution0.getSupportLowerBound(), 0.01);
  }
}
