/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:01:43 GMT 2023
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
      FDistribution fDistribution0 = new FDistribution(757.884732017, 757.884732017);
      assertEquals(Double.POSITIVE_INFINITY, fDistribution0.getSupportUpperBound(), 0.01);
      assertEquals(757.884732017, fDistribution0.getNumeratorDegreesOfFreedom(), 0.01);
      
      double double0 = fDistribution0.sample();
      assertEquals(1.0988354378756104, double0, 0.01);
      
      double double1 = fDistribution0.getNumericalVariance();
      assertEquals(1.0026459060691215, fDistribution0.getNumericalMean(), 0.01);
      assertEquals(0.0053269276687805695, double1, 0.01);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      FDistribution fDistribution0 = new FDistribution(757.884732017, 757.884732017);
      boolean boolean0 = fDistribution0.isSupportUpperBoundInclusive();
      assertEquals(757.884732017, fDistribution0.getNumeratorDegreesOfFreedom(), 0.01);
      assertEquals(757.884732017, fDistribution0.getDenominatorDegreesOfFreedom(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      FDistribution fDistribution0 = new FDistribution(757.884732017, 757.884732017);
      boolean boolean0 = fDistribution0.isSupportLowerBoundInclusive();
      assertEquals(757.884732017, fDistribution0.getNumeratorDegreesOfFreedom(), 0.01);
      assertTrue(boolean0);
      assertEquals(1.0026459060691215, fDistribution0.getNumericalMean(), 0.01);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      FDistribution fDistribution0 = new FDistribution(757.884732017, 757.884732017);
      fDistribution0.density(Double.NaN);
      assertEquals(757.884732017, fDistribution0.getNumeratorDegreesOfFreedom(), 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      FDistribution fDistribution0 = null;
      try {
        fDistribution0 = new FDistribution((-1.0), (-1.0));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // degrees of freedom (-1)
         //
         verifyException("org.apache.commons.math3.distribution.FDistribution", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      FDistribution fDistribution0 = null;
      try {
        fDistribution0 = new FDistribution(0.5, 0.0, (-2001.1825231807727));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // degrees of freedom (0)
         //
         verifyException("org.apache.commons.math3.distribution.FDistribution", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      FDistribution fDistribution0 = new FDistribution(0.05694137513637543, 0.05694137513637543, 0.05694137513637543);
      assertEquals(Double.POSITIVE_INFINITY, fDistribution0.getSupportUpperBound(), 0.01);
      assertTrue(fDistribution0.isSupportConnected());
      
      double double0 = fDistribution0.sample();
      assertEquals(0.05694137513637543, fDistribution0.getDenominatorDegreesOfFreedom(), 0.01);
      assertEquals(Double.NaN, fDistribution0.getNumericalMean(), 0.01);
      assertEquals(9.886483283430228E15, double0, 0.01);
  }
}