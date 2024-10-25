/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:13:34 GMT 2023
 */

package org.apache.commons.math.distribution;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.distribution.FDistributionImpl;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class FDistributionImpl_ESTest extends FDistributionImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      FDistributionImpl fDistributionImpl0 = new FDistributionImpl(1.0, 2.1743961811521265E-4);
      fDistributionImpl0.setDenominatorDegreesOfFreedom(1039.82886834);
      double double0 = fDistributionImpl0.inverseCumulativeProbability(2.1743961811521265E-4);
      assertEquals(1039.82886834, fDistributionImpl0.getDenominatorDegreesOfFreedom(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      FDistributionImpl fDistributionImpl0 = new FDistributionImpl(1.0, 2.1743961811521265E-4);
      double double0 = fDistributionImpl0.inverseCumulativeProbability(0.0);
      assertEquals(2.1743961811521265E-4, fDistributionImpl0.getDenominatorDegreesOfFreedom(), 0.01);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, fDistributionImpl0.getNumeratorDegreesOfFreedom(), 0.01);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      FDistributionImpl fDistributionImpl0 = new FDistributionImpl(1.0, 2.1743961811521265E-4);
      double double0 = fDistributionImpl0.inverseCumulativeProbability(1.0);
      assertEquals(2.1743961811521265E-4, fDistributionImpl0.getDenominatorDegreesOfFreedom(), 0.01);
      assertEquals(1.0, fDistributionImpl0.getNumeratorDegreesOfFreedom(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      FDistributionImpl fDistributionImpl0 = null;
      try {
        fDistributionImpl0 = new FDistributionImpl((-2673.18), (-2673.18));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // degrees of freedom must be positive.
         //
         verifyException("org.apache.commons.math.distribution.FDistributionImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      FDistributionImpl fDistributionImpl0 = null;
      try {
        fDistributionImpl0 = new FDistributionImpl(2023.470039724713, (-948.191));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // degrees of freedom must be positive.
         //
         verifyException("org.apache.commons.math.distribution.FDistributionImpl", e);
      }
  }
}
