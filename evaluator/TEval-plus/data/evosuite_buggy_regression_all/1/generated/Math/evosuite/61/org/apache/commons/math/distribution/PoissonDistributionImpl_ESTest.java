/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:05:26 GMT 2023
 */

package org.apache.commons.math.distribution;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.distribution.PoissonDistributionImpl;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PoissonDistributionImpl_ESTest extends PoissonDistributionImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      PoissonDistributionImpl poissonDistributionImpl0 = new PoissonDistributionImpl(1.0, 602.17645966956, 796);
      poissonDistributionImpl0.getMean();
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      PoissonDistributionImpl poissonDistributionImpl0 = new PoissonDistributionImpl(1915.572273666193, 1.2958646899018938E-9);
      int int0 = poissonDistributionImpl0.sample();
      assertEquals(1924, int0);
      assertEquals(1915.572273666193, poissonDistributionImpl0.getMean(), 0.01);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      PoissonDistributionImpl poissonDistributionImpl0 = null;
      try {
        poissonDistributionImpl0 = new PoissonDistributionImpl(0.0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // the Poisson mean must be positive (0)
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      PoissonDistributionImpl poissonDistributionImpl0 = new PoissonDistributionImpl(1.0, 1023);
      double double0 = poissonDistributionImpl0.normalApproximateProbability(894);
      assertEquals(1.0, poissonDistributionImpl0.getMean(), 0.01);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      PoissonDistributionImpl poissonDistributionImpl0 = new PoissonDistributionImpl(1915.572273666193, 1.2958646899018938E-9);
      double double0 = poissonDistributionImpl0.probability((-1866));
      assertEquals(0.0, double0, 0.01);
      assertEquals(1915.572273666193, poissonDistributionImpl0.getMean(), 0.01);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      PoissonDistributionImpl poissonDistributionImpl0 = new PoissonDistributionImpl(1915.572273666193, 1.2958646899018938E-9);
      double double0 = poissonDistributionImpl0.probability(0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      PoissonDistributionImpl poissonDistributionImpl0 = new PoissonDistributionImpl(1915.572273666193, 1.2958646899018938E-9);
      double double0 = poissonDistributionImpl0.probability(Integer.MAX_VALUE);
      assertEquals(1915.572273666193, poissonDistributionImpl0.getMean(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      PoissonDistributionImpl poissonDistributionImpl0 = new PoissonDistributionImpl(1915.572273666193, 1.2958646899018938E-9);
      double double0 = poissonDistributionImpl0.probability(2147352575);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1915.572273666193, poissonDistributionImpl0.getMean(), 0.01);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      PoissonDistributionImpl poissonDistributionImpl0 = new PoissonDistributionImpl(1915.572273666193, 1.2958646899018938E-9);
      double double0 = poissonDistributionImpl0.cumulativeProbability((double) (-2400));
      assertEquals(0.0, double0, 0.01);
      assertEquals(1915.572273666193, poissonDistributionImpl0.getMean(), 0.01);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      PoissonDistributionImpl poissonDistributionImpl0 = new PoissonDistributionImpl(1.0, 1023);
      int int0 = poissonDistributionImpl0.inverseCumulativeProbability(1.0);
      assertEquals(Integer.MAX_VALUE, int0);
      assertEquals(1.0, poissonDistributionImpl0.getMean(), 0.01);
  }
}
