/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:43:47 GMT 2023
 */

package org.apache.commons.math.distribution;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.distribution.NormalDistributionImpl;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NormalDistributionImpl_ESTest extends NormalDistributionImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      NormalDistributionImpl normalDistributionImpl0 = new NormalDistributionImpl();
      double double0 = normalDistributionImpl0.getMean();
      assertEquals(1.0, normalDistributionImpl0.getStandardDeviation(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      NormalDistributionImpl normalDistributionImpl0 = new NormalDistributionImpl();
      double double0 = normalDistributionImpl0.density(1934.8738101008794);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, normalDistributionImpl0.getStandardDeviation(), 0.01);
      assertEquals(0.0, normalDistributionImpl0.getMean(), 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      NormalDistributionImpl normalDistributionImpl0 = new NormalDistributionImpl();
      double double0 = normalDistributionImpl0.sample();
      assertEquals(1.0, normalDistributionImpl0.getStandardDeviation(), 0.01);
      assertEquals(0.0, double0, 0.01);
      assertEquals(0.0, normalDistributionImpl0.getMean(), 0.01);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      NormalDistributionImpl normalDistributionImpl0 = new NormalDistributionImpl();
      double double0 = normalDistributionImpl0.getStandardDeviation();
      assertEquals(1.0, double0, 0.01);
      assertEquals(0.0, normalDistributionImpl0.getMean(), 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      NormalDistributionImpl normalDistributionImpl0 = new NormalDistributionImpl();
      double double0 = normalDistributionImpl0.inverseCumulativeProbability(1.0E-9);
      assertEquals((-5.997807014826545), double0, 0.01);
      assertEquals(0.0, normalDistributionImpl0.getMean(), 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      NormalDistributionImpl normalDistributionImpl0 = null;
      try {
        normalDistributionImpl0 = new NormalDistributionImpl((-7.800414592973399E-9), (-7.800414592973399E-9), (-7.800414592973399E-9));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -0 is smaller than, or equal to, the minimum (0): standard deviation (-0)
         //
         verifyException("org.apache.commons.math.distribution.NormalDistributionImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      NormalDistributionImpl normalDistributionImpl0 = new NormalDistributionImpl();
      double double0 = normalDistributionImpl0.inverseCumulativeProbability(0.0);
      assertEquals(Double.NEGATIVE_INFINITY, double0, 0.01);
      assertEquals(0.0, normalDistributionImpl0.getMean(), 0.01);
      assertEquals(1.0, normalDistributionImpl0.getStandardDeviation(), 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      NormalDistributionImpl normalDistributionImpl0 = new NormalDistributionImpl();
      double double0 = normalDistributionImpl0.inverseCumulativeProbability(1.0);
      assertEquals(1.0, normalDistributionImpl0.getStandardDeviation(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
      assertEquals(0.0, normalDistributionImpl0.getMean(), 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      NormalDistributionImpl normalDistributionImpl0 = new NormalDistributionImpl();
      double double0 = normalDistributionImpl0.getDomainLowerBound(531.589541309043);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, normalDistributionImpl0.getStandardDeviation(), 0.01);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      NormalDistributionImpl normalDistributionImpl0 = new NormalDistributionImpl();
      double double0 = normalDistributionImpl0.getDomainUpperBound(274.91213);
      assertEquals(1.7976931348623157E308, double0, 0.01);
      assertEquals(0.0, normalDistributionImpl0.getMean(), 0.01);
      assertEquals(1.0, normalDistributionImpl0.getStandardDeviation(), 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      NormalDistributionImpl normalDistributionImpl0 = new NormalDistributionImpl();
      double double0 = normalDistributionImpl0.getInitialDomain(0.5);
      assertEquals(1.0, normalDistributionImpl0.getStandardDeviation(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      NormalDistributionImpl normalDistributionImpl0 = new NormalDistributionImpl(1934.8738101, 1934.8738101);
      double double0 = normalDistributionImpl0.getInitialDomain(1934.8738101);
      assertEquals(3869.7476202, double0, 0.01);
  }
}
