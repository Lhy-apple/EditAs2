/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:13:15 GMT 2023
 */

package org.apache.commons.math.special;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.special.Gamma;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Gamma_ESTest extends Gamma_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      double double0 = Gamma.regularizedGammaP((-1919.46790703), (-1919.46790703));
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      double double0 = Gamma.logGamma(Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      double double0 = Gamma.logGamma((-622.61));
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      double double0 = Gamma.regularizedGammaP(Double.NaN, (-1016.4278610839));
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      double double0 = Gamma.regularizedGammaP((-1016.4278610839), Double.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      double double0 = Gamma.regularizedGammaP(1430.242, 0.0, 0.0, 310);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      double double0 = Gamma.regularizedGammaP(1.0E-8, (-973.0), 0.5040833780715299, (-640));
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      try { 
        Gamma.regularizedGammaQ(2241.925, 1.0E-8, (-59.59796035547549), 3752);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Maximal number of iterations (3,752) exceeded
         //
         verifyException("org.apache.commons.math.special.Gamma", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      double double0 = Gamma.regularizedGammaQ(0.5040833780715299, 0.5040833780715299);
      assertEquals(0.31797369434491896, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      try { 
        Gamma.regularizedGammaP((double) 4, 380.89433, 0.0, 4);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Continued fraction convergents failed to converge for value 380.894
         //
         verifyException("org.apache.commons.math.util.ContinuedFraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      double double0 = Gamma.regularizedGammaQ(Double.NaN, Double.NaN, 1083.1684, 4);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      double double0 = Gamma.regularizedGammaQ(0.0, Double.NaN, 371.3375375214683, 0);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      double double0 = Gamma.regularizedGammaQ((-1024.2030940724), 552.5460013587157, (-28.8967856166), 3);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      double double0 = Gamma.regularizedGammaQ(1292.8731843, (-2189.683142364722));
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      double double0 = Gamma.regularizedGammaQ(6.283185307179586, 0.0);
      assertEquals(1.0, double0, 0.01);
  }
}