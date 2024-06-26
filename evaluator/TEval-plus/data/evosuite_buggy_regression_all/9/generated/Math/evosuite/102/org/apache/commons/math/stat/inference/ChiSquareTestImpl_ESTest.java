/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:51:35 GMT 2023
 */

package org.apache.commons.math.stat.inference;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.distribution.ChiSquaredDistributionImpl;
import org.apache.commons.math.distribution.DistributionFactory;
import org.apache.commons.math.stat.inference.ChiSquareTestImpl;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ChiSquareTestImpl_ESTest extends ChiSquareTestImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      DistributionFactory distributionFactory0 = chiSquareTestImpl0.getDistributionFactory();
      assertNotNull(distributionFactory0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      long[][] longArray0 = new long[4][3];
      ChiSquaredDistributionImpl chiSquaredDistributionImpl0 = new ChiSquaredDistributionImpl(0.5);
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl(chiSquaredDistributionImpl0);
      chiSquareTestImpl0.chiSquareTest(longArray0, 8.441822398385275E-5);
      assertEquals(6.0, chiSquaredDistributionImpl0.getDegreesOfFreedom(), 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      long[] longArray0 = new long[2];
      ChiSquaredDistributionImpl chiSquaredDistributionImpl0 = new ChiSquaredDistributionImpl(0.5);
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl(chiSquaredDistributionImpl0);
      try { 
        chiSquareTestImpl0.chiSquareTestDataSetsComparison(longArray0, longArray0, 8.441822398385275E-5);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // observed counts cannot all be 0
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[2];
      double[] doubleArray0 = new double[0];
      try { 
        chiSquareTestImpl0.chiSquare(doubleArray0, longArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // observed, expected array lengths incorrect
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[2];
      double[] doubleArray0 = new double[3];
      try { 
        chiSquareTestImpl0.chiSquare(doubleArray0, longArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // observed, expected array lengths incorrect
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = 1.0E-8;
      doubleArray0[1] = 1.0E-8;
      long[] longArray0 = new long[2];
      longArray0[0] = (-2442L);
      try { 
        chiSquareTestImpl0.chiSquare(doubleArray0, longArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // observed counts must be non-negative and expected counts must be postive
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      double[] doubleArray0 = new double[3];
      long[] longArray0 = new long[3];
      try { 
        chiSquareTestImpl0.chiSquareTest(doubleArray0, longArray0, (-9.837447530487956E-5));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // bad significance level: -9.837447530487956E-5
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      double[] doubleArray0 = new double[3];
      long[] longArray0 = new long[3];
      try { 
        chiSquareTestImpl0.chiSquareTest(doubleArray0, longArray0, 2.1743961811521265E-4);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // observed counts must be non-negative and expected counts must be postive
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      try { 
        chiSquareTestImpl0.chiSquareTest((double[]) null, (long[]) null, 2359.39954);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // bad significance level: 2359.39954
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      double[] doubleArray0 = new double[3];
      doubleArray0[0] = 2.1743961811521265E-4;
      doubleArray0[1] = 2.1743961811521265E-4;
      doubleArray0[2] = 0.5;
      long[] longArray0 = new long[3];
      boolean boolean0 = chiSquareTestImpl0.chiSquareTest(doubleArray0, longArray0, 2.1743961811521265E-4);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      double[] doubleArray0 = new double[3];
      doubleArray0[0] = 2.1743961811521265E-4;
      doubleArray0[1] = 2.1743961811521265E-4;
      doubleArray0[2] = 0.5;
      long[] longArray0 = new long[3];
      longArray0[2] = 835L;
      boolean boolean0 = chiSquareTestImpl0.chiSquareTest(doubleArray0, longArray0, 2.1743961811521265E-4);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      try { 
        chiSquareTestImpl0.chiSquareTest((long[][]) null, (-2078.2409515));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // bad significance level: -2078.2409515
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      long[][] longArray0 = new long[4][3];
      ChiSquaredDistributionImpl chiSquaredDistributionImpl0 = new ChiSquaredDistributionImpl(0.5);
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl(chiSquaredDistributionImpl0);
      try { 
        chiSquareTestImpl0.chiSquareTest(longArray0, 1.0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // bad significance level: 1.0
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      long[] longArray0 = new long[2];
      longArray0[1] = 731L;
      long[] longArray1 = new long[2];
      longArray1[0] = 731L;
      ChiSquaredDistributionImpl chiSquaredDistributionImpl0 = new ChiSquaredDistributionImpl(0.5);
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl(chiSquaredDistributionImpl0);
      long[][] longArray2 = new long[7][2];
      longArray2[0] = longArray0;
      longArray2[1] = longArray0;
      longArray2[2] = longArray0;
      longArray2[3] = longArray1;
      longArray2[4] = longArray1;
      longArray2[5] = longArray1;
      longArray2[6] = longArray0;
      boolean boolean0 = chiSquareTestImpl0.chiSquareTest(longArray2, 8.441822398385275E-5);
      assertEquals(6.0, chiSquaredDistributionImpl0.getDegreesOfFreedom(), 0.01);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[0];
      try { 
        chiSquareTestImpl0.chiSquareDataSetsComparison(longArray0, longArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // oberved1, observed2 array lengths incorrect
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[2];
      long[][] longArray1 = new long[2][3];
      try { 
        chiSquareTestImpl0.chiSquareTestDataSetsComparison(longArray1[0], longArray0, 0.5);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // oberved1, observed2 array lengths incorrect
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[2];
      longArray0[0] = (-750L);
      try { 
        chiSquareTestImpl0.chiSquareDataSetsComparison(longArray0, longArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // observed counts must be non-negative
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[3];
      long[] longArray1 = new long[3];
      longArray1[1] = (-813L);
      try { 
        chiSquareTestImpl0.chiSquareTestDataSetsComparison(longArray0, longArray1, 0.5);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // observed counts must be non-negative
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      long[] longArray0 = new long[2];
      longArray0[0] = 731L;
      ChiSquaredDistributionImpl chiSquaredDistributionImpl0 = new ChiSquaredDistributionImpl(0.5);
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl(chiSquaredDistributionImpl0);
      try { 
        chiSquareTestImpl0.chiSquareTestDataSetsComparison(longArray0, longArray0, 8.441822398385275E-5);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // observed counts must not both be zero
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      long[] longArray0 = new long[2];
      longArray0[0] = 731L;
      longArray0[1] = 731L;
      long[] longArray1 = new long[2];
      longArray1[0] = 731L;
      ChiSquaredDistributionImpl chiSquaredDistributionImpl0 = new ChiSquaredDistributionImpl(0.5);
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl(chiSquaredDistributionImpl0);
      boolean boolean0 = chiSquareTestImpl0.chiSquareTestDataSetsComparison(longArray1, longArray0, 8.441822398385275E-5);
      assertEquals(1.0, chiSquaredDistributionImpl0.getDegreesOfFreedom(), 0.01);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[2];
      try { 
        chiSquareTestImpl0.chiSquareTestDataSetsComparison(longArray0, longArray0, (double) 0L);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // bad significance level: 0.0
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      long[] longArray0 = new long[2];
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      try { 
        chiSquareTestImpl0.chiSquareTestDataSetsComparison(longArray0, longArray0, 1538.6889042446264);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // bad significance level: 1538.6889042446264
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[2];
      longArray0[0] = 717L;
      longArray0[1] = 717L;
      boolean boolean0 = chiSquareTestImpl0.chiSquareTestDataSetsComparison(longArray0, longArray0, 0.5);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[][] longArray0 = new long[0][5];
      try { 
        chiSquareTestImpl0.chiSquareTest(longArray0, 0.5);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Input table must have at least two rows
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[][] longArray0 = new long[7][1];
      try { 
        chiSquareTestImpl0.chiSquareTest(longArray0, 0.5);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Input table must have at least two columns
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[2];
      long[][] longArray1 = new long[2][3];
      longArray1[0] = longArray0;
      try { 
        chiSquareTestImpl0.chiSquareTest(longArray1, 0.5);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Input table must be rectangular
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ChiSquaredDistributionImpl chiSquaredDistributionImpl0 = new ChiSquaredDistributionImpl(380.7581);
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl(chiSquaredDistributionImpl0);
      long[][] longArray0 = new long[3][7];
      long[] longArray1 = new long[2];
      longArray1[0] = (-2653L);
      longArray0[0] = longArray1;
      longArray0[1] = longArray1;
      longArray0[2] = longArray1;
      try { 
        chiSquareTestImpl0.chiSquareTest(longArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // All entries in input 2-way table must be non-negative
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }
}
