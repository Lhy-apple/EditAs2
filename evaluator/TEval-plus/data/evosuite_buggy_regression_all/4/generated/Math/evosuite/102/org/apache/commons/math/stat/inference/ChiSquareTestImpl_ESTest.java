/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:14:19 GMT 2023
 */

package org.apache.commons.math.stat.inference;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
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
      long[] longArray0 = new long[2];
      double[] doubleArray0 = new double[2];
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
  public void test01()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[][] longArray0 = new long[5][0];
      try { 
        chiSquareTestImpl0.chiSquareTest(longArray0, 4.9E-324);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Input table must have at least two columns
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[5];
      try { 
        chiSquareTestImpl0.chiSquareTestDataSetsComparison(longArray0, longArray0, 1.0E-8);
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
      DistributionFactory distributionFactory0 = chiSquareTestImpl0.getDistributionFactory();
      assertNotNull(distributionFactory0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      double[] doubleArray0 = new double[1];
      long[] longArray0 = new long[0];
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
      long[] longArray0 = new long[6];
      double[] doubleArray0 = new double[2];
      try { 
        chiSquareTestImpl0.chiSquareTest(doubleArray0, longArray0, 0.5);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // observed, expected array lengths incorrect
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      double[] doubleArray0 = new double[3];
      doubleArray0[0] = 3.6899182659531625E-6;
      doubleArray0[1] = 3.6899182659531625E-6;
      doubleArray0[2] = 3.6899182659531625E-6;
      long[] longArray0 = new long[3];
      longArray0[1] = (-266L);
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
  public void test07()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[2];
      double[] doubleArray0 = new double[0];
      try { 
        chiSquareTestImpl0.chiSquareTest(doubleArray0, longArray0, (-4597.667032));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // bad significance level: -4597.667032
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[6];
      double[] doubleArray0 = new double[2];
      try { 
        chiSquareTestImpl0.chiSquareTest(doubleArray0, longArray0, 2273.006840229461);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // bad significance level: 2273.006840229461
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[2];
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = 440.48258373;
      doubleArray0[1] = Double.NaN;
      boolean boolean0 = chiSquareTestImpl0.chiSquareTest(doubleArray0, longArray0, Double.NaN);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[2];
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = 440.48258373;
      doubleArray0[1] = 440.48258373;
      boolean boolean0 = chiSquareTestImpl0.chiSquareTest(doubleArray0, longArray0, 2.1743961811521265E-4);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[9];
      long[][] longArray1 = new long[5][0];
      longArray1[0] = longArray0;
      longArray1[1] = longArray0;
      longArray1[2] = longArray0;
      longArray1[3] = longArray0;
      longArray1[4] = longArray0;
      boolean boolean0 = chiSquareTestImpl0.chiSquareTest(longArray1, 4.9E-324);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      try { 
        chiSquareTestImpl0.chiSquareTest((long[][]) null, (-3215.35539411521));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // bad significance level: -3215.35539411521
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[][] longArray0 = new long[6][1];
      try { 
        chiSquareTestImpl0.chiSquareTest(longArray0, (double) 1040L);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // bad significance level: 1040.0
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[9];
      longArray0[0] = 1040L;
      longArray0[1] = 1040L;
      longArray0[2] = 1040L;
      longArray0[3] = 1040L;
      longArray0[4] = 1040L;
      long[][] longArray1 = new long[5][0];
      long[] longArray2 = new long[9];
      longArray2[5] = 1040L;
      longArray2[6] = 1040L;
      longArray2[7] = 27040000L;
      longArray2[8] = 1040L;
      longArray1[0] = longArray2;
      longArray1[1] = longArray0;
      longArray1[2] = longArray0;
      longArray1[3] = longArray0;
      longArray1[4] = longArray0;
      boolean boolean0 = chiSquareTestImpl0.chiSquareTest(longArray1, 4.9E-324);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[1];
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
  public void test16()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[2];
      long[][] longArray1 = new long[2][6];
      try { 
        chiSquareTestImpl0.chiSquareDataSetsComparison(longArray0, longArray1[0]);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // oberved1, observed2 array lengths incorrect
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[9];
      longArray0[0] = (-1L);
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
  public void test18()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[2];
      long[] longArray1 = new long[2];
      longArray1[1] = (-927L);
      try { 
        chiSquareTestImpl0.chiSquareTestDataSetsComparison(longArray0, longArray1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // observed counts must be non-negative
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[5];
      longArray0[1] = 6240L;
      long[] longArray1 = new long[5];
      longArray1[0] = 1040L;
      try { 
        chiSquareTestImpl0.chiSquareTestDataSetsComparison(longArray0, longArray1, 1.0E-8);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // observed counts must not both be zero
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[5];
      longArray0[0] = 1040L;
      longArray0[1] = 6240L;
      longArray0[2] = 1040L;
      longArray0[3] = 1040L;
      longArray0[4] = 1040L;
      boolean boolean0 = chiSquareTestImpl0.chiSquareTestDataSetsComparison(longArray0, longArray0, 1.0E-8);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[0];
      try { 
        chiSquareTestImpl0.chiSquareTestDataSetsComparison(longArray0, longArray0, (-193.0));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // bad significance level: -193.0
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      try { 
        chiSquareTestImpl0.chiSquareTestDataSetsComparison((long[]) null, (long[]) null, (double) 229L);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // bad significance level: 229.0
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[5];
      longArray0[1] = 6240L;
      longArray0[2] = 1040L;
      longArray0[3] = 1040L;
      longArray0[4] = 1040L;
      long[] longArray1 = new long[5];
      longArray1[0] = 1040L;
      boolean boolean0 = chiSquareTestImpl0.chiSquareTestDataSetsComparison(longArray0, longArray1, 1.0E-8);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[][] longArray0 = new long[0][1];
      try { 
        chiSquareTestImpl0.chiSquareTest(longArray0, 1.0E-6);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Input table must have at least two rows
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[2];
      long[][] longArray1 = new long[2][6];
      longArray1[1] = longArray0;
      try { 
        chiSquareTestImpl0.chiSquare(longArray1);
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
      ChiSquareTestImpl chiSquareTestImpl0 = new ChiSquareTestImpl();
      long[] longArray0 = new long[4];
      longArray0[0] = (-265L);
      long[][] longArray1 = new long[5][1];
      longArray1[0] = longArray0;
      longArray1[1] = longArray0;
      longArray1[2] = longArray0;
      longArray1[3] = longArray0;
      longArray1[4] = longArray0;
      try { 
        chiSquareTestImpl0.chiSquare(longArray1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // All entries in input 2-way table must be non-negative
         //
         verifyException("org.apache.commons.math.stat.inference.ChiSquareTestImpl", e);
      }
  }
}