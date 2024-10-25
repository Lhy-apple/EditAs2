/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 18:06:51 GMT 2023
 */

package org.apache.commons.math3.analysis.differentiation;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.analysis.differentiation.DSCompiler;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DSCompiler_ESTest extends DSCompiler_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[4];
      dSCompiler0.pow(doubleArray0, 0, doubleArray0, 0, doubleArray0, 0);
      assertArrayEquals(new double[] {Double.NaN, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      int[] intArray0 = new int[1];
      try { 
        dSCompiler0.getPartialDerivativeIndex(intArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 1 != 0
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(2, 2);
      // Undeclared exception!
      try { 
        dSCompiler0.expm1((double[]) null, (-509), (double[]) null, (-509));
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      int int0 = dSCompiler0.getOrder();
      assertEquals(0, int0);
      assertEquals(0, dSCompiler0.getFreeParameters());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      // Undeclared exception!
      DSCompiler.getCompiler(558, 558);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 2);
      double[] doubleArray0 = new double[4];
      dSCompiler0.atanh(doubleArray0, 2, doubleArray0, 0);
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
      assertEquals(2, dSCompiler0.getOrder());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      // Undeclared exception!
      DSCompiler.getCompiler(1155, 1);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      assertEquals(0, dSCompiler0.getFreeParameters());
      
      int[] intArray0 = new int[0];
      int int0 = dSCompiler0.getPartialDerivativeIndex(intArray0);
      assertEquals(0, int0);
      assertEquals(0, dSCompiler0.getOrder());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(1458, 0);
      double[] doubleArray0 = new double[9];
      dSCompiler0.linearCombination((-1.1109834472051523E-8), doubleArray0, 3, (-737.35), doubleArray0, 0, doubleArray0, 3);
      assertEquals(1458, dSCompiler0.getFreeParameters());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[6];
      dSCompiler0.linearCombination((double) 0, doubleArray0, 0, (-1275.444), doubleArray0, 0, 1994.54639476, doubleArray0, 0, doubleArray0, 0);
      assertEquals(0, dSCompiler0.getOrder());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(2, 0);
      assertEquals(1, dSCompiler0.getSize());
      
      double[] doubleArray0 = new double[5];
      dSCompiler0.linearCombination(0.0, doubleArray0, 0, (double) 2, doubleArray0, 2, (-773.94121252), doubleArray0, 2, (-1876.83614355373), doubleArray0, 4, doubleArray0, 0);
      assertEquals(2, dSCompiler0.getFreeParameters());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(2, 0);
      double[] doubleArray0 = new double[4];
      dSCompiler0.remainder(doubleArray0, 0, doubleArray0, 2, doubleArray0, 0);
      assertEquals(0, dSCompiler0.getOrder());
      assertArrayEquals(new double[] {Double.NaN, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
      assertEquals(2, dSCompiler0.getFreeParameters());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(2, 2);
      double[] doubleArray0 = new double[5];
      // Undeclared exception!
      try { 
        dSCompiler0.remainder(doubleArray0, 2, doubleArray0, 2, doubleArray0, 2);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 5
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(1, 1);
      double[] doubleArray0 = new double[6];
      // Undeclared exception!
      try { 
        dSCompiler0.pow(doubleArray0, 1, 1.0, doubleArray0, 122);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 122
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[1];
      // Undeclared exception!
      try { 
        dSCompiler0.pow(doubleArray0, 7, 0, doubleArray0, (-477));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -477
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 929);
      double[] doubleArray0 = new double[14];
      // Undeclared exception!
      try { 
        dSCompiler0.pow(doubleArray0, 0, 2971, doubleArray0, 929);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 929
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(2, 0);
      double[] doubleArray0 = new double[5];
      dSCompiler0.rootN(doubleArray0, 0, 0, doubleArray0, 0);
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[4];
      // Undeclared exception!
      try { 
        dSCompiler0.rootN(doubleArray0, (-118), 3, doubleArray0, 4);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -118
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(2, 2);
      dSCompiler0.atan2(doubleArray0, 0, doubleArray0, 2, doubleArray0, 2);
      assertArrayEquals(new double[] {0.0, 0.0, Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(2, 2);
      double[] doubleArray0 = new double[5];
      // Undeclared exception!
      try { 
        dSCompiler0.pow(doubleArray0, 2, doubleArray0, 2, doubleArray0, (-1234));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 5
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[1];
      // Undeclared exception!
      try { 
        dSCompiler0.log1p(doubleArray0, 0, doubleArray0, 193);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 193
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 929);
      assertNotNull(dSCompiler0);
      
      double[] doubleArray0 = new double[3];
      dSCompiler0.log1p(doubleArray0, 2, doubleArray0, 2);
      assertEquals(929, dSCompiler0.getOrder());
      assertArrayEquals(new double[] {0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[4];
      dSCompiler0.log10(doubleArray0, 0, doubleArray0, 0);
      assertArrayEquals(new double[] {Double.NEGATIVE_INFINITY, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
      assertEquals(0, dSCompiler0.getOrder());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(15, 2);
      // Undeclared exception!
      try { 
        dSCompiler0.log10(doubleArray0, 2, doubleArray0, 777);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 777
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(1458, 0);
      double[] doubleArray0 = new double[3];
      // Undeclared exception!
      try { 
        dSCompiler0.cos(doubleArray0, 0, doubleArray0, 1458);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 1458
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 76);
      double[] doubleArray0 = new double[7];
      dSCompiler0.cos(doubleArray0, 0, doubleArray0, 0);
      assertEquals(76, dSCompiler0.getOrder());
      assertArrayEquals(new double[] {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(2, 0);
      double[] doubleArray0 = new double[5];
      dSCompiler0.sin(doubleArray0, 2, doubleArray0, 2);
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 1327);
      double[] doubleArray0 = new double[8];
      dSCompiler0.sin(doubleArray0, 0, doubleArray0, 0);
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[4];
      dSCompiler0.tan(doubleArray0, 0, doubleArray0, 0);
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
      assertEquals(0, dSCompiler0.getOrder());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(2, 2);
      double[] doubleArray0 = new double[9];
      // Undeclared exception!
      try { 
        dSCompiler0.tan(doubleArray0, 2, doubleArray0, 1533);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 1533
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[1];
      dSCompiler0.acos(doubleArray0, 0, doubleArray0, 0);
      assertEquals(0, dSCompiler0.getOrder());
      assertArrayEquals(new double[] {1.5707963267948966}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(2, 2);
      double[] doubleArray0 = new double[3];
      // Undeclared exception!
      try { 
        dSCompiler0.acos(doubleArray0, 2, doubleArray0, 2);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 3
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 929);
      double[] doubleArray0 = new double[5];
      // Undeclared exception!
      dSCompiler0.acos(doubleArray0, 0, doubleArray0, 929);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      dSCompiler0.asin(doubleArray0, 0, doubleArray0, 0);
      assertArrayEquals(new double[] {0.0}, doubleArray0, 0.01);
      assertEquals(0, dSCompiler0.getOrder());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 7);
      dSCompiler0.asin(doubleArray0, 0, doubleArray0, 0);
      assertEquals(7, dSCompiler0.getOrder());
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 1327);
      double[] doubleArray0 = new double[1];
      // Undeclared exception!
      dSCompiler0.atan(doubleArray0, 0, doubleArray0, 0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(2, 0);
      double[] doubleArray0 = new double[9];
      doubleArray0[0] = 37.717349303088895;
      doubleArray0[2] = (-3.951314467739045E140);
      // Undeclared exception!
      try { 
        dSCompiler0.atan2(doubleArray0, 0, doubleArray0, 2, doubleArray0, 1193);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 1193
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      doubleArray0[2] = (-3.951314467739045E140);
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(2, 2);
      assertEquals(6, dSCompiler0.getSize());
      
      dSCompiler0.atan2(doubleArray0, 0, doubleArray0, 2, doubleArray0, 2);
      assertArrayEquals(new double[] {0.0, 0.0, (-3.141592653589793), -0.0, 1.0, -0.0, -0.0, -0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(2, 0);
      double[] doubleArray0 = new double[5];
      dSCompiler0.cosh(doubleArray0, 0, doubleArray0, 0);
      assertArrayEquals(new double[] {1.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(2, 2);
      double[] doubleArray0 = new double[5];
      // Undeclared exception!
      try { 
        dSCompiler0.cosh(doubleArray0, 2, doubleArray0, 2);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 5
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[3];
      dSCompiler0.sinh(doubleArray0, 0, doubleArray0, 0);
      assertEquals(0, dSCompiler0.getOrder());
      assertArrayEquals(new double[] {0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 1327);
      double[] doubleArray0 = new double[8];
      // Undeclared exception!
      try { 
        dSCompiler0.sinh(doubleArray0, 0, doubleArray0, 1015);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 1015
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(3, 0);
      assertEquals(1, dSCompiler0.getSize());
      
      double[] doubleArray0 = new double[5];
      dSCompiler0.tanh(doubleArray0, 3, doubleArray0, 3);
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 2);
      dSCompiler0.tanh(doubleArray0, 3, doubleArray0, 3);
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[9];
      dSCompiler0.acosh(doubleArray0, 0, doubleArray0, 2);
      assertArrayEquals(new double[] {0.0, 0.0, Double.NaN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(2, 2);
      double[] doubleArray0 = new double[8];
      dSCompiler0.acosh(doubleArray0, 2, doubleArray0, 2);
      assertArrayEquals(new double[] {0.0, 0.0, Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 929);
      double[] doubleArray0 = new double[3];
      // Undeclared exception!
      dSCompiler0.acosh(doubleArray0, 0, doubleArray0, 0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(1284, 0);
      double[] doubleArray0 = new double[9];
      dSCompiler0.asinh(doubleArray0, 3, doubleArray0, 3);
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 34);
      double[] doubleArray0 = new double[8];
      dSCompiler0.asinh(doubleArray0, 3, doubleArray0, 0);
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(2, 0);
      double[] doubleArray0 = new double[5];
      dSCompiler0.atanh(doubleArray0, 4, doubleArray0, 0);
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 929);
      double[] doubleArray0 = new double[3];
      // Undeclared exception!
      dSCompiler0.atanh(doubleArray0, 0, doubleArray0, 620);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(2, 2);
      double[] doubleArray0 = new double[9];
      dSCompiler0.taylor(doubleArray0, 2, doubleArray0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(2, 2);
      dSCompiler0.checkCompatibility(dSCompiler0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(2, 0);
      DSCompiler dSCompiler1 = DSCompiler.getCompiler(0, 0);
      try { 
        dSCompiler0.checkCompatibility(dSCompiler1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 2 != 0
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 929);
      DSCompiler dSCompiler1 = DSCompiler.getCompiler(0, 7);
      try { 
        dSCompiler1.checkCompatibility(dSCompiler0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 7 != 929
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }
}
