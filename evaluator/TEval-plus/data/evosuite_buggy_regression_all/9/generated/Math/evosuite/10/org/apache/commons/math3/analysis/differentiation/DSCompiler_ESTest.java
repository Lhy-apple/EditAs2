/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:36:57 GMT 2023
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
      double[] doubleArray0 = new double[20];
      dSCompiler0.atan2(doubleArray0, 0, doubleArray0, 0, doubleArray0, 0);
      assertEquals(20, doubleArray0.length);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[6];
      dSCompiler0.pow(doubleArray0, 0, doubleArray0, 0, doubleArray0, 0);
      assertEquals(0, dSCompiler0.getOrder());
      assertArrayEquals(new double[] {Double.NaN, 0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
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
  public void test03()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[6];
      dSCompiler0.expm1(doubleArray0, 3, doubleArray0, 3);
      assertEquals(0, dSCompiler0.getFreeParameters());
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
      assertEquals(0, dSCompiler0.getOrder());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      int int0 = dSCompiler0.getOrder();
      assertEquals(0, dSCompiler0.getFreeParameters());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      // Undeclared exception!
      DSCompiler.getCompiler(1, 4904);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      // Undeclared exception!
      DSCompiler.getCompiler(1259, 1259);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 1);
      assertNotNull(dSCompiler0);
      assertEquals(0, dSCompiler0.getFreeParameters());
      
      int[] intArray0 = new int[0];
      int int0 = dSCompiler0.getPartialDerivativeIndex(intArray0);
      assertEquals(1, dSCompiler0.getOrder());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 1);
      double[] doubleArray0 = new double[4];
      dSCompiler0.linearCombination(3097.21, doubleArray0, 1, (double) 1, doubleArray0, 1, doubleArray0, 0);
      assertEquals(1, dSCompiler0.getOrder());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[20];
      dSCompiler0.linearCombination((double) 0, doubleArray0, 0, (double) 0, doubleArray0, 0, 1050.688525, doubleArray0, 0, doubleArray0, 0);
      assertEquals(20, doubleArray0.length);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[4];
      dSCompiler0.linearCombination(1.0, doubleArray0, 0, 0.0, doubleArray0, 0, 1.0, doubleArray0, 0, (double) 0, doubleArray0, 0, doubleArray0, 0);
      assertEquals(1, dSCompiler0.getSize());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[6];
      dSCompiler0.subtract(doubleArray0, 0, doubleArray0, 3, doubleArray0, 3);
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
      assertEquals(0, dSCompiler0.getOrder());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[4];
      dSCompiler0.remainder(doubleArray0, 0, doubleArray0, 1, doubleArray0, 1);
      assertEquals(0, dSCompiler0.getOrder());
      assertArrayEquals(new double[] {0.0, Double.NaN, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(21, 1);
      double[] doubleArray0 = new double[4];
      // Undeclared exception!
      try { 
        dSCompiler0.remainder(doubleArray0, 1, doubleArray0, 1, doubleArray0, 1);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 4
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 2);
      double[] doubleArray0 = new double[6];
      // Undeclared exception!
      try { 
        dSCompiler0.pow(doubleArray0, 2, (double) 2, doubleArray0, 974);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 974
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[1];
      // Undeclared exception!
      try { 
        dSCompiler0.pow(doubleArray0, 34, 0, doubleArray0, 34);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 34
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 124);
      double[] doubleArray0 = new double[4];
      // Undeclared exception!
      try { 
        dSCompiler0.pow(doubleArray0, 0, 199, doubleArray0, 2914);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 2914
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 7);
      double[] doubleArray0 = new double[6];
      dSCompiler0.rootN(doubleArray0, 0, 0, doubleArray0, 0);
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 2);
      double[] doubleArray0 = new double[6];
      dSCompiler0.pow(doubleArray0, 2, doubleArray0, 1, doubleArray0, 1);
      assertEquals(2, dSCompiler0.getOrder());
      assertArrayEquals(new double[] {0.0, Double.NaN, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[21];
      // Undeclared exception!
      try { 
        dSCompiler0.log1p(doubleArray0, 2, doubleArray0, 1196);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 1196
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 1);
      double[] doubleArray0 = new double[8];
      // Undeclared exception!
      try { 
        dSCompiler0.log1p(doubleArray0, 0, doubleArray0, 30);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 30
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[1];
      dSCompiler0.log10(doubleArray0, 0, doubleArray0, 0);
      assertArrayEquals(new double[] {Double.NEGATIVE_INFINITY}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 2);
      double[] doubleArray0 = new double[5];
      // Undeclared exception!
      try { 
        dSCompiler0.log10(doubleArray0, 0, doubleArray0, 55);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 55
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      // Undeclared exception!
      try { 
        dSCompiler0.cos(doubleArray0, 1, doubleArray0, (-1033));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -1033
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 2);
      double[] doubleArray0 = new double[6];
      dSCompiler0.cos(doubleArray0, 0, doubleArray0, 0);
      assertArrayEquals(new double[] {1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[5];
      dSCompiler0.sin(doubleArray0, 0, doubleArray0, 0);
      assertEquals(0, dSCompiler0.getOrder());
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 2);
      double[] doubleArray0 = new double[5];
      dSCompiler0.sin(doubleArray0, 2, doubleArray0, 0);
      assertEquals(2, dSCompiler0.getOrder());
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[1];
      // Undeclared exception!
      try { 
        dSCompiler0.tan(doubleArray0, 0, doubleArray0, 10);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 10
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 2);
      double[] doubleArray0 = new double[6];
      // Undeclared exception!
      try { 
        dSCompiler0.tan(doubleArray0, 1, doubleArray0, 473);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 473
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[5];
      dSCompiler0.acos(doubleArray0, 0, doubleArray0, 0);
      assertArrayEquals(new double[] {1.5707963267948966, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 7);
      double[] doubleArray0 = new double[6];
      dSCompiler0.acos(doubleArray0, 0, doubleArray0, 0);
      assertEquals(7, dSCompiler0.getOrder());
      assertArrayEquals(new double[] {1.5707963267948966, 0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[5];
      dSCompiler0.asin(doubleArray0, 0, doubleArray0, 0);
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
      assertEquals(0, dSCompiler0.getOrder());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 2);
      double[] doubleArray0 = new double[5];
      dSCompiler0.asin(doubleArray0, 2, doubleArray0, 2);
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 1007);
      double[] doubleArray0 = new double[5];
      // Undeclared exception!
      dSCompiler0.asin(doubleArray0, 3, doubleArray0, 4477);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 2);
      double[] doubleArray0 = new double[6];
      dSCompiler0.atan(doubleArray0, 0, doubleArray0, 0);
      assertEquals(2, dSCompiler0.getOrder());
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 1214);
      double[] doubleArray0 = new double[5];
      // Undeclared exception!
      dSCompiler0.atan(doubleArray0, 0, doubleArray0, 1214);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[1];
      dSCompiler0.cosh(doubleArray0, 0, doubleArray0, 0);
      assertEquals(0, dSCompiler0.getOrder());
      assertArrayEquals(new double[] {1.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[20];
      dSCompiler0.sinh(doubleArray0, 0, doubleArray0, 0);
      assertEquals(0, dSCompiler0.getOrder());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(1, 1);
      double[] doubleArray0 = new double[7];
      dSCompiler0.sinh(doubleArray0, 1, doubleArray0, 1);
      assertEquals(1, dSCompiler0.getOrder());
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 2);
      double[] doubleArray0 = new double[6];
      dSCompiler0.sinh(doubleArray0, 2, doubleArray0, 0);
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
      assertEquals(2, dSCompiler0.getOrder());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[20];
      // Undeclared exception!
      try { 
        dSCompiler0.tanh(doubleArray0, 0, doubleArray0, 54);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 54
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 2);
      double[] doubleArray0 = new double[6];
      dSCompiler0.tanh(doubleArray0, 2, doubleArray0, 2);
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[3];
      dSCompiler0.acosh(doubleArray0, 0, doubleArray0, 2);
      assertArrayEquals(new double[] {0.0, 0.0, Double.NaN}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[1];
      dSCompiler0.asinh(doubleArray0, 0, doubleArray0, 0);
      assertEquals(0, dSCompiler0.getOrder());
      assertArrayEquals(new double[] {0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 10);
      // Undeclared exception!
      try { 
        dSCompiler0.asinh(doubleArray0, 2, doubleArray0, 10);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 10
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 0);
      double[] doubleArray0 = new double[20];
      dSCompiler0.atanh(doubleArray0, 0, doubleArray0, 0);
      assertEquals(1, dSCompiler0.getSize());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 2);
      double[] doubleArray0 = new double[6];
      dSCompiler0.atanh(doubleArray0, 0, doubleArray0, 0);
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(1, 1);
      assertEquals(2, dSCompiler0.getSize());
      
      double[] doubleArray0 = new double[3];
      double double0 = dSCompiler0.taylor(doubleArray0, 1, doubleArray0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(171, 0);
      dSCompiler0.checkCompatibility(dSCompiler0);
      assertEquals(1, dSCompiler0.getSize());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 2);
      DSCompiler dSCompiler1 = DSCompiler.getCompiler(2, 2);
      try { 
        dSCompiler0.checkCompatibility(dSCompiler1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 0 != 2
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      DSCompiler dSCompiler0 = DSCompiler.getCompiler(0, 1);
      DSCompiler dSCompiler1 = DSCompiler.getCompiler(0, 2);
      try { 
        dSCompiler1.checkCompatibility(dSCompiler0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 2 != 1
         //
         verifyException("org.apache.commons.math3.analysis.differentiation.DSCompiler", e);
      }
  }
}