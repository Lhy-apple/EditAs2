/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:28:34 GMT 2023
 */

package org.apache.commons.math.complex;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.List;
import org.apache.commons.math.complex.Complex;
import org.apache.commons.math.complex.ComplexField;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Complex_ESTest extends Complex_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = (Complex)complex0.INF.readResolve();
      assertTrue(complex1.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Complex complex0 = Complex.valueOf(8.251545029714408E-9, 8.251545029714408E-9);
      String string0 = complex0.toString();
      assertEquals("(8.251545029714408E-9, 8.251545029714408E-9)", string0);
      assertFalse(complex0.isInfinite());
      assertFalse(complex0.isNaN());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Complex complex0 = new Complex((-1288.715), (-1288.715));
      complex0.pow(complex0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Complex complex0 = Complex.I;
      ComplexField complexField0 = complex0.getField();
      assertNotNull(complexField0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      boolean boolean0 = complex0.isNaN();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.pow((-1763.771));
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
      assertFalse(complex1.isInfinite());
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      List<Complex> list0 = complex0.I.nthRoot(35);
      assertEquals(35, list0.size());
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertFalse(list0.contains(complex0));
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.INF.sqrt();
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
      assertFalse(complex1.isInfinite());
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Complex complex0 = Complex.valueOf(1.1102230246251565E-16, Double.POSITIVE_INFINITY);
      Complex complex1 = complex0.multiply(1.1102230246251565E-16);
      assertTrue(complex0.isInfinite());
      assertEquals(1.1102230246251565E-16, complex0.getReal(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex0.getImaginary(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      double double0 = complex0.abs();
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.NaN.add(complex0);
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.acos();
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.NaN.add((-101.6307588081196));
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.ONE.add(Double.NaN);
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.INF.add((-827.061450623));
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
      assertTrue(complex1.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.INF.conjugate();
      assertEquals(Double.NEGATIVE_INFINITY, complex1.getImaginary(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
      assertTrue(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.NaN.conjugate();
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.NaN.divide(complex0);
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.atan();
      boolean boolean0 = complex1.equals(complex0);
      assertEquals(0.0, complex0.getReal(), 0.01);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = Complex.NaN;
      Complex complex2 = complex0.ZERO.divide(complex1);
      assertSame(complex2, complex1);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.atan();
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Complex complex0 = new Complex((-1460.8326065));
      Complex complex1 = Complex.INF;
      Complex complex2 = complex0.divide(complex1);
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertEquals(0.0, complex2.getReal(), 0.01);
      assertEquals((-1460.8326065), complex0.getReal(), 0.01);
      assertEquals(0.0, complex2.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.atan();
      assertEquals(Double.POSITIVE_INFINITY, complex0.getReal(), 0.01);
      assertTrue(complex1.isNaN());
      assertEquals(Double.POSITIVE_INFINITY, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.divide((-2850.2607411206473));
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Complex complex0 = Complex.valueOf(8.251545029714408E-9, 8.251545029714408E-9);
      Complex complex1 = complex0.divide(8.251545029714408E-9);
      assertEquals(8.251545029714408E-9, complex0.getReal(), 0.01);
      assertEquals(1.0, complex1.getImaginary(), 0.01);
      assertEquals(1.0, complex1.getReal(), 0.01);
      assertFalse(complex1.isInfinite());
      assertEquals(8.251545029714408E-9, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.INF.divide(Double.NaN);
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Complex complex0 = Complex.valueOf(0.0, 0.0);
      Complex complex1 = complex0.divide(0.0);
      assertEquals(0.0, complex0.getReal(), 0.01);
      assertFalse(complex0.isNaN());
      assertFalse(complex0.isInfinite());
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
      assertEquals(0.0, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.divide(Double.POSITIVE_INFINITY);
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertNotSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.divide(Double.POSITIVE_INFINITY);
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      boolean boolean0 = complex0.equals(complex0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      boolean boolean0 = complex0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = Complex.NaN;
      Complex complex2 = complex0.sinh();
      boolean boolean0 = complex2.equals(complex1);
      assertFalse(complex1.equals((Object)complex0));
      assertEquals(0.0, complex2.getImaginary(), 0.01);
      assertEquals(0.0, complex2.getReal(), 0.01);
      assertFalse(complex2.isInfinite());
      assertTrue(complex0.equals((Object)complex2));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = Complex.NaN;
      boolean boolean0 = complex1.equals(complex0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Complex complex0 = Complex.valueOf((-1288.715), (-1288.715));
      Complex complex1 = new Complex((-1288.715), (-3914.7053639));
      boolean boolean0 = complex1.equals(complex0);
      assertFalse(complex0.equals((Object)complex1));
      assertFalse(boolean0);
      assertEquals((-1288.715), complex1.getReal(), 0.01);
      assertFalse(complex1.isInfinite());
      assertEquals((-1288.715), complex0.getImaginary(), 0.01);
      assertEquals((-1288.715), complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Complex complex0 = Complex.I;
      complex0.I.hashCode();
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = Complex.ONE;
      Complex complex2 = complex1.multiply(complex0);
      assertSame(complex2, complex0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = new Complex((-4.0E-9), Double.POSITIVE_INFINITY);
      Complex complex2 = complex1.multiply(complex0);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
      assertTrue(complex1.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex2.getReal(), 0.01);
      assertEquals((-4.0E-9), complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = Complex.INF;
      Complex complex2 = complex0.multiply(complex1);
      assertTrue(complex2.isInfinite());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Complex complex0 = Complex.valueOf(8.251545029714408E-9, 8.251545029714408E-9);
      Complex complex1 = new Complex((-1.0), Double.POSITIVE_INFINITY);
      Complex complex2 = complex0.multiply(complex1);
      assertEquals(Double.POSITIVE_INFINITY, complex2.getReal(), 0.01);
      assertTrue(complex1.isInfinite());
      assertEquals((-1.0), complex1.getReal(), 0.01);
      assertEquals(8.251545029714408E-9, complex0.getImaginary(), 0.01);
      assertEquals(8.251545029714408E-9, complex0.getReal(), 0.01);
      assertFalse(complex0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.multiply(1.1102230246251565E-16);
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.multiply(Double.NaN);
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.multiply((-4.0E-9));
      assertEquals(-0.0, complex1.getImaginary(), 0.01);
      assertFalse(complex1.isInfinite());
      assertEquals(-0.0, complex1.getReal(), 0.01);
      assertFalse(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.multiply(Double.POSITIVE_INFINITY);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.NaN.negate();
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.NaN.subtract(complex0);
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.sqrt1z();
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.NaN.subtract(2363.38602786);
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.I.subtract(Double.NaN);
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Complex complex0 = new Complex(0.0, 858.31712848553);
      Complex complex1 = complex0.subtract(858.31712848553);
      assertFalse(complex1.isInfinite());
      assertEquals(858.31712848553, complex0.getImaginary(), 0.01);
      assertEquals((-858.31712848553), complex1.getReal(), 0.01);
      assertEquals(858.31712848553, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.NaN.acos();
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Complex complex0 = Complex.valueOf((-1288.715), (-1288.715));
      Complex complex1 = complex0.asin();
      assertEquals((-8.201121647703376), complex1.getImaginary(), 0.01);
      assertEquals((-0.7853980881318972), complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.asin();
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.atan();
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.INF.cos();
      assertTrue(complex1.isNaN());
      assertFalse(complex1.isInfinite());
      assertNotSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.NaN.cos();
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Complex complex0 = Complex.valueOf((-1288.715), (-1288.715));
      Complex complex1 = complex0.cosh();
      assertEquals((-1288.715), complex0.getImaginary(), 0.01);
      assertTrue(complex1.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
      assertFalse(complex0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.NaN.cosh();
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.NaN.exp();
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.sin();
      assertTrue(complex1.isNaN());
      assertFalse(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.NaN.sinh();
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.ONE.sqrt1z();
      assertFalse(complex1.isInfinite());
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.I.sqrt();
      assertFalse(complex1.isNaN());
      assertEquals(0.7071067811865475, complex1.getImaginary(), 0.01);
      assertEquals(0.7071067811865476, complex1.getReal(), 0.01);
      assertFalse(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Complex complex0 = new Complex((-1288.715), (-1288.715));
      Complex complex1 = complex0.ZERO.tan();
      assertEquals(0.0, complex1.getImaginary(), 0.01);
      assertFalse(complex0.isNaN());
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertFalse(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.NaN.tan();
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.ZERO.tanh();
      assertEquals(0.0, complex1.getImaginary(), 0.01);
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertFalse(complex1.isNaN());
      assertFalse(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.NaN.tanh();
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Complex complex0 = Complex.INF;
      // Undeclared exception!
      try { 
        complex0.nthRoot((-838));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // cannot compute nth root for null or negative n: -838
         //
         verifyException("org.apache.commons.math.complex.Complex", e);
      }
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Complex complex0 = Complex.I;
      List<Complex> list0 = complex0.NaN.nthRoot(715);
      assertFalse(list0.contains(complex0));
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Complex complex0 = Complex.I;
      List<Complex> list0 = complex0.INF.nthRoot(45);
      assertEquals(1, list0.size());
      assertFalse(list0.contains(complex0));
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Complex complex0 = Complex.valueOf(Double.NaN, Double.NaN);
      assertTrue(complex0.isNaN());
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      Complex complex0 = Complex.valueOf(1.1102230246251565E-16, Double.NaN);
      assertEquals(Double.NaN, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      Complex complex0 = Complex.valueOf(0.0);
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertFalse(complex0.isNaN());
      assertEquals(0.0, complex0.getReal(), 0.01);
      assertFalse(complex0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      Complex complex0 = Complex.valueOf(Double.NaN);
      assertEquals(Double.NaN, complex0.getImaginary(), 0.01);
  }
}
