/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:24:43 GMT 2023
 */

package org.apache.commons.math3.complex;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.List;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.complex.ComplexField;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Complex_ESTest extends Complex_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Complex complex0 = Complex.valueOf(6.283185307179586);
      Complex complex1 = complex0.INF.pow(6.283185307179586);
      assertFalse(complex0.isInfinite());
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertTrue(complex1.isNaN());
      assertEquals(6.283185307179586, complex0.getReal(), 0.01);
      assertFalse(complex0.isNaN());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      ComplexField complexField0 = complex0.ZERO.getField();
      assertNotNull(complexField0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = (Complex)complex0.ONE.readResolve();
      assertFalse(complex1.isInfinite());
      assertEquals(0.0, complex1.getImaginary(), 0.01);
      assertEquals(1.0, complex1.getReal(), 0.01);
      assertFalse(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Complex complex0 = Complex.INF;
      String string0 = complex0.ONE.toString();
      assertEquals("(1.0, 0.0)", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.NaN.pow(complex0);
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      boolean boolean0 = complex0.isNaN();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.INF.sqrt();
      assertFalse(complex1.isInfinite());
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Complex complex0 = Complex.valueOf((double) 19, Double.POSITIVE_INFINITY);
      Complex complex1 = Complex.valueOf((double) 19);
      Complex complex2 = complex1.pow(complex0);
      assertEquals(Double.POSITIVE_INFINITY, complex0.getImaginary(), 0.01);
      assertTrue(complex0.isInfinite());
      assertEquals(19.0, complex1.getReal(), 0.01);
      assertEquals(19.0, complex0.getReal(), 0.01);
      assertEquals(Double.NaN, complex2.getImaginary(), 0.01);
      assertFalse(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Complex complex0 = new Complex((-20.0), (-20.0));
      Complex complex1 = complex0.INF.divide(Double.POSITIVE_INFINITY);
      double double0 = complex1.abs();
      assertEquals((-20.0), complex0.getReal(), 0.01);
      assertFalse(complex0.isNaN());
      assertFalse(complex0.isInfinite());
      assertEquals(Double.NaN, double0, 0.01);
      assertEquals((-20.0), complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      double double0 = complex0.abs();
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.INF.asin();
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Complex complex0 = Complex.valueOf((double) 19, (double) 19);
      Complex complex1 = complex0.NaN.add((double) 19);
      assertFalse(complex0.isNaN());
      assertFalse(complex0.isInfinite());
      assertEquals(19.0, complex0.getImaginary(), 0.01);
      assertEquals(19.0, complex0.getReal(), 0.01);
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Complex complex0 = Complex.valueOf(2019.607608677722, Double.POSITIVE_INFINITY);
      Complex complex1 = complex0.add(2019.607608677722);
      assertEquals(4039.215217355444, complex1.getReal(), 0.01);
      assertTrue(complex1.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Complex complex0 = new Complex((-1.0), (-1.0));
      Complex complex1 = complex0.add(Double.NaN);
      assertFalse(complex0.isInfinite());
      assertEquals((-1.0), complex0.getImaginary(), 0.01);
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
      assertFalse(complex0.isNaN());
      assertEquals((-1.0), complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Complex complex0 = new Complex((-1.0), (-1.0));
      Complex complex1 = complex0.INF.conjugate();
      assertFalse(complex0.isInfinite());
      assertTrue(complex1.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
      assertEquals(Double.NEGATIVE_INFINITY, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.NaN.conjugate();
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Complex complex0 = Complex.valueOf((double) 19, (double) 19);
      Complex complex1 = complex0.NaN.divide(complex0);
      assertEquals(19.0, complex0.getImaginary(), 0.01);
      assertFalse(complex0.isInfinite());
      assertFalse(complex0.isNaN());
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
      assertEquals(19.0, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Complex complex0 = Complex.valueOf((-505.0292868256604));
      Complex complex1 = complex0.ZERO.atan();
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = Complex.NaN;
      Complex complex2 = complex0.ZERO.divide(complex1);
      assertSame(complex2, complex1);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.I.atan();
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.INF.atan();
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = Complex.INF;
      Complex complex2 = complex0.ZERO.divide(complex1);
      assertEquals(0.0, complex2.getReal(), 0.01);
      assertEquals(0.0, complex2.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.NaN.divide((-1665.5766));
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Complex complex0 = Complex.valueOf((double) 20, (double) 20);
      Complex complex1 = complex0.INF.divide(Double.NaN);
      assertEquals(20.0, complex0.getImaginary(), 0.01);
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
      assertEquals(20.0, complex0.getReal(), 0.01);
      assertFalse(complex0.isNaN());
      assertFalse(complex0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Complex complex0 = Complex.valueOf((double) 19, (double) 19);
      Complex complex1 = complex0.INF.divide(0.0);
      assertEquals(19.0, complex0.getReal(), 0.01);
      assertFalse(complex0.isNaN());
      assertFalse(complex0.isInfinite());
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
      assertEquals(19.0, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.ZERO.divide((double) 19);
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
      assertFalse(complex1.equals((Object)complex0));
      assertFalse(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Complex complex0 = Complex.valueOf((double) 19, (double) 19);
      Complex complex1 = complex0.divide(Double.POSITIVE_INFINITY);
      assertEquals(19.0, complex0.getImaginary(), 0.01);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertEquals(19.0, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.I.reciprocal();
      assertFalse(complex1.isInfinite());
      assertEquals((-1.0), complex1.getImaginary(), 0.01);
      assertFalse(complex1.isNaN());
      assertEquals(0.0, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.NaN.reciprocal();
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.ZERO.reciprocal();
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.reciprocal();
      assertEquals(0.0, complex1.getImaginary(), 0.01);
      assertEquals(0.0, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.negate();
      boolean boolean0 = complex0.equals(complex1);
      assertEquals(-0.0, complex1.getImaginary(), 0.01);
      assertEquals(-0.0, complex1.getReal(), 0.01);
      assertFalse(complex1.isInfinite());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      boolean boolean0 = complex0.equals(complex0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      boolean boolean0 = complex0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = Complex.NaN;
      boolean boolean0 = complex0.equals(complex1);
      assertFalse(complex1.equals((Object)complex0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.acos();
      boolean boolean0 = complex1.equals(complex0);
      assertEquals(-0.0, complex1.getImaginary(), 0.01);
      assertFalse(boolean0);
      assertEquals(1.5707963267948966, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Complex complex0 = Complex.valueOf((-0.4863099752211644));
      Complex complex1 = Complex.valueOf((-0.4863099752211644), (-0.4863099752211644));
      boolean boolean0 = complex0.equals(complex1);
      assertFalse(boolean0);
      assertEquals((-0.4863099752211644), complex1.getImaginary(), 0.01);
      assertEquals((-0.4863099752211644), complex0.getReal(), 0.01);
      assertFalse(complex1.equals((Object)complex0));
      assertEquals((-0.4863099752211644), complex1.getReal(), 0.01);
      assertFalse(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      complex0.ZERO.hashCode();
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      complex0.NaN.hashCode();
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = Complex.NaN;
      Complex complex2 = complex0.multiply(complex1);
      assertSame(complex2, complex1);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Complex complex0 = Complex.valueOf(1.9722862688653467E192);
      Complex complex1 = complex0.acos();
      assertFalse(complex0.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = Complex.INF;
      Complex complex2 = complex0.multiply(complex1);
      assertEquals(Double.POSITIVE_INFINITY, complex2.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.multiply(1);
      assertTrue(complex1.equals((Object)complex0));
      assertFalse(complex1.isInfinite());
      assertFalse(complex1.isNaN());
      assertEquals(0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.multiply(2091641698);
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.multiply(19);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Complex complex0 = Complex.valueOf((double) 19, Double.POSITIVE_INFINITY);
      Complex complex1 = complex0.multiply(19);
      assertTrue(complex0.isInfinite());
      assertEquals(19.0, complex0.getReal(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex0.getImaginary(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.multiply((double) 8);
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Complex complex0 = Complex.valueOf(Double.NaN, Double.NaN);
      Complex complex1 = complex0.INF.multiply(Double.NaN);
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Complex complex0 = Complex.valueOf((double) 19, Double.POSITIVE_INFINITY);
      Complex complex1 = complex0.multiply(2.0139224764581208E15);
      assertTrue(complex0.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex0.getImaginary(), 0.01);
      assertEquals(19.0, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.multiply(2.0139224764581208E15);
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertFalse(complex1.isNaN());
      assertFalse(complex1.isInfinite());
      assertEquals(2.0139224764581208E15, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Complex complex0 = new Complex((-1.0), (-1.0));
      Complex complex1 = complex0.multiply(Double.POSITIVE_INFINITY);
      assertEquals((-1.0), complex0.getReal(), 0.01);
      assertFalse(complex0.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
      assertEquals((-1.0), complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Complex complex0 = Complex.valueOf((double) 19, (double) 19);
      Complex complex1 = complex0.NaN.subtract(complex0);
      assertEquals(19.0, complex0.getReal(), 0.01);
      assertEquals(19.0, complex0.getImaginary(), 0.01);
      assertFalse(complex0.isNaN());
      assertTrue(complex1.isNaN());
      assertFalse(complex0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Complex complex0 = new Complex((-20.0), (-20.0));
      Complex complex1 = complex0.NaN.sinh();
      Complex complex2 = complex0.ONE.subtract(complex1);
      assertEquals((-20.0), complex0.getReal(), 0.01);
      assertFalse(complex0.isInfinite());
      assertFalse(complex0.isNaN());
      assertSame(complex2, complex1);
      assertEquals((-20.0), complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.subtract(377.89666435);
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.INF.subtract((double) 10);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
      assertTrue(complex1.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Complex complex0 = Complex.valueOf(Double.NaN);
      Complex complex1 = complex0.I.subtract(Double.NaN);
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Complex complex0 = Complex.valueOf(Double.NaN, Double.NaN);
      Complex complex1 = complex0.acos();
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.NaN.asin();
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Complex complex0 = Complex.valueOf((double) 19, (double) 19);
      Complex complex1 = complex0.NaN.atan();
      assertFalse(complex0.isInfinite());
      assertFalse(complex0.isNaN());
      assertEquals(19.0, complex0.getReal(), 0.01);
      assertEquals(19.0, complex0.getImaginary(), 0.01);
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.INF.cos();
      assertTrue(complex1.isNaN());
      assertFalse(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.cos();
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Complex complex0 = Complex.valueOf((-0.4863099752211644));
      Complex complex1 = complex0.cosh();
      assertEquals(1.1205976043905925, complex1.getReal(), 0.01);
      assertFalse(complex1.isInfinite());
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertEquals(-0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.NaN.cosh();
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Complex complex0 = Complex.valueOf((-6.587776621471115E234), (-6.587776621471115E234));
      Complex complex1 = complex0.sin();
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
      assertEquals((-6.587776621471115E234), complex0.getImaginary(), 0.01);
      assertFalse(complex0.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
      assertEquals((-6.587776621471115E234), complex0.getReal(), 0.01);
      assertTrue(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.NaN.sin();
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.ZERO.sinh();
      assertEquals(0.0, complex1.getImaginary(), 0.01);
      assertFalse(complex1.isNaN());
      assertFalse(complex1.isInfinite());
      assertEquals(0.0, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Complex complex0 = Complex.valueOf(1.9722862688653467E192, 1.9722862688653467E192);
      Complex complex1 = complex0.acos();
      assertFalse(complex0.isInfinite());
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
      assertEquals(1.9722862688653467E192, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Complex complex0 = Complex.valueOf((double) 19, (double) 19);
      Complex complex1 = complex0.tanh();
      Complex complex2 = complex1.acos();
      assertEquals((-4.313560832547568E-9), complex2.getImaginary(), 0.01);
      assertEquals(4.313560765676772E-9, complex2.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.ONE.asin();
      assertEquals(1.5707963267948966, complex1.getReal(), 0.01);
      assertEquals(-0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.NaN.tan();
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      Complex complex0 = Complex.valueOf((-0.4863099752211644));
      Complex complex1 = complex0.tan();
      assertFalse(complex1.isInfinite());
      assertEquals((-0.5286575883396332), complex1.getReal(), 0.01);
      assertEquals((-0.4863099752211644), complex0.getReal(), 0.01);
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.tan();
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      Complex complex0 = Complex.valueOf((double) 24, (double) 24);
      Complex complex1 = complex0.tan();
      assertEquals(1.0, complex1.getImaginary(), 0.01);
      assertEquals(24.0, complex0.getReal(), 0.01);
      assertFalse(complex1.isInfinite());
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertEquals(24.0, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      Complex complex0 = Complex.valueOf((double) 19, (-1877.05));
      Complex complex1 = complex0.tan();
      assertFalse(complex1.isInfinite());
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertEquals((-1877.05), complex0.getImaginary(), 0.01);
      assertEquals(19.0, complex0.getReal(), 0.01);
      assertEquals((-1.0), complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.tanh();
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.tanh();
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      Complex complex0 = Complex.valueOf((double) 24, (double) 24);
      Complex complex1 = complex0.tanh();
      assertEquals(24.0, complex0.getReal(), 0.01);
      assertEquals(1.0, complex1.getReal(), 0.01);
      assertFalse(complex1.isInfinite());
      assertEquals(24.0, complex0.getImaginary(), 0.01);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      Complex complex0 = Complex.valueOf((double) (-16778), (double) (-16778));
      Complex complex1 = complex0.tanh();
      assertEquals((-16778.0), complex0.getImaginary(), 0.01);
      assertFalse(complex1.isInfinite());
      assertEquals((-16778.0), complex0.getReal(), 0.01);
      assertEquals((-1.0), complex1.getReal(), 0.01);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test78()  throws Throwable  {
      Complex complex0 = Complex.valueOf((double) 19, (double) 19);
      List<Complex> list0 = complex0.NaN.nthRoot(19);
      assertEquals(19.0, complex0.getReal(), 0.01);
      assertEquals(1, list0.size());
      assertFalse(list0.contains(complex0));
      assertFalse(complex0.isInfinite());
      assertEquals(19.0, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      Complex complex0 = Complex.valueOf((double) (-20), (double) (-20));
      try { 
        complex0.nthRoot((-20));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // cannot compute nth root for null or negative n: -20
         //
         verifyException("org.apache.commons.math3.complex.Complex", e);
      }
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      List<Complex> list0 = complex0.ONE.nthRoot(19);
      assertTrue(list0.contains(complex0));
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertEquals(19, list0.size());
  }

  @Test(timeout = 4000)
  public void test81()  throws Throwable  {
      Complex complex0 = Complex.valueOf((double) 19, Double.NaN);
      assertEquals(Double.NaN, complex0.getReal(), 0.01);
  }
}