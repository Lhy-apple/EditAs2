/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:05:12 GMT 2023
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
      complex0.acos();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      String string0 = complex0.toString();
      assertEquals("(0.0, 0.0)", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Complex complex0 = Complex.valueOf(2362.108355078492);
      Complex complex1 = complex0.pow(complex0);
      assertFalse(complex1.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
      assertEquals(0.0, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
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
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.INF.pow(6.283185307179586);
      assertTrue(complex1.isNaN());
      assertFalse(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      double double0 = complex0.abs();
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Complex complex0 = new Complex(0.0, 0.0);
      double double0 = complex0.abs();
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertEquals(0.0, double0, 0.01);
      assertEquals(0.0, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.INF.asin();
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertTrue(complex1.isNaN());
      assertEquals(0.0, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Complex complex0 = new Complex((-917.455479161), Double.POSITIVE_INFINITY);
      Complex complex1 = complex0.acos();
      assertEquals((-917.455479161), complex0.getReal(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex0.getImaginary(), 0.01);
      assertTrue(complex0.isInfinite());
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.add(Double.NaN);
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Complex complex0 = new Complex(0.9415234641111108, Double.NaN);
      Complex complex1 = complex0.I.add(Double.NaN);
      assertTrue(complex1.equals((Object)complex0));
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
      assertEquals(0.9415234641111108, complex0.getReal(), 0.01);
      assertEquals(Double.NaN, complex0.getImaginary(), 0.01);
      assertFalse(complex0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.add(0.07490822288864472);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
      assertEquals(0.07490822288864472, complex1.getReal(), 0.01);
      assertFalse(complex1.isNaN());
      assertFalse(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Complex complex0 = Complex.valueOf(526.454466477, Double.POSITIVE_INFINITY);
      Complex complex1 = complex0.ZERO.conjugate();
      assertEquals(Double.POSITIVE_INFINITY, complex0.getImaginary(), 0.01);
      assertFalse(complex1.isInfinite());
      assertEquals(-0.0, complex1.getImaginary(), 0.01);
      assertTrue(complex0.isInfinite());
      assertEquals(526.454466477, complex0.getReal(), 0.01);
      assertEquals(0.0, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.NaN.conjugate();
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Complex complex0 = new Complex((-6.808799537740455E-17), (-6.808799537740455E-17));
      Complex complex1 = complex0.NaN.divide(complex0);
      assertFalse(complex0.isNaN());
      assertEquals((-6.808799537740455E-17), complex0.getImaginary(), 0.01);
      assertEquals((-6.808799537740455E-17), complex0.getReal(), 0.01);
      assertFalse(complex0.isInfinite());
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Complex complex0 = new Complex(0.0, 0.0);
      Complex complex1 = complex0.ZERO.atan();
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertTrue(complex1.equals((Object)complex0));
      assertEquals(0.0, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = Complex.NaN;
      Complex complex2 = complex0.ZERO.divide(complex1);
      assertSame(complex2, complex1);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.I.atan();
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
      assertTrue(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Complex complex0 = new Complex(0.0, 0.0);
      Complex complex1 = complex0.divide(complex0);
      assertFalse(complex0.isNaN());
      assertEquals(0.0, complex0.getReal(), 0.01);
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
      assertFalse(complex0.isInfinite());
      assertEquals(0.0, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = Complex.INF;
      Complex complex2 = complex0.ONE.divide(complex1);
      assertNotSame(complex2, complex0);
      assertEquals(0.0, complex2.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.NaN.divide(Double.POSITIVE_INFINITY);
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Complex complex0 = Complex.valueOf(2362.108355078492);
      Complex complex1 = complex0.divide(1.0);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
      assertFalse(complex1.isInfinite());
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertEquals(2362.108355078492, complex1.getReal(), 0.01);
      assertTrue(complex1.equals((Object)complex0));
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Complex complex0 = new Complex(0.0, (-6.808799537740455E-17));
      Complex complex1 = complex0.INF.divide(Double.NaN);
      assertEquals((-6.808799537740455E-17), complex0.getImaginary(), 0.01);
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
      assertFalse(complex0.isInfinite());
      assertFalse(complex0.isNaN());
      assertEquals(0.0, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Complex complex0 = new Complex(0.0, (-6.808799537740455E-17));
      Complex complex1 = complex0.ZERO.divide(0.0);
      assertEquals((-6.808799537740455E-17), complex0.getImaginary(), 0.01);
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
      assertFalse(complex0.isInfinite());
      assertEquals(0.0, complex0.getReal(), 0.01);
      assertFalse(complex0.isNaN());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Complex complex0 = Complex.valueOf(0.0);
      Complex complex1 = complex0.cosh();
      Complex complex2 = complex1.divide(0.0);
      assertFalse(complex1.isInfinite());
      assertEquals(0.0, complex0.getReal(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex2.getImaginary(), 0.01);
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertEquals(1.0, complex1.getReal(), 0.01);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Complex complex0 = new Complex(2399.0529745, 2399.0529745);
      Complex complex1 = complex0.INF.divide(Double.POSITIVE_INFINITY);
      assertFalse(complex0.isInfinite());
      assertEquals(2399.0529745, complex0.getImaginary(), 0.01);
      assertEquals(2399.0529745, complex0.getReal(), 0.01);
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
      assertFalse(complex0.isNaN());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.divide(Double.POSITIVE_INFINITY);
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Complex complex0 = Complex.valueOf(2362.108355078492);
      Complex complex1 = Complex.valueOf(2362.108355078492, 2362.108355078492);
      boolean boolean0 = complex0.equals(complex1);
      assertFalse(boolean0);
      assertEquals(2362.108355078492, complex1.getReal(), 0.01);
      assertFalse(complex1.isInfinite());
      assertEquals(2362.108355078492, complex1.getImaginary(), 0.01);
      assertFalse(complex1.equals((Object)complex0));
      assertEquals(2362.108355078492, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Complex complex0 = Complex.valueOf(2362.108355078492);
      boolean boolean0 = complex0.equals(complex0);
      assertFalse(complex0.isInfinite());
      assertTrue(boolean0);
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertFalse(complex0.isNaN());
      assertEquals(2362.108355078492, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Complex complex0 = new Complex(1450.5091744);
      Object object0 = new Object();
      boolean boolean0 = complex0.equals(object0);
      assertFalse(boolean0);
      assertFalse(complex0.isInfinite());
      assertFalse(complex0.isNaN());
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertEquals(1450.5091744, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.NaN.atan();
      boolean boolean0 = complex0.equals(complex1);
      assertTrue(complex1.isNaN());
      assertFalse(complex1.equals((Object)complex0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Complex complex0 = new Complex((-1441.65785971744));
      Complex complex1 = complex0.divide(complex0);
      boolean boolean0 = complex0.equals(complex1);
      assertFalse(boolean0);
      assertEquals(-0.0, complex1.getImaginary(), 0.01);
      assertFalse(complex1.isInfinite());
      assertEquals(1.0, complex1.getReal(), 0.01);
      assertFalse(complex1.equals((Object)complex0));
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Complex complex0 = Complex.valueOf(0.167, 0.167);
      Complex complex1 = (Complex)complex0.readResolve();
      boolean boolean0 = complex0.equals(complex1);
      assertFalse(complex1.isNaN());
      assertEquals(0.167, complex1.getImaginary(), 0.01);
      assertEquals(0.167, complex1.getReal(), 0.01);
      assertFalse(complex1.isInfinite());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Complex complex0 = new Complex(4.455505956692757, 1.0);
      complex0.hashCode();
      assertFalse(complex0.isInfinite());
      assertFalse(complex0.isNaN());
      assertEquals(4.455505956692757, complex0.getReal(), 0.01);
      assertEquals(1.0, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.INF.sqrt();
      complex1.hashCode();
      assertFalse(complex1.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.INF.multiply(complex0);
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = Complex.INF;
      Complex complex2 = complex0.multiply(complex1);
      assertEquals(Double.POSITIVE_INFINITY, complex2.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Complex complex0 = new Complex(-0.0, (-6.808799537740455E-17));
      Complex complex1 = Complex.valueOf(-0.0, Double.POSITIVE_INFINITY);
      Complex complex2 = complex0.multiply(complex1);
      assertEquals(-0.0, complex1.getReal(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex2.getImaginary(), 0.01);
      assertTrue(complex1.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
      assertFalse(complex0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.NaN.atan();
      Complex complex2 = complex1.pow(398.4378594);
      assertSame(complex2, complex1);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Complex complex0 = new Complex((-1150.25884998571), (-1150.25884998571));
      Complex complex1 = complex0.multiply(Double.NaN);
      assertEquals((-1150.25884998571), complex0.getImaginary(), 0.01);
      assertEquals((-1150.25884998571), complex0.getReal(), 0.01);
      assertFalse(complex0.isNaN());
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
      assertFalse(complex0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Complex complex0 = Complex.valueOf(2362.108355078492);
      Complex complex1 = complex0.pow(2362.108355078492);
      assertFalse(complex1.isInfinite());
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Complex complex0 = Complex.valueOf(526.454466477, Double.POSITIVE_INFINITY);
      Complex complex1 = complex0.multiply(526.454466477);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
      assertTrue(complex0.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex0.getImaginary(), 0.01);
      assertEquals(526.454466477, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.multiply(Double.POSITIVE_INFINITY);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.NaN.negate();
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.NaN.subtract(complex0);
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.NaN.sqrt1z();
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.subtract((-2058.78070958));
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Complex complex0 = Complex.valueOf(2362.108355078492);
      Complex complex1 = complex0.ONE.subtract(0.0);
      assertFalse(complex0.isInfinite());
      assertEquals(1.0, complex1.getReal(), 0.01);
      assertEquals(2362.108355078492, complex0.getReal(), 0.01);
      assertFalse(complex0.isNaN());
      assertEquals(0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.ZERO.subtract(Double.NaN);
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.acos();
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Complex complex0 = Complex.valueOf(2362.108355078492);
      Complex complex1 = complex0.cos();
      assertEquals(0.9325728765417594, complex1.getReal(), 0.01);
      assertFalse(complex1.isInfinite());
      assertEquals(0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.NaN.cos();
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.cosh();
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.sin();
      assertEquals(0.0, complex1.getImaginary(), 0.01);
      assertFalse(complex1.isInfinite());
      assertFalse(complex1.isNaN());
      assertEquals(0.8414709848078965, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.NaN.sin();
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.I.sinh();
      assertEquals(0.8414709848078965, complex1.getImaginary(), 0.01);
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertFalse(complex1.isNaN());
      assertFalse(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.NaN.sinh();
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.I.sqrt();
      assertFalse(complex1.isInfinite());
      assertEquals(0.7071067811865475, complex1.getImaginary(), 0.01);
      assertEquals(0.7071067811865476, complex1.getReal(), 0.01);
      assertFalse(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Complex complex0 = Complex.valueOf(0.167, 0.167);
      Complex complex1 = complex0.INF.tan();
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
      assertTrue(complex1.isNaN());
      assertFalse(complex0.isNaN());
      assertFalse(complex0.isInfinite());
      assertEquals(0.167, complex0.getReal(), 0.01);
      assertEquals(0.167, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.tan();
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Complex complex0 = Complex.valueOf(2362.108355078492);
      Complex complex1 = complex0.INF.tanh();
      assertTrue(complex1.isNaN());
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertFalse(complex0.isNaN());
      assertEquals(2362.108355078492, complex0.getReal(), 0.01);
      assertFalse(complex0.isInfinite());
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Complex complex0 = new Complex((-2532.60768393747), (-2532.60768393747));
      Complex complex1 = complex0.NaN.tanh();
      assertEquals((-2532.60768393747), complex0.getImaginary(), 0.01);
      assertFalse(complex0.isNaN());
      assertEquals((-2532.60768393747), complex0.getReal(), 0.01);
      assertFalse(complex0.isInfinite());
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Complex complex0 = Complex.INF;
      List<Complex> list0 = complex0.I.nthRoot(12);
      assertFalse(list0.contains(complex0));
      assertEquals(Double.POSITIVE_INFINITY, complex0.getImaginary(), 0.01);
      assertEquals(12, list0.size());
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Complex complex0 = Complex.INF;
      // Undeclared exception!
      try { 
        complex0.ZERO.nthRoot((-1));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // cannot compute nth root for null or negative n: -1
         //
         verifyException("org.apache.commons.math.complex.Complex", e);
      }
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      List<Complex> list0 = complex0.NaN.nthRoot(17);
      assertEquals(1, list0.size());
      assertTrue(list0.contains(complex0));
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Complex complex0 = new Complex((-1377.2973123148954), 0.0);
      List<Complex> list0 = complex0.INF.nthRoot(164);
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertEquals((-1377.2973123148954), complex0.getReal(), 0.01);
      assertFalse(complex0.isNaN());
      assertFalse(complex0.isInfinite());
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Complex complex0 = Complex.valueOf(Double.NaN, Double.NaN);
      assertEquals(Double.NaN, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Complex complex0 = Complex.valueOf(-0.0, Double.NaN);
      assertEquals(Double.NaN, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Complex complex0 = Complex.valueOf(Double.NaN);
      assertEquals(Double.NaN, complex0.getImaginary(), 0.01);
  }
}
