/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:56:38 GMT 2023
 */

package org.apache.commons.math.complex;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.complex.Complex;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Complex_ESTest extends Complex_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.I.sqrt();
      assertEquals(0.7071067811865476, complex1.getReal(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex0.abs(), 0.01);
      assertEquals(0.7071067811865475, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.ONE.asin();
      assertEquals(1.5707963267948966, complex1.getReal(), 0.01);
      assertEquals(-0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.I.conjugate();
      Complex complex2 = complex1.atan();
      assertTrue(complex2.isInfinite());
      assertFalse(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.atan();
      assertEquals(0.0, complex1.abs(), 0.01);
      assertTrue(complex1.equals((Object)complex0));
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.divide(complex0);
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.pow(complex0);
      Complex complex2 = complex0.divide(complex1);
      assertTrue(complex2.equals((Object)complex1));
      assertEquals(Double.POSITIVE_INFINITY, complex0.abs(), 0.01);
      assertNotSame(complex2, complex1);
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = Complex.ONE;
      boolean boolean0 = complex1.equals(complex0);
      assertFalse(boolean0);
      assertEquals(1.0, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      boolean boolean0 = complex0.equals(complex0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      boolean boolean0 = complex0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = Complex.I;
      boolean boolean0 = complex1.equals(complex0);
      assertFalse(complex0.equals((Object)complex1));
      assertTrue(complex0.isNaN());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.I.conjugate();
      Complex complex2 = complex0.I.divide(complex0);
      boolean boolean0 = complex2.equals(complex1);
      assertEquals((-1.0), complex1.getImaginary(), 0.01);
      assertFalse(complex1.equals((Object)complex2));
      assertEquals(0.0, complex2.abs(), 0.01);
      assertFalse(boolean0);
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertEquals(0.0, complex2.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = Complex.ONE;
      Complex complex2 = complex0.I.divide(complex1);
      Complex complex3 = Complex.I;
      boolean boolean0 = complex2.equals(complex3);
      assertTrue(boolean0);
      assertFalse(complex3.equals((Object)complex1));
      assertEquals(1.0, complex3.abs(), 0.01);
      assertTrue(complex0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Complex complex0 = Complex.INF;
      complex0.hashCode();
      assertFalse(complex0.isNaN());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Complex complex0 = new Complex(0.5, Double.NaN);
      double double0 = complex0.abs();
      assertTrue(complex0.isNaN());
      assertEquals(Double.NaN, double0, 0.01);
      assertEquals(0.5, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      boolean boolean0 = complex0.isInfinite();
      assertTrue(complex0.isNaN());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Complex complex0 = new Complex((-0.746512178863), Double.POSITIVE_INFINITY);
      Complex complex1 = complex0.atan();
      assertTrue(complex1.isNaN());
      assertEquals((-0.746512178863), complex0.getReal(), 0.01);
      assertTrue(complex0.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = Complex.NaN;
      Complex complex2 = complex0.pow(complex1);
      assertEquals(1.0, complex0.abs(), 0.01);
      assertSame(complex2, complex1);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Complex complex0 = new Complex((-0.746512178863), Double.POSITIVE_INFINITY);
      Complex complex1 = complex0.acos();
      assertEquals(Double.POSITIVE_INFINITY, complex0.getImaginary(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex0.abs(), 0.01);
      assertTrue(complex0.isInfinite());
      assertEquals(Double.NaN, complex1.abs(), 0.01);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = Complex.INF;
      Complex complex2 = complex0.pow(complex1);
      assertEquals(1.0, complex0.abs(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex1.abs(), 0.01);
      assertEquals(Double.NaN, complex2.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = new Complex(2495.6962, Double.POSITIVE_INFINITY);
      Complex complex2 = complex0.multiply(complex1);
      assertEquals(2495.6962, complex1.getReal(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
      assertTrue(complex2.isInfinite());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.NaN.negate();
      assertTrue(complex1.isNaN());
      assertFalse(complex0.isNaN());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.NaN.subtract(complex0);
      assertTrue(complex1.isNaN());
      assertFalse(complex0.isNaN());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.NaN.sqrt1z();
      assertTrue(complex1.isNaN());
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex0.abs(), 0.01);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.NaN.acos();
      assertTrue(complex1.isNaN());
      assertEquals(Double.POSITIVE_INFINITY, complex0.abs(), 0.01);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.atan();
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.ZERO.cos();
      assertEquals(1.0, complex1.abs(), 0.01);
      assertFalse(complex1.equals((Object)complex0));
      assertEquals(1.0, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.NaN.cos();
      assertTrue(complex1.isNaN());
      assertTrue(complex0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.ZERO.cosh();
      assertTrue(complex1.equals((Object)complex0));
      assertEquals(1.0, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.NaN.cosh();
      assertFalse(complex0.isNaN());
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Complex complex0 = Complex.INF;
      // Undeclared exception!
      try { 
        complex0.pow((Complex) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.complex.Complex", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.sin();
      assertEquals(0.8414709848078965, complex1.abs(), 0.01);
      assertEquals(0.8414709848078965, complex1.getReal(), 0.01);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.NaN.sin();
      assertFalse(complex0.isNaN());
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Complex complex0 = Complex.INF;
      complex0.sinh();
      assertEquals(Double.POSITIVE_INFINITY, complex0.abs(), 0.01);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.sinh();
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.ONE.tan();
      assertEquals(1.557407724654902, complex1.getReal(), 0.01);
      assertEquals(1.557407724654902, complex1.abs(), 0.01);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.NaN.tan();
      assertFalse(complex0.isNaN());
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.tanh();
      assertEquals(1.557407724654902, complex1.getImaginary(), 0.01);
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertEquals(1.557407724654902, complex1.abs(), 0.01);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.tanh();
      assertSame(complex1, complex0);
  }
}