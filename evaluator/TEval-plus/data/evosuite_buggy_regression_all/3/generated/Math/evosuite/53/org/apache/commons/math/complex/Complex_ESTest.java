/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 18:08:54 GMT 2023
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
      Complex complex0 = new Complex(3.4082234423358275, 3.4082234423358275);
      boolean boolean0 = complex0.isNaN();
      assertEquals(3.4082234423358275, complex0.getReal(), 0.01);
      assertFalse(boolean0);
      assertEquals(3.4082234423358275, complex0.getImaginary(), 0.01);
      assertFalse(complex0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.NaN.sqrt1z();
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Complex complex0 = Complex.I;
      String string0 = complex0.toString();
      assertEquals("(0.0, 1.0)", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.ZERO.acos();
      assertEquals(1.5707963267948966, complex1.getReal(), 0.01);
      assertEquals(-0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.ONE.pow(complex0);
      assertEquals(1.0, complex1.getReal(), 0.01);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
      assertFalse(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Complex complex0 = Complex.I;
      ComplexField complexField0 = complex0.getField();
      assertNotNull(complexField0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Complex complex0 = Complex.I;
      List<Complex> list0 = complex0.I.nthRoot(235);
      assertEquals(235, list0.size());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.sqrt();
      assertFalse(complex1.equals((Object)complex0));
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
      assertFalse(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.INF.atan();
      assertTrue(complex1.isNaN());
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
      double double0 = complex0.abs();
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.ZERO.conjugate();
      assertFalse(complex1.isInfinite());
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertFalse(complex1.isNaN());
      assertEquals(-0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.NaN.divide(complex0);
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.ZERO.divide(complex0);
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.divide(complex0);
      boolean boolean0 = complex0.equals(complex1);
      assertTrue(complex1.isNaN());
      assertEquals(Double.POSITIVE_INFINITY, complex0.getReal(), 0.01);
      assertFalse(complex1.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex0.getImaginary(), 0.01);
      assertFalse(complex1.equals((Object)complex0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Object object0 = new Object();
      boolean boolean0 = complex0.equals(object0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Complex complex0 = Complex.INF;
      boolean boolean0 = complex0.equals(complex0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = Complex.ONE;
      boolean boolean0 = complex0.equals(complex1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.createComplex(0.0, 0.0);
      boolean boolean0 = complex0.equals(complex1);
      assertFalse(complex1.isInfinite());
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertFalse(complex1.isNaN());
      assertFalse(boolean0);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
      assertFalse(complex1.equals((Object)complex0));
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = (Complex)complex0.readResolve();
      boolean boolean0 = complex0.equals(complex1);
      assertTrue(boolean0);
      assertFalse(complex1.isInfinite());
      assertEquals(0.0, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Complex complex0 = Complex.I;
      complex0.hashCode();
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.NaN.log();
      Complex complex2 = complex1.I.multiply(complex1);
      assertFalse(complex2.isInfinite());
      assertSame(complex2, complex1);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = new Complex(0.058823529411764705, Double.POSITIVE_INFINITY);
      Complex complex2 = complex1.multiply(complex0);
      assertEquals(0.058823529411764705, complex1.getReal(), 0.01);
      assertTrue(complex1.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex2.getImaginary(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = Complex.INF;
      Complex complex2 = complex0.multiply(complex1);
      assertEquals(Double.POSITIVE_INFINITY, complex2.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = new Complex((-1755.073901629311), Double.POSITIVE_INFINITY);
      Complex complex2 = complex0.multiply(complex1);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
      assertTrue(complex1.isInfinite());
      assertEquals((-1755.073901629311), complex1.getReal(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex2.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.multiply((-1726.9));
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.I.multiply(0.0);
      assertFalse(complex1.isNaN());
      assertFalse(complex1.isInfinite());
      assertEquals(0.0, complex1.getImaginary(), 0.01);
      assertEquals(0.0, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.multiply(Double.NaN);
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.multiply(541.4062421868156);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Complex complex0 = new Complex(1199.628774345, Double.POSITIVE_INFINITY);
      Complex complex1 = complex0.multiply(1199.628774345);
      assertTrue(complex0.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
      assertEquals(1199.628774345, complex0.getReal(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.multiply(Double.POSITIVE_INFINITY);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.NaN.negate();
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.NaN.subtract(complex0);
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.NaN.acos();
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.asin();
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.NaN.asin();
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.cos();
      assertFalse(complex1.isInfinite());
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.NaN.cos();
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.cosh();
      assertFalse(complex1.isInfinite());
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.NaN.exp();
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.I.sin();
      assertFalse(complex1.isInfinite());
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertFalse(complex1.isNaN());
      assertEquals(1.1752011936438014, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.NaN.sin();
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.sinh();
      assertTrue(complex1.isNaN());
      assertFalse(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.sinh();
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.I.sqrt();
      assertEquals(0.7071067811865476, complex1.getReal(), 0.01);
      assertEquals(0.7071067811865475, complex1.getImaginary(), 0.01);
      assertFalse(complex1.isNaN());
      assertFalse(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.tan();
      assertFalse(complex1.isNaN());
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertFalse(complex1.isInfinite());
      assertEquals(0.761594155955765, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.tanh();
      assertFalse(complex1.isInfinite());
      assertEquals(1.557407724654902, complex1.getImaginary(), 0.01);
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertFalse(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Complex complex0 = Complex.I;
      try { 
        complex0.I.nthRoot((-477));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // cannot compute nth root for null or negative n: -477
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Complex complex0 = Complex.I;
      List<Complex> list0 = complex0.NaN.nthRoot(656);
      assertEquals(1, list0.size());
      assertFalse(list0.contains(complex0));
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      List<Complex> list0 = complex0.INF.nthRoot(512);
      assertFalse(list0.contains(complex0));
      assertEquals(1, list0.size());
  }
}