/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:05:36 GMT 2023
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
      Complex complex0 = new Complex(1.325E-8, 1.325E-8);
      String string0 = complex0.toString();
      assertFalse(complex0.isNaN());
      assertEquals("(1.325E-8, 1.325E-8)", string0);
      assertFalse(complex0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      complex0.I.pow(complex0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Complex complex0 = Complex.INF;
      ComplexField complexField0 = complex0.INF.getField();
      assertNotNull(complexField0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      boolean boolean0 = complex0.isNaN();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Complex complex0 = Complex.valueOf(1.325E-8);
      Complex complex1 = complex0.ONE.pow(Double.NaN);
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertEquals(1.325E-8, complex0.getReal(), 0.01);
      assertFalse(complex0.isNaN());
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
      assertFalse(complex0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      List<Complex> list0 = complex0.I.nthRoot(181);
      assertEquals(181, list0.size());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Complex complex0 = Complex.valueOf(151.8948878062917, (-4516.2025));
      Complex complex1 = complex0.tan();
      assertEquals(151.8948878062917, complex0.getReal(), 0.01);
      assertEquals((-4516.2025), complex0.getImaginary(), 0.01);
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertTrue(complex1.isNaN());
      assertFalse(complex0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      double double0 = complex0.abs();
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Complex complex0 = Complex.valueOf(151.8948878062917, Double.NEGATIVE_INFINITY);
      Complex complex1 = complex0.acos();
      assertEquals(151.8948878062917, complex0.getReal(), 0.01);
      assertTrue(complex0.isInfinite());
      assertEquals(Double.NEGATIVE_INFINITY, complex0.getImaginary(), 0.01);
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      double double0 = complex0.abs();
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Complex complex0 = Complex.valueOf(1.325E-8, Double.POSITIVE_INFINITY);
      Complex complex1 = complex0.asin();
      assertEquals(Double.POSITIVE_INFINITY, complex0.getImaginary(), 0.01);
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
      assertEquals(1.325E-8, complex0.getReal(), 0.01);
      assertTrue(complex0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.NaN.add(1.325E-8);
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.ZERO.add((-591.4397575237164));
      assertFalse(complex1.isInfinite());
      assertFalse(complex1.isNaN());
      assertEquals((-591.4397575237164), complex1.getReal(), 0.01);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.add(Double.NaN);
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.INF.conjugate();
      assertTrue(complex1.isInfinite());
      assertEquals(Double.NEGATIVE_INFINITY, complex1.getImaginary(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Complex complex0 = new Complex((-0.48532772611885094), (-0.48532772611885094));
      Complex complex1 = complex0.NaN.conjugate();
      assertFalse(complex0.isNaN());
      assertFalse(complex0.isInfinite());
      assertEquals((-0.48532772611885094), complex0.getReal(), 0.01);
      assertTrue(complex1.isNaN());
      assertEquals((-0.48532772611885094), complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Complex complex0 = Complex.valueOf(1.325E-8);
      Complex complex1 = complex0.NaN.divide(complex0);
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
      assertFalse(complex0.isInfinite());
      assertEquals(1.325E-8, complex0.getReal(), 0.01);
      assertFalse(complex0.isNaN());
      assertEquals(0.0, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Complex complex0 = new Complex(1.325E-8, 1.325E-8);
      Complex complex1 = Complex.NaN;
      Complex complex2 = complex0.ZERO.divide(complex1);
      assertSame(complex2, complex1);
      assertEquals(1.325E-8, complex0.getReal(), 0.01);
      assertEquals(1.325E-8, complex0.getImaginary(), 0.01);
      assertFalse(complex0.isNaN());
      assertFalse(complex0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.I.atan();
      Complex complex2 = complex1.divide(Double.POSITIVE_INFINITY);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
      assertTrue(complex2.isNaN());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Complex complex0 = Complex.valueOf(0.0, 0.0);
      Complex complex1 = complex0.divide(complex0);
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertFalse(complex0.isInfinite());
      assertFalse(complex0.isNaN());
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
      assertEquals(0.0, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = Complex.I;
      Complex complex2 = complex1.divide(complex0);
      assertEquals(0.0, complex2.getReal(), 0.01);
      assertEquals(0.0, complex2.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Complex complex0 = Complex.valueOf(1.325E-8, Double.POSITIVE_INFINITY);
      Complex complex1 = complex0.divide(complex0);
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
      assertTrue(complex0.isInfinite());
      assertFalse(complex1.isInfinite());
      assertEquals(1.325E-8, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.divide(3629.0);
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Complex complex0 = Complex.valueOf(1.325E-8);
      Complex complex1 = complex0.divide(Double.POSITIVE_INFINITY);
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertEquals(1.325E-8, complex0.getReal(), 0.01);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.ZERO.divide(Double.NaN);
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.divide(0.0);
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.divide(0.0);
      assertTrue(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.divide(1.325E-8);
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertFalse(complex1.isNaN());
      assertFalse(complex1.isInfinite());
      assertNotSame(complex1, complex0);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Complex complex0 = new Complex(1.3245471311735498E-8, 1.3245471311735498E-8);
      Complex complex1 = complex0.atan();
      boolean boolean0 = complex0.equals(complex1);
      assertEquals(1.3245471210926935E-8, complex1.getImaginary(), 0.01);
      assertEquals(1.3245471311735498E-8, complex1.getReal(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Complex complex0 = new Complex(1.3245471311735498E-8, 1.3245471311735498E-8);
      boolean boolean0 = complex0.equals(complex0);
      assertFalse(complex0.isInfinite());
      assertFalse(complex0.isNaN());
      assertTrue(boolean0);
      assertEquals(1.3245471311735498E-8, complex0.getImaginary(), 0.01);
      assertEquals(1.3245471311735498E-8, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Complex complex0 = Complex.valueOf((-523.5731143523807));
      boolean boolean0 = complex0.equals((Object) null);
      assertFalse(complex0.isNaN());
      assertFalse(boolean0);
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertEquals((-523.5731143523807), complex0.getReal(), 0.01);
      assertFalse(complex0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Complex complex0 = Complex.valueOf((-471.1188927329418));
      Complex complex1 = Complex.NaN;
      boolean boolean0 = complex0.equals(complex1);
      assertFalse(complex1.equals((Object)complex0));
      assertFalse(complex0.isInfinite());
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertEquals((-471.1188927329418), complex0.getReal(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Complex complex0 = new Complex(1.3245471311735498E-8, 1.3245471311735498E-8);
      Complex complex1 = Complex.INF;
      boolean boolean0 = complex0.equals(complex1);
      assertEquals(1.3245471311735498E-8, complex0.getImaginary(), 0.01);
      assertFalse(complex0.isNaN());
      assertFalse(complex0.isInfinite());
      assertFalse(boolean0);
      assertEquals(1.3245471311735498E-8, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Complex complex0 = Complex.valueOf((-471.1188927329418));
      Complex complex1 = (Complex)complex0.readResolve();
      boolean boolean0 = complex0.equals(complex1);
      assertEquals((-471.1188927329418), complex0.getReal(), 0.01);
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertFalse(complex1.isInfinite());
      assertTrue(boolean0);
      assertFalse(complex1.isNaN());
      assertEquals(0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      complex0.ONE.hashCode();
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Complex complex0 = Complex.valueOf(Double.NaN);
      complex0.NaN.hashCode();
      assertEquals(Double.NaN, complex0.getImaginary(), 0.01);
      assertTrue(complex0.isNaN());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = Complex.INF;
      Complex complex2 = complex0.I.pow(complex1);
      assertNotSame(complex2, complex0);
      assertFalse(complex2.isInfinite());
      assertEquals(Double.NaN, complex2.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = new Complex(615.3564, Double.POSITIVE_INFINITY);
      Complex complex2 = complex0.multiply(complex1);
      assertEquals(615.3564, complex1.getReal(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
      assertTrue(complex2.isInfinite());
      assertTrue(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.NaN.pow(2388.7248851);
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Complex complex0 = new Complex(1.325E-8, 1.325E-8);
      Complex complex1 = complex0.multiply(Double.POSITIVE_INFINITY);
      assertFalse(complex0.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex1.getReal(), 0.01);
      assertEquals(1.325E-8, complex0.getImaginary(), 0.01);
      assertEquals(1.325E-8, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Complex complex0 = new Complex(1.325E-8, 1.325E-8);
      Complex complex1 = Complex.valueOf(1.325E-8, Double.POSITIVE_INFINITY);
      Complex complex2 = complex1.divide(complex0);
      Complex complex3 = complex2.multiply(Double.POSITIVE_INFINITY);
      assertEquals(1.325E-8, complex1.getReal(), 0.01);
      assertNotSame(complex3, complex2);
      assertTrue(complex1.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, complex3.getReal(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
      assertTrue(complex3.equals((Object)complex2));
      assertFalse(complex0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Complex complex0 = Complex.valueOf(151.8948878062917, Double.NEGATIVE_INFINITY);
      Complex complex1 = complex0.multiply(1.325E-8);
      assertEquals(Double.POSITIVE_INFINITY, complex1.getImaginary(), 0.01);
      assertEquals(Double.NEGATIVE_INFINITY, complex0.getImaginary(), 0.01);
      assertTrue(complex0.isInfinite());
      assertEquals(151.8948878062917, complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.ZERO.multiply(9.220590270857665E-9);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertFalse(complex1.isNaN());
      assertFalse(complex1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.NaN.negate();
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = Complex.valueOf(151.8948878062917, Double.NEGATIVE_INFINITY);
      Complex complex2 = complex1.exp();
      Complex complex3 = complex2.subtract(complex0);
      assertEquals(Double.NEGATIVE_INFINITY, complex1.getImaginary(), 0.01);
      assertEquals(151.8948878062917, complex1.getReal(), 0.01);
      assertTrue(complex1.isInfinite());
      assertFalse(complex2.isInfinite());
      assertTrue(complex3.isNaN());
      assertNotSame(complex3, complex2);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.NaN.sqrt1z();
      assertSame(complex1, complex0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.NaN.subtract(5264.241261069);
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Complex complex0 = Complex.valueOf(1.325E-8);
      Complex complex1 = complex0.ZERO.subtract(Double.NaN);
      assertEquals(1.325E-8, complex0.getReal(), 0.01);
      assertFalse(complex0.isNaN());
      assertFalse(complex0.isInfinite());
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
      assertEquals(0.0, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.subtract(918.7090557145635);
      assertEquals((-918.7090557145635), complex1.getReal(), 0.01);
      assertFalse(complex1.isNaN());
      assertFalse(complex1.isInfinite());
      assertEquals(0.0, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.NaN.acos();
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Complex complex0 = new Complex((-824.31040578502), (-824.31040578502));
      Complex complex1 = complex0.NaN.asin();
      assertFalse(complex0.isNaN());
      assertFalse(complex0.isInfinite());
      assertEquals((-824.31040578502), complex0.getImaginary(), 0.01);
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
      assertEquals((-824.31040578502), complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Complex complex0 = Complex.NaN;
      Complex complex1 = complex0.atan();
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.cos();
      assertEquals(1.5430806348152437, complex1.getReal(), 0.01);
      assertFalse(complex1.isInfinite());
      assertEquals(-0.0, complex1.getImaginary(), 0.01);
      assertFalse(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Complex complex0 = new Complex(1.325E-8, 1.325E-8);
      Complex complex1 = complex0.NaN.cos();
      assertFalse(complex0.isInfinite());
      assertFalse(complex0.isNaN());
      assertEquals(Double.NaN, complex1.getReal(), 0.01);
      assertEquals(1.325E-8, complex0.getReal(), 0.01);
      assertEquals(1.325E-8, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.I.cosh();
      assertFalse(complex1.isInfinite());
      assertFalse(complex1.isNaN());
      assertEquals(0.0, complex1.getImaginary(), 0.01);
      assertEquals(0.5403023058681398, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Complex complex0 = new Complex(0.11764700710773468, 0.11764700710773468);
      Complex complex1 = complex0.NaN.cosh();
      assertEquals(0.11764700710773468, complex0.getImaginary(), 0.01);
      assertFalse(complex0.isInfinite());
      assertFalse(complex0.isNaN());
      assertEquals(0.11764700710773468, complex0.getReal(), 0.01);
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.sin();
      assertEquals(0.0, complex1.getImaginary(), 0.01);
      assertFalse(complex1.isNaN());
      assertFalse(complex1.isInfinite());
      assertEquals(0.8414709848078965, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Complex complex0 = Complex.I;
      Complex complex1 = complex0.NaN.sin();
      assertTrue(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.sinh();
      assertFalse(complex1.isInfinite());
      assertEquals(0.0, complex1.getImaginary(), 0.01);
      assertFalse(complex1.isNaN());
      assertEquals(0.0, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Complex complex0 = new Complex((-0.8414709715624252));
      Complex complex1 = complex0.NaN.sinh();
      assertFalse(complex0.isNaN());
      assertFalse(complex0.isInfinite());
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
      assertEquals((-0.8414709715624252), complex0.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.acos();
      assertEquals(-0.0, complex1.getImaginary(), 0.01);
      assertEquals(0.0, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Complex complex0 = Complex.ONE;
      Complex complex1 = complex0.I.sqrt();
      assertFalse(complex1.isNaN());
      assertFalse(complex1.isInfinite());
      assertEquals(0.7071067811865475, complex1.getImaginary(), 0.01);
      assertEquals(0.7071067811865476, complex1.getReal(), 0.01);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Complex complex0 = new Complex(0.16666666666666666);
      Complex complex1 = complex0.NaN.tan();
      assertFalse(complex0.isNaN());
      assertEquals(0.16666666666666666, complex0.getReal(), 0.01);
      assertEquals(0.0, complex0.getImaginary(), 0.01);
      assertFalse(complex0.isInfinite());
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Complex complex0 = Complex.INF;
      Complex complex1 = complex0.ZERO.tanh();
      assertFalse(complex1.isInfinite());
      assertEquals(0.0, complex1.getReal(), 0.01);
      assertEquals(0.0, complex1.getImaginary(), 0.01);
      assertFalse(complex1.isNaN());
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Complex complex0 = Complex.ZERO;
      Complex complex1 = complex0.NaN.tanh();
      assertEquals(Double.NaN, complex1.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Complex complex0 = Complex.I;
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
  public void test66()  throws Throwable  {
      Complex complex0 = new Complex(1.325E-8, 1.325E-8);
      List<Complex> list0 = complex0.NaN.nthRoot(245);
      assertFalse(list0.contains(complex0));
      assertEquals(1.325E-8, complex0.getReal(), 0.01);
      assertEquals(1, list0.size());
      assertEquals(1.325E-8, complex0.getImaginary(), 0.01);
      assertFalse(complex0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Complex complex0 = Complex.INF;
      List<Complex> list0 = complex0.INF.nthRoot(2146819203);
      assertTrue(list0.contains(complex0));
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Complex complex0 = Complex.valueOf(Double.NaN, Double.NaN);
      assertEquals(Double.NaN, complex0.getImaginary(), 0.01);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Complex complex0 = Complex.valueOf(0.0, Double.NaN);
      assertEquals(Double.NaN, complex0.getReal(), 0.01);
  }
}
