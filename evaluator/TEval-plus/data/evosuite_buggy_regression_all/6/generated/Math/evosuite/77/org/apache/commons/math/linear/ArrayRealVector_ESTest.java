/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:54:00 GMT 2023
 */

package org.apache.commons.math.linear;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.linear.ArrayRealVector;
import org.apache.commons.math.linear.OpenMapRealVector;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.linear.RealVector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ArrayRealVector_ESTest extends ArrayRealVector_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      double[] doubleArray0 = arrayRealVector0.toArray();
      assertEquals(0, doubleArray0.length);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      double double0 = openMapRealVector0.getL1Distance((RealVector) arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      String string0 = arrayRealVector0.toString();
      assertEquals("{0}", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      assertEquals(4, arrayRealVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      double double0 = arrayRealVector0.getL1Distance((RealVector) arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(0, arrayRealVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      // Undeclared exception!
      try { 
        arrayRealVector0.projection(doubleArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // vector length mismatch: got 0 but expected 2
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      double[] doubleArray0 = new double[14];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      double double0 = arrayRealVector0.getDistance((RealVector) arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      double double0 = arrayRealVector0.getLInfDistance((RealVector) arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(0, arrayRealVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      ArrayRealVector arrayRealVector1 = arrayRealVector0.append(arrayRealVector0);
      assertEquals(0, arrayRealVector1.getDimension());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.ebeMultiply((RealVector) arrayRealVector0);
      assertEquals(0.0, realVector0.getL1Norm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      try { 
        openMapRealVector0.add((RealVector) arrayRealVector0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // vector must have at least one element
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(419, 419);
      RealVector realVector0 = arrayRealVector0.mapCosh();
      assertFalse(realVector0.equals((Object)arrayRealVector0));
      assertEquals(1.9524141421179683E184, realVector0.getL1Norm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = null;
      try {
        arrayRealVector0 = new ArrayRealVector((double[]) null, (ArrayRealVector) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.linear.ArrayRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      ArrayRealVector arrayRealVector1 = new ArrayRealVector(arrayRealVector0);
      assertEquals(0, arrayRealVector1.getDimension());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      // Undeclared exception!
      try { 
        arrayRealVector0.setSubVector(379, doubleArray0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // index 379 out of allowed range [0, 1]
         //
         verifyException("org.apache.commons.math.linear.AbstractRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(384);
      ArrayRealVector arrayRealVector1 = arrayRealVector0.ebeDivide(arrayRealVector0);
      assertEquals(384, arrayRealVector1.getDimension());
      assertEquals(Double.NaN, arrayRealVector1.getLInfNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      // Undeclared exception!
      try { 
        arrayRealVector0.getSubVector((-2562), (-2562));
        fail("Expecting exception: NegativeArraySizeException");
      
      } catch(NegativeArraySizeException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.linear.ArrayRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      RealVector realVector0 = arrayRealVector0.append(doubleArray0);
      assertEquals(5, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealMatrix realMatrix0 = arrayRealVector0.outerProduct((RealVector) arrayRealVector0);
      assertEquals(1, realMatrix0.getColumnDimension());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      openMapRealVector0.mapAcosToSelf();
      RealVector realVector0 = arrayRealVector0.add((RealVector) openMapRealVector0);
      RealVector realVector1 = realVector0.unitVector();
      assertEquals(1.5707963267948966, realVector0.getNorm(), 0.01);
      assertEquals(1.0, realVector1.getNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Double[] doubleArray0 = new Double[0];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      // Undeclared exception!
      try { 
        arrayRealVector0.set(856, (ArrayRealVector) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.linear.ArrayRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      OpenMapRealVector openMapRealVector1 = (OpenMapRealVector)arrayRealVector0.projection((RealVector) openMapRealVector0);
      assertEquals(Double.NaN, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(377, 377);
      // Undeclared exception!
      try { 
        arrayRealVector0.setEntry(377, 377);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // index 377 out of allowed range [0, 376]
         //
         verifyException("org.apache.commons.math.linear.AbstractRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.append(1.0E-12);
      assertEquals(1.0E-12, realVector0.getLInfNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      double[] doubleArray0 = new double[0];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.append((RealVector) openMapRealVector0);
      assertEquals(0, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.add((RealVector) arrayRealVector0);
      assertEquals(0.0, realVector0.getL1Norm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      arrayRealVector0.set(3553.5856706538);
      assertEquals(0.0, arrayRealVector0.getL1Norm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = null;
      try {
        arrayRealVector0 = new ArrayRealVector((double[]) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.linear.ArrayRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, true);
      assertEquals(0.0, arrayRealVector0.getLInfNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = null;
      try {
        arrayRealVector0 = new ArrayRealVector(doubleArray0, (-251), (-251));
        fail("Expecting exception: NegativeArraySizeException");
      
      } catch(NegativeArraySizeException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.linear.ArrayRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      double[] doubleArray0 = new double[0];
      ArrayRealVector arrayRealVector0 = null;
      try {
        arrayRealVector0 = new ArrayRealVector(doubleArray0, 25, 25);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // position 25 and size 25 dont fit to the size of the input array {2}
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Double[] doubleArray0 = new Double[2];
      ArrayRealVector arrayRealVector0 = null;
      try {
        arrayRealVector0 = new ArrayRealVector(doubleArray0, (-46), 1);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -46
         //
         verifyException("org.apache.commons.math.linear.ArrayRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Double[] doubleArray0 = new Double[1];
      ArrayRealVector arrayRealVector0 = null;
      try {
        arrayRealVector0 = new ArrayRealVector(doubleArray0, 194, 194);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // position 194 and size 194 dont fit to the size of the input array {2}
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Double[] doubleArray0 = new Double[20];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, 0, 0);
      assertFalse(arrayRealVector0.isNaN());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      double[] doubleArray0 = new double[0];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      ArrayRealVector arrayRealVector1 = new ArrayRealVector(arrayRealVector0, false);
      assertEquals(0.0, arrayRealVector1.getNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      ArrayRealVector arrayRealVector1 = new ArrayRealVector((RealVector) arrayRealVector0, arrayRealVector0);
      assertEquals(4, arrayRealVector1.getDimension());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      openMapRealVector0.mapAcosToSelf();
      RealVector realVector0 = arrayRealVector0.subtract((RealVector) openMapRealVector0);
      assertEquals(3.141592653589793, realVector0.getLInfNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.subtract((RealVector) arrayRealVector0);
      assertFalse(realVector0.isNaN());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapAdd(3217.569191767337);
      assertEquals(0.0, arrayRealVector0.getLInfNorm(), 0.01);
      assertEquals(6435.138383534674, realVector0.getL1Norm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Double[] doubleArray0 = new Double[2];
      Double double0 = new Double(6.283185307179586);
      doubleArray0[0] = double0;
      doubleArray0[1] = doubleArray0[0];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapSubtract(0.0);
      assertEquals(8.885765876316732, realVector0.getNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      RealVector realVector0 = openMapRealVector0.projection((RealVector) arrayRealVector0);
      assertFalse(realVector0.equals((Object)arrayRealVector0));
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapPow(1541.533876499077);
      assertEquals(0.0, realVector0.getLInfNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapExp();
      assertFalse(realVector0.equals((Object)arrayRealVector0));
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapExpm1();
      assertEquals(2, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapLog();
      assertFalse(realVector0.equals((Object)arrayRealVector0));
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapLog10();
      assertFalse(realVector0.equals((Object)arrayRealVector0));
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapLog1p();
      assertEquals(0.0, realVector0.getLInfNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(384);
      RealVector realVector0 = arrayRealVector0.mapSinh();
      assertEquals(384, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapTanh();
      assertNotSame(realVector0, arrayRealVector0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(381, 381);
      RealVector realVector0 = arrayRealVector0.mapCos();
      assertEquals(9.382602725516435E116, arrayRealVector0.getLInfNorm(), 0.01);
      assertEquals(12.626912722368392, realVector0.getNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapSin();
      assertEquals(0.0, realVector0.getNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapTanToSelf();
      assertFalse(realVector0.isInfinite());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(419, 419);
      RealVector realVector0 = arrayRealVector0.mapAcos();
      boolean boolean0 = realVector0.isInfinite();
      assertFalse(boolean0);
      assertTrue(realVector0.isNaN());
      assertEquals(8576.716096502203, arrayRealVector0.getNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapAtan();
      assertEquals(0.0, realVector0.getL1Norm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      double[] doubleArray0 = new double[10];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapInv();
      assertFalse(arrayRealVector0.isInfinite());
      assertEquals(Double.POSITIVE_INFINITY, realVector0.getL1Norm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      double[] doubleArray0 = new double[12];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapAbs();
      assertEquals(12, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapSqrt();
      assertNotSame(arrayRealVector0, realVector0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      double[] doubleArray0 = new double[12];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapCbrt();
      assertEquals(12, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapCeil();
      assertTrue(realVector0.equals((Object)arrayRealVector0));
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(381, 381);
      RealVector realVector0 = arrayRealVector0.mapFloor();
      assertEquals(7436.823313754335, realVector0.getNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapRint();
      assertEquals(0.0, realVector0.getLInfNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(417, 417);
      RealVector realVector0 = arrayRealVector0.mapSignum();
      assertEquals(1.6923032801030364E125, realVector0.getLInfNorm(), 0.01);
      assertFalse(realVector0.equals((Object)arrayRealVector0));
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(417, 417);
      RealVector realVector0 = arrayRealVector0.mapUlp();
      assertEquals(2.3703705664956942E-11, realVector0.getL1Norm(), 0.01);
      assertEquals(8515.380966228111, arrayRealVector0.getNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.ebeMultiply((RealVector) openMapRealVector0);
      assertFalse(realVector0.isNaN());
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.ebeDivide((RealVector) openMapRealVector0);
      assertEquals(Double.NaN, realVector0.getL1Norm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      try { 
        arrayRealVector0.ebeDivide((RealVector) arrayRealVector0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // vector must have at least one element
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      openMapRealVector0.mapAcosToSelf();
      double double0 = arrayRealVector0.dotProduct((RealVector) openMapRealVector0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Double[] doubleArray0 = new Double[2];
      Double double0 = new Double((-140.20358092837));
      doubleArray0[0] = double0;
      doubleArray0[1] = doubleArray0[0];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      double double1 = arrayRealVector0.getL1Norm();
      assertEquals(280.40716185674, double1, 0.01);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      double double0 = arrayRealVector0.getLInfNorm();
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      double double0 = arrayRealVector0.getDistance((RealVector) openMapRealVector0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      double double0 = arrayRealVector0.getL1Distance((RealVector) openMapRealVector0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      double double0 = arrayRealVector0.getL1Distance(doubleArray0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      double double0 = arrayRealVector0.getLInfDistance((RealVector) openMapRealVector0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      double double0 = arrayRealVector0.getLInfDistance(doubleArray0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(1756);
      try { 
        arrayRealVector0.unitVector();
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // zero norm
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(381, 381);
      arrayRealVector0.unitize();
      assertEquals(19.519221295943176, arrayRealVector0.getL1Norm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      try { 
        arrayRealVector0.unitize();
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // cannot normalize a zero norm vector
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      RealMatrix realMatrix0 = arrayRealVector0.outerProduct((RealVector) openMapRealVector0);
      assertEquals(2, realMatrix0.getColumnDimension());
  }

  @Test(timeout = 4000)
  public void test78()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      arrayRealVector0.setSubVector((-634), (RealVector) openMapRealVector0);
      assertEquals(0.0, arrayRealVector0.getNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      double[] doubleArray0 = new double[0];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(3045);
      // Undeclared exception!
      try { 
        arrayRealVector0.setSubVector(3045, (RealVector) openMapRealVector0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // index 3,045 out of allowed range [0, -1]
         //
         verifyException("org.apache.commons.math.linear.AbstractRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
      Double[] doubleArray0 = new Double[2];
      Double double0 = new Double(6.283185307179586);
      doubleArray0[0] = double0;
      doubleArray0[1] = doubleArray0[0];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      arrayRealVector0.hashCode();
      assertFalse(arrayRealVector0.isNaN());
      assertEquals(12.566370614359172, arrayRealVector0.getL1Norm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test81()  throws Throwable  {
      Double[] doubleArray0 = new Double[2];
      Double double0 = new Double(6.283185307179586);
      doubleArray0[0] = double0;
      doubleArray0[1] = doubleArray0[0];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 6.283185307179586);
      RealVector realVector0 = openMapRealVector0.mapAsin();
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(realVector0);
      arrayRealVector0.hashCode();
      assertTrue(arrayRealVector0.isNaN());
  }

  @Test(timeout = 4000)
  public void test82()  throws Throwable  {
      Double[] doubleArray0 = new Double[2];
      Double double0 = new Double(6.283185307179586);
      doubleArray0[0] = double0;
      doubleArray0[1] = doubleArray0[0];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      boolean boolean0 = arrayRealVector0.isInfinite();
      assertFalse(arrayRealVector0.isNaN());
      assertFalse(boolean0);
      assertEquals(12.566370614359172, arrayRealVector0.getL1Norm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test83()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(5, Double.NEGATIVE_INFINITY);
      boolean boolean0 = arrayRealVector0.isInfinite();
      assertTrue(boolean0);
      assertEquals(5, arrayRealVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test84()  throws Throwable  {
      Double[] doubleArray0 = new Double[2];
      Double double0 = new Double(6.283185307179586);
      doubleArray0[0] = double0;
      doubleArray0[1] = doubleArray0[0];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 6.283185307179586);
      boolean boolean0 = arrayRealVector0.equals(openMapRealVector0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test85()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      boolean boolean0 = arrayRealVector0.equals(arrayRealVector0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test86()  throws Throwable  {
      Double[] doubleArray0 = new Double[2];
      Double double0 = new Double(6.283185307179586);
      doubleArray0[0] = double0;
      doubleArray0[1] = doubleArray0[0];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      boolean boolean0 = arrayRealVector0.equals((Object) null);
      assertFalse(boolean0);
      assertEquals(8.885765876316732, arrayRealVector0.getNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test87()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      boolean boolean0 = arrayRealVector0.equals("org.apache.commons.math.MathRuntimeException$3");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test88()  throws Throwable  {
      Double[] doubleArray0 = new Double[2];
      Double double0 = new Double(6.283185307179586);
      doubleArray0[0] = double0;
      doubleArray0[1] = doubleArray0[0];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      ArrayRealVector arrayRealVector1 = new ArrayRealVector(arrayRealVector0, arrayRealVector0);
      boolean boolean0 = arrayRealVector1.equals(arrayRealVector0);
      assertEquals(12.566370614359172, arrayRealVector1.getNorm(), 0.01);
      assertFalse(arrayRealVector0.equals((Object)arrayRealVector1));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test89()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      doubleArray0[0] = (-17.11631545000808);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapAsin();
      boolean boolean0 = arrayRealVector0.equals(realVector0);
      assertTrue(realVector0.isNaN());
      assertFalse(realVector0.equals((Object)arrayRealVector0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test90()  throws Throwable  {
      Double[] doubleArray0 = new Double[2];
      Double double0 = new Double(6.283185307179586);
      doubleArray0[0] = double0;
      doubleArray0[1] = doubleArray0[0];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 6.283185307179586);
      RealVector realVector0 = openMapRealVector0.mapAtan();
      boolean boolean0 = arrayRealVector0.equals(realVector0);
      assertEquals(12.566370614359172, arrayRealVector0.getL1Norm(), 0.01);
      assertFalse(boolean0);
  }
}