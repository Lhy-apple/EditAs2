/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:29:26 GMT 2023
 */

package org.apache.commons.math.linear;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.analysis.polynomials.PolynomialFunction;
import org.apache.commons.math.linear.ArrayRealVector;
import org.apache.commons.math.linear.OpenMapRealVector;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.linear.RealVector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class OpenMapRealVector_ESTest extends OpenMapRealVector_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 5.415076100864309);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.append(1.0);
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(3, openMapRealVector1.getDimension());
      assertEquals(2, openMapRealVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 832.678);
      OpenMapRealVector openMapRealVector1 = (OpenMapRealVector)openMapRealVector0.projection((RealVector) openMapRealVector0);
      boolean boolean0 = openMapRealVector1.isInfinite();
      assertFalse(boolean0);
      assertEquals(0.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(2, openMapRealVector1.getDimension());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Double[] doubleArray0 = new Double[2];
      Double double0 = new Double(0.0);
      doubleArray0[0] = double0;
      doubleArray0[1] = doubleArray0[0];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.unitVector();
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(2, openMapRealVector1.getDimension());
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertNotSame(openMapRealVector1, openMapRealVector0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Double[] doubleArray0 = new Double[3];
      Double double0 = new Double(1.0E-12);
      doubleArray0[0] = double0;
      doubleArray0[1] = double0;
      doubleArray0[2] = double0;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      double[] doubleArray1 = openMapRealVector0.toArray();
      assertArrayEquals(new double[] {1.0E-12, 1.0E-12, 1.0E-12}, doubleArray1, 0.01);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      double[] doubleArray0 = new double[8];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-707.19));
      double double0 = openMapRealVector0.getSparcity();
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 767.2);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAdd((-3103.817188276015));
      RealVector realVector0 = openMapRealVector0.projection((RealVector) openMapRealVector1);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(0.0, openMapRealVector0.getSparcity(), 0.01);
      assertFalse(realVector0.equals((Object)openMapRealVector1));
      assertFalse(realVector0.equals((Object)openMapRealVector0));
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Double[] doubleArray0 = new Double[1];
      Double double0 = new Double(1.0);
      doubleArray0[0] = double0;
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      try { 
        openMapRealVector0.setSubVector((-795), (RealVector) arrayRealVector0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // index -795 out of allowed range [0, 0]
         //
         verifyException("org.apache.commons.math.linear.AbstractRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(0, 0);
      assertEquals(Double.NaN, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      double[] doubleArray0 = new double[1];
      try { 
        openMapRealVector0.projection(doubleArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // vector length mismatch: got 0 but expected 1
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      doubleArray0[2] = 2702.16;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 767.2);
      OpenMapRealVector openMapRealVector1 = (OpenMapRealVector)openMapRealVector0.projection((RealVector) openMapRealVector0);
      assertEquals(0.25, openMapRealVector1.getSparcity(), 0.01);
      assertTrue(openMapRealVector1.equals((Object)openMapRealVector0));
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(2);
      OpenMapRealVector.OpenMapSparseIterator openMapRealVector_OpenMapSparseIterator0 = openMapRealVector0.new OpenMapSparseIterator();
      // Undeclared exception!
      try { 
        openMapRealVector_OpenMapSparseIterator0.remove();
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Not supported
         //
         verifyException("org.apache.commons.math.linear.OpenMapRealVector$OpenMapSparseIterator", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Double[] doubleArray0 = new Double[9];
      doubleArray0[0] = (Double) 0.0;
      OpenMapRealVector openMapRealVector0 = null;
      try {
        openMapRealVector0 = new OpenMapRealVector(doubleArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.linear.OpenMapRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(1386, 1.0E-12);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(1386);
      RealVector realVector0 = openMapRealVector0.add((RealVector) arrayRealVector0);
      assertFalse(realVector0.isNaN());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 3091.0992541047467);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector((RealVector) openMapRealVector0);
      OpenMapRealVector openMapRealVector2 = openMapRealVector1.mapAdd((-717.3494107));
      OpenMapRealVector openMapRealVector3 = openMapRealVector2.add(openMapRealVector0);
      assertTrue(openMapRealVector3.equals((Object)openMapRealVector2));
      assertEquals(1.0, openMapRealVector3.getSparcity(), 0.01);
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      doubleArray0[2] = 2702.16;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 767.2);
      OpenMapRealVector openMapRealVector1 = (OpenMapRealVector)openMapRealVector0.add((RealVector) openMapRealVector0);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertNotSame(openMapRealVector1, openMapRealVector0);
      assertEquals(0.25, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(0.25, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      doubleArray0[2] = 2702.16;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 767.2);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAdd((-3103.817188276015));
      OpenMapRealVector openMapRealVector2 = (OpenMapRealVector)openMapRealVector0.add((RealVector) openMapRealVector1);
      assertEquals(0.25, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(0.75, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(1.0, openMapRealVector2.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = 3101.4262731;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 3101.4262731);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.append((RealVector) openMapRealVector0);
      OpenMapRealVector openMapRealVector2 = openMapRealVector1.getSubVector(1, 1);
      assertEquals(2, openMapRealVector1.getDimension());
      assertEquals(1.0, openMapRealVector2.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Double[] doubleArray0 = new Double[1];
      Double double0 = new Double(1.0);
      doubleArray0[0] = double0;
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.append((RealVector) arrayRealVector0);
      assertEquals(2, openMapRealVector1.getDimension());
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1.0));
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(0);
      ArrayRealVector arrayRealVector1 = new ArrayRealVector(openMapRealVector0, arrayRealVector0);
      RealVector realVector0 = openMapRealVector0.projection((RealVector) arrayRealVector1);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertTrue(realVector0.isNaN());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      doubleArray0[0] = (-1251.7345844459182);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.ebeDivide((RealVector) openMapRealVector0);
      assertEquals(0.3333333333333333, openMapRealVector1.getSparcity(), 0.01);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertNotSame(openMapRealVector1, openMapRealVector0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = 3101.4262731;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 3101.4262731);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.ebeDivide(doubleArray0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1698.0665));
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.ebeMultiply(doubleArray0);
      assertNotSame(openMapRealVector1, openMapRealVector0);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
      assertTrue(openMapRealVector1.equals((Object)openMapRealVector0));
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.getSubVector(1, 1);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(1, openMapRealVector1.getDimension());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-6090.042));
      double double0 = openMapRealVector0.getDistance((RealVector) openMapRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(2);
      double[] doubleArray0 = new double[6];
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray0, (-1698.0665));
      double double0 = openMapRealVector0.getDistance(openMapRealVector1);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(openMapRealVector0);
      double double0 = openMapRealVector0.getDistance((RealVector) arrayRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(0.0, arrayRealVector0.getL1Norm(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      double[] doubleArray0 = new double[4];
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray0, 0.0);
      double double0 = openMapRealVector0.getL1Distance(openMapRealVector1);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      double[] doubleArray0 = new double[0];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      double double0 = openMapRealVector0.getL1Distance((RealVector) arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Double[] doubleArray0 = new Double[1];
      Double double0 = new Double(1.0);
      doubleArray0[0] = double0;
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      double double1 = openMapRealVector0.getL1Distance((RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(0.0, double1, 0.01);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      double[] doubleArray0 = new double[12];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, true);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      double double0 = openMapRealVector0.getL1Distance(doubleArray0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(0.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Double[] doubleArray0 = new Double[1];
      Double double0 = new Double(1.0);
      doubleArray0[0] = double0;
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      double double1 = openMapRealVector0.getLInfNorm();
      assertEquals(1.0, double1, 0.01);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[1] = (-6090.042);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 832.678);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.subtract(openMapRealVector0);
      double double0 = openMapRealVector0.getLInfDistance((RealVector) openMapRealVector1);
      assertEquals(6090.042, double0, 0.01);
      assertEquals(0.5, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      doubleArray0[0] = 3101.4262731;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 3101.4262731);
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      OpenMapRealVector openMapRealVector1 = (OpenMapRealVector)openMapRealVector0.map(polynomialFunction0);
      double double0 = openMapRealVector0.getLInfDistance((RealVector) openMapRealVector1);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(0.16666666666666666, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(3101.4262731, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Double[] doubleArray0 = new Double[1];
      Double double0 = new Double(1.0);
      doubleArray0[0] = double0;
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      double double1 = openMapRealVector0.getLInfDistance((RealVector) arrayRealVector0);
      assertEquals(0.0, double1, 0.01);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = (double) (-5);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 810.1832401053499);
      double double0 = openMapRealVector0.getLInfDistance(doubleArray0);
      assertEquals(5.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[1] = (-6090.042);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 832.678);
      boolean boolean0 = openMapRealVector0.isInfinite();
      assertEquals(0.5, openMapRealVector0.getSparcity(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 284.2531909);
      assertEquals(0.0, openMapRealVector0.getSparcity(), 0.01);
      
      openMapRealVector0.mapLog10ToSelf();
      boolean boolean0 = openMapRealVector0.isInfinite();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[1] = (-6090.042);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 832.678);
      boolean boolean0 = openMapRealVector0.isNaN();
      assertFalse(boolean0);
      assertEquals(0.5, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 832.678);
      RealVector realVector0 = openMapRealVector0.projection((RealVector) openMapRealVector0);
      boolean boolean0 = realVector0.isNaN();
      assertTrue(boolean0);
      assertEquals(0.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      Double[] doubleArray1 = new Double[5];
      doubleArray1[0] = (Double) 0.0;
      doubleArray1[1] = (Double) 0.0;
      doubleArray1[2] = (Double) 0.0;
      doubleArray1[3] = (Double) 0.0;
      doubleArray1[4] = (Double) 0.0;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray1, (double) doubleArray1[2]);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(5, realMatrix0.getRowDimension());
      assertEquals(5, realMatrix0.getColumnDimension());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(43);
      openMapRealVector0.setSubVector(1, doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Double[] doubleArray0 = new Double[1];
      Double double0 = new Double(1.0);
      doubleArray0[0] = double0;
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      openMapRealVector0.set((double) doubleArray0[0]);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[1] = (-6090.042);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 832.678);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.subtract(doubleArray0);
      OpenMapRealVector openMapRealVector2 = openMapRealVector1.subtract(openMapRealVector0);
      assertEquals(0.5, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(0.5, openMapRealVector2.getSparcity(), 0.01);
      assertNotSame(openMapRealVector2, openMapRealVector0);
      assertFalse(openMapRealVector2.equals((Object)openMapRealVector0));
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Double[] doubleArray0 = new Double[1];
      Double double0 = new Double(1.0);
      doubleArray0[0] = double0;
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.subtract((RealVector) arrayRealVector0);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.subtract((RealVector) openMapRealVector0);
      assertTrue(openMapRealVector1.equals((Object)openMapRealVector0));
      assertNotSame(openMapRealVector1, openMapRealVector0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      // Undeclared exception!
      try { 
        openMapRealVector0.unitize();
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // cannot normalize a zero norm vector
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      doubleArray0[0] = 3101.4262731;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 3101.4262731);
      openMapRealVector0.hashCode();
      assertEquals(0.16666666666666666, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-6090.042));
      boolean boolean0 = openMapRealVector0.equals((Object) null);
      assertFalse(boolean0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(openMapRealVector0);
      boolean boolean0 = openMapRealVector0.equals(arrayRealVector0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(2);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector();
      boolean boolean0 = openMapRealVector1.equals(openMapRealVector0);
      assertFalse(openMapRealVector0.equals((Object)openMapRealVector1));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1794.5306);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray0, 1.0E-12);
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector1);
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
      assertFalse(boolean0);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-6090.042));
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(openMapRealVector0);
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector1);
      assertTrue(boolean0);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = (-6090.042);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-6090.042));
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.ebeMultiply((RealVector) openMapRealVector0);
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector1);
      assertFalse(boolean0);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[1] = (-6090.042);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 832.678);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.subtract(openMapRealVector0);
      boolean boolean0 = openMapRealVector1.equals(openMapRealVector0);
      assertFalse(boolean0);
      assertEquals(0.5, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
  }
}
