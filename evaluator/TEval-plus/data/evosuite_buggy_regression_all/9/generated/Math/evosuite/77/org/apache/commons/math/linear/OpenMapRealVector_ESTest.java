/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:47:02 GMT 2023
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
public class OpenMapRealVector_ESTest extends OpenMapRealVector_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      double[] doubleArray0 = new double[34];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2.8640018862484546));
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.append((-2.8640018862484546));
      assertEquals(34, openMapRealVector0.getDimension());
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(35, openMapRealVector1.getDimension());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1802.06452518));
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.copy();
      boolean boolean0 = openMapRealVector1.equals(openMapRealVector0);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      double[] doubleArray0 = new double[34];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1644.9084650662);
      // Undeclared exception!
      try { 
        openMapRealVector0.unitVector();
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // cannot normalize a zero norm vector
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      double[] doubleArray0 = new double[18];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2630.98));
      double[] doubleArray1 = openMapRealVector0.toArray();
      assertEquals(18, doubleArray1.length);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(31);
      double double0 = openMapRealVector0.getSparcity();
      assertEquals(31, openMapRealVector0.getDimension());
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1802.06452518));
      openMapRealVector0.setSubVector(0, (RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1899.46));
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(openMapRealVector0);
      RealVector realVector0 = openMapRealVector0.add((RealVector) arrayRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(0.0, realVector0.getNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector((-559));
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
  public void test08()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1802.06452518));
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector2 = openMapRealVector0.add(openMapRealVector1);
      assertTrue(openMapRealVector2.equals((Object)openMapRealVector0));
      assertNotSame(openMapRealVector2, openMapRealVector0);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2630.98));
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(341, 1.0E-12);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.append((RealVector) arrayRealVector0);
      assertNotSame(openMapRealVector1, openMapRealVector0);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      double[] doubleArray0 = new double[18];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2630.98));
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.append((RealVector) openMapRealVector0);
      assertNotSame(openMapRealVector1, openMapRealVector0);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Double[] doubleArray0 = new Double[6];
      Double double0 = new Double((-1.0));
      doubleArray0[0] = double0;
      doubleArray0[1] = doubleArray0[0];
      Double double1 = new Double(0.0);
      doubleArray0[2] = double1;
      doubleArray0[3] = double1;
      doubleArray0[4] = doubleArray0[0];
      Double double2 = new Double(0.0);
      doubleArray0[5] = double2;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      double[] doubleArray1 = new double[6];
      doubleArray1[1] = 1.0E-12;
      doubleArray1[2] = 1204.66400923;
      doubleArray1[3] = (double) doubleArray0[1];
      doubleArray1[4] = (double) doubleArray0[0];
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.projection(doubleArray1);
      assertEquals(0.6666666666666666, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(0.5, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1180.680140535224);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      try { 
        openMapRealVector0.dotProduct((RealVector) arrayRealVector0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // vector length mismatch: got 1 but expected 0
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      double[] doubleArray0 = new double[15];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1.9238066916169958));
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.ebeDivide((RealVector) openMapRealVector0);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertNotSame(openMapRealVector1, openMapRealVector0);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      double[] doubleArray0 = new double[34];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2.8640018862484546));
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.ebeDivide(doubleArray0);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertNotSame(openMapRealVector1, openMapRealVector0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2572.597754));
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, false);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.ebeMultiply((RealVector) arrayRealVector0);
      assertNotSame(openMapRealVector1, openMapRealVector0);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
      assertTrue(openMapRealVector1.equals((Object)openMapRealVector0));
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      double[] doubleArray0 = new double[15];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2.01438918350801));
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.ebeMultiply(doubleArray0);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
      assertNotSame(openMapRealVector1, openMapRealVector0);
      assertTrue(openMapRealVector1.equals((Object)openMapRealVector0));
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(40, 40, Double.NEGATIVE_INFINITY);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAdd(300.7779);
      OpenMapRealVector openMapRealVector2 = openMapRealVector1.getSubVector(2, 2);
      assertEquals(2, openMapRealVector2.getDimension());
      assertEquals(1.0, openMapRealVector2.getSparcity(), 0.01);
      assertEquals(40, openMapRealVector1.getDimension());
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      double[] doubleArray0 = new double[15];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-3578.4199));
      Double[] doubleArray1 = new Double[9];
      doubleArray1[0] = (Double) 1.0E-12;
      doubleArray1[1] = (Double) 1.0E-12;
      doubleArray1[2] = (Double) 1.0E-12;
      doubleArray1[3] = (Double) 1.0E-12;
      doubleArray1[4] = (Double) 1.0E-12;
      doubleArray1[5] = (Double) 1.0E-12;
      doubleArray1[6] = (Double) 1.0E-12;
      doubleArray1[7] = (Double) 1.0E-12;
      doubleArray1[8] = (Double) 1.0E-12;
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray1, (double) doubleArray1[3]);
      double double0 = openMapRealVector1.getDistance(openMapRealVector0);
      assertEquals(2.9999999999999997E-12, double0, 0.01);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 2251.55);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      double double0 = openMapRealVector0.getDistance((RealVector) arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(0.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      double[] doubleArray0 = new double[19];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-43.81));
      double double0 = openMapRealVector0.getDistance((RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      double[] doubleArray0 = new double[34];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1.0));
      double double0 = openMapRealVector0.getL1Distance((RealVector) openMapRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      double[] doubleArray0 = new double[15];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-147.62679609555));
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray0, 1.0E-12);
      double double0 = openMapRealVector1.getL1Distance(openMapRealVector0);
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1802.06452518));
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, false);
      double double0 = openMapRealVector0.getL1Distance((RealVector) arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      double[] doubleArray0 = new double[34];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2630.98));
      double double0 = openMapRealVector0.getLInfNorm();
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1802.06452518));
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.copy();
      assertTrue(openMapRealVector1.equals((Object)openMapRealVector0));
      
      RealVector realVector0 = openMapRealVector1.mapLog10ToSelf();
      double double0 = realVector0.getLInfDistance((RealVector) openMapRealVector0);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1802.06452518));
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray0);
      double double0 = openMapRealVector1.getLInfDistance((RealVector) openMapRealVector0);
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2630.98));
      RealVector realVector0 = openMapRealVector0.mapExp();
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray0);
      double double0 = openMapRealVector1.getLInfDistance(realVector0);
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(1.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = (-1736.611527624544);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 2251.55);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      double double0 = openMapRealVector0.getLInfDistance((RealVector) arrayRealVector0);
      assertEquals(1736.611527624544, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1802.06452518));
      boolean boolean0 = openMapRealVector0.isInfinite();
      assertFalse(boolean0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1131.912928));
      openMapRealVector0.unitize();
      boolean boolean0 = openMapRealVector0.isInfinite();
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1802.06452518));
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      
      openMapRealVector0.mapLog10ToSelf();
      boolean boolean0 = openMapRealVector0.isInfinite();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1802.06452518));
      boolean boolean0 = openMapRealVector0.isNaN();
      assertFalse(boolean0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      doubleArray0[0] = (-2572.597754);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1.0));
      openMapRealVector0.mapAsinToSelf();
      boolean boolean0 = openMapRealVector0.isNaN();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      double[] doubleArray0 = new double[15];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-29.395987302950793));
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      assertEquals(15, realMatrix0.getRowDimension());
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(15, realMatrix0.getColumnDimension());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1802.06452518));
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(1, 1);
      assertEquals(1, openMapRealVector1.getDimension());
      
      OpenMapRealVector openMapRealVector2 = openMapRealVector1.subtract(openMapRealVector0);
      assertNotSame(openMapRealVector2, openMapRealVector1);
      assertEquals(0.0, openMapRealVector2.getSparcity(), 0.01);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = (-3.792526593021523);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.subtract((RealVector) arrayRealVector0);
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(0.5, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2630.98));
      openMapRealVector0.set((-802.288996915455));
      assertEquals(1, openMapRealVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1.0));
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.subtract((RealVector) openMapRealVector0);
      assertTrue(openMapRealVector1.equals((Object)openMapRealVector0));
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
      assertNotSame(openMapRealVector1, openMapRealVector0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      double[] doubleArray0 = new double[15];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2.01438918350801));
      openMapRealVector0.hashCode();
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1802.06452518));
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector0);
      assertTrue(boolean0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1802.06452518));
      boolean boolean0 = openMapRealVector0.equals((Object) null);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1802.06452518));
      Object object0 = new Object();
      boolean boolean0 = openMapRealVector0.equals(object0);
      assertFalse(boolean0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1802.06452518));
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector();
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector1);
      assertFalse(boolean0);
      assertEquals(Double.NaN, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1802.06452518));
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(1, 1);
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector1);
      assertFalse(boolean0);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      double[] doubleArray0 = new double[34];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1.0));
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      
      openMapRealVector0.mapAcosToSelf();
      OpenMapRealVector openMapRealVector1 = (OpenMapRealVector)openMapRealVector0.add((RealVector) openMapRealVector0);
      boolean boolean0 = openMapRealVector1.equals(openMapRealVector0);
      assertFalse(boolean0);
  }
}
