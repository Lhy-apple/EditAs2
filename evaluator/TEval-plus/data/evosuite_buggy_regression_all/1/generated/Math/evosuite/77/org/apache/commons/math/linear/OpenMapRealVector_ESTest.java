/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:08:34 GMT 2023
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
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32, 32);
      try { 
        openMapRealVector0.setSubVector(32, (RealVector) openMapRealVector0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // index 32 out of allowed range [0, 31]
         //
         verifyException("org.apache.commons.math.linear.AbstractRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(2131);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.append(3522.0);
      assertEquals(2131, openMapRealVector0.getDimension());
      assertEquals(4.6904315196998124E-4, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(1, 1);
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
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(31, 0, 31);
      double double0 = openMapRealVector0.getSparcity();
      assertEquals(31, openMapRealVector0.getDimension());
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(1, 1);
      openMapRealVector0.mapLog10ToSelf();
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.subtract((RealVector) openMapRealVector0);
      boolean boolean0 = openMapRealVector1.isNaN();
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAdd(1072.9488784001);
      assertEquals(0.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
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
  public void test07()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32);
      // Undeclared exception!
      try { 
        openMapRealVector0.projection((double[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.linear.OpenMapRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32);
      RealVector realVector0 = openMapRealVector0.mapLog10ToSelf();
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector();
      OpenMapRealVector openMapRealVector2 = new OpenMapRealVector(openMapRealVector1, 32);
      RealVector realVector1 = openMapRealVector2.projection(realVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertFalse(realVector1.equals((Object)openMapRealVector0));
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32, 32);
      RealVector realVector0 = openMapRealVector0.mapLog10ToSelf();
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(32, 32);
      RealVector realVector1 = realVector0.add((RealVector) arrayRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertFalse(realVector1.equals((Object)arrayRealVector0));
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = 1761.2783065;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.ebeDivide(doubleArray0);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertNotSame(openMapRealVector1, openMapRealVector0);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Double[] doubleArray0 = new Double[3];
      Double double0 = new Double((-3344.0));
      doubleArray0[0] = double0;
      Double double1 = new Double(1.0);
      doubleArray0[1] = double1;
      OpenMapRealVector openMapRealVector0 = null;
      try {
        openMapRealVector0 = new OpenMapRealVector(doubleArray0, 2134.404661);
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
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(40, 40);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector((RealVector) openMapRealVector0);
      assertTrue(openMapRealVector1.equals((Object)openMapRealVector0));
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32, 32);
      RealVector realVector0 = openMapRealVector0.mapLog10ToSelf();
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector((RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertTrue(openMapRealVector1.equals((Object)realVector0));
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32);
      openMapRealVector0.mapLog10ToSelf();
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector();
      OpenMapRealVector openMapRealVector2 = new OpenMapRealVector(openMapRealVector1, 32);
      openMapRealVector0.add(openMapRealVector2);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32, 32);
      openMapRealVector0.mapLog10ToSelf();
      openMapRealVector0.add((RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32, 32);
      OpenMapRealVector openMapRealVector1 = (OpenMapRealVector)openMapRealVector0.mapLog10ToSelf();
      OpenMapRealVector openMapRealVector2 = openMapRealVector1.append((RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertNotSame(openMapRealVector0, openMapRealVector2);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Double[] doubleArray0 = new Double[1];
      doubleArray0[0] = (Double) 2131.0;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(2121);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.append((RealVector) arrayRealVector0);
      assertEquals(4.71253534401508E-4, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1761.2783065);
      openMapRealVector0.dotProduct(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32, 32);
      RealVector realVector0 = openMapRealVector0.mapLog10ToSelf();
      openMapRealVector0.ebeDivide(realVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(16);
      RealVector realVector0 = openMapRealVector0.mapLog10ToSelf();
      openMapRealVector0.ebeMultiply(realVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(1, 492.210348);
      openMapRealVector0.mapLog10ToSelf();
      double[] doubleArray0 = new double[1];
      openMapRealVector0.ebeMultiply(doubleArray0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(2477, 2131);
      openMapRealVector0.mapCosToSelf();
      openMapRealVector0.getSubVector(11, 11);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = 1761.2783065;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      double[] doubleArray1 = openMapRealVector0.toArray();
      assertArrayEquals(new double[] {1761.2783065}, doubleArray1, 0.01);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32, 32);
      OpenMapRealVector openMapRealVector1 = (OpenMapRealVector)openMapRealVector0.mapLog10ToSelf();
      Double[] doubleArray0 = new Double[3];
      doubleArray0[0] = (Double) 1.0E-12;
      doubleArray0[1] = (Double) 1.0E-12;
      doubleArray0[2] = (Double) 1.0E-12;
      OpenMapRealVector openMapRealVector2 = new OpenMapRealVector(doubleArray0);
      double double0 = openMapRealVector2.getDistance(openMapRealVector1);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      double[] doubleArray0 = new double[0];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      double double0 = openMapRealVector0.getDistance((RealVector) arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32);
      double double0 = openMapRealVector0.getDistance((RealVector) openMapRealVector0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      double double0 = openMapRealVector0.getDistance(doubleArray0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(0.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32, 32);
      RealVector realVector0 = openMapRealVector0.mapCoshToSelf();
      double double0 = openMapRealVector0.getL1Distance(realVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      double[] doubleArray0 = new double[10];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(openMapRealVector0);
      openMapRealVector1.mapCoshToSelf();
      double double0 = openMapRealVector0.getL1Distance((RealVector) openMapRealVector1);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(10.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(openMapRealVector0);
      double double0 = openMapRealVector0.getL1Distance((RealVector) arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(0.0, arrayRealVector0.getNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32, 32);
      openMapRealVector0.mapCoshToSelf();
      double double0 = openMapRealVector0.getLInfNorm();
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(32.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32);
      RealVector realVector0 = openMapRealVector0.mapLog10ToSelf();
      double double0 = realVector0.getLInfDistance((RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(openMapRealVector0);
      openMapRealVector1.mapCosToSelf();
      double double0 = openMapRealVector1.getLInfDistance((RealVector) openMapRealVector0);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32, 32);
      RealVector realVector0 = openMapRealVector0.mapCoshToSelf();
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.subtract(realVector0);
      double double0 = openMapRealVector1.getLInfDistance(realVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(openMapRealVector0);
      double double0 = openMapRealVector0.getLInfDistance((RealVector) arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(0.0, arrayRealVector0.getLInfNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      doubleArray0[2] = 1.0;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 257.58796146719925);
      double double0 = openMapRealVector0.getLInfDistance(doubleArray0);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(40, 40);
      openMapRealVector0.mapCosToSelf();
      boolean boolean0 = openMapRealVector0.isInfinite();
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      doubleArray0[1] = Double.NaN;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      boolean boolean0 = openMapRealVector0.isInfinite();
      assertEquals(0.16666666666666666, openMapRealVector0.getSparcity(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(2131);
      RealVector realVector0 = openMapRealVector0.mapLog10ToSelf();
      boolean boolean0 = realVector0.isInfinite();
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32, 32);
      openMapRealVector0.mapCoshToSelf();
      boolean boolean0 = openMapRealVector0.isNaN();
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(7, realMatrix0.getRowDimension());
      assertEquals(7, realMatrix0.getColumnDimension());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32);
      double[] doubleArray0 = new double[7];
      openMapRealVector0.setSubVector(14, doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(2131);
      openMapRealVector0.set(1.0E-12);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(2131, 2131);
      openMapRealVector0.mapLog10ToSelf();
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(2131, Double.NaN);
      OpenMapRealVector openMapRealVector2 = openMapRealVector1.subtract((RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertFalse(openMapRealVector2.equals((Object)openMapRealVector1));
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(openMapRealVector0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.subtract((RealVector) arrayRealVector0);
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
      assertNotSame(openMapRealVector1, openMapRealVector0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32, 32);
      OpenMapRealVector openMapRealVector1 = (OpenMapRealVector)openMapRealVector0.mapLog10ToSelf();
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(32, 32);
      OpenMapRealVector openMapRealVector2 = openMapRealVector1.subtract((RealVector) arrayRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertTrue(openMapRealVector2.equals((Object)openMapRealVector1));
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(1, 1);
      OpenMapRealVector openMapRealVector1 = (OpenMapRealVector)openMapRealVector0.mapLog10ToSelf();
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      
      OpenMapRealVector openMapRealVector2 = openMapRealVector1.unitVector();
      assertFalse(openMapRealVector2.equals((Object)openMapRealVector0));
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = (double) 1;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      openMapRealVector0.hashCode();
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(1);
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector0.DEFAULT_ZERO_TOLERANCE);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(2131);
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector((-422), 243.2149053512416);
      boolean boolean0 = openMapRealVector0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(1);
      boolean boolean0 = openMapRealVector1.equals(openMapRealVector0);
      assertFalse(boolean0);
      assertEquals(0.0, openMapRealVector0.getSparcity(), 0.01);
      assertFalse(openMapRealVector0.equals((Object)openMapRealVector1));
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(1);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(1, 1);
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector1);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = (-2576.6334624);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(openMapRealVector0);
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector1);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(32);
      OpenMapRealVector openMapRealVector1 = (OpenMapRealVector)openMapRealVector0.mapLog10ToSelf();
      OpenMapRealVector openMapRealVector2 = new OpenMapRealVector(32, 1.0E-12);
      boolean boolean0 = openMapRealVector1.equals(openMapRealVector2);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(1);
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = (double) 1;
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray0);
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector1);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertFalse(boolean0);
  }
}