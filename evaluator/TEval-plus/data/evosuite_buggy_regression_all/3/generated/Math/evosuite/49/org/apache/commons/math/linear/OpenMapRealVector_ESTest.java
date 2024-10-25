/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 18:08:38 GMT 2023
 */

package org.apache.commons.math.linear;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.analysis.function.Inverse;
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
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(2923, 2923);
      openMapRealVector0.append((RealVector) openMapRealVector0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.append(1.0E-12);
      assertNotSame(openMapRealVector1, openMapRealVector0);
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      double[] doubleArray0 = new double[49];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.unitVector();
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertNotSame(openMapRealVector1, openMapRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertEquals(49, openMapRealVector1.getDimension());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      double[] doubleArray0 = new double[12];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      double[] doubleArray1 = openMapRealVector0.toArray();
      assertEquals(12, doubleArray1.length);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      double[] doubleArray0 = new double[21];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.subtract((RealVector) openMapRealVector0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      assertNotSame(openMapRealVector1, openMapRealVector0);
      assertTrue(openMapRealVector1.equals((Object)openMapRealVector0));
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(openMapRealVector0);
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector1);
      assertTrue(boolean0);
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Double[] doubleArray0 = new Double[0];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(Double.NaN, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      double[] doubleArray0 = new double[65];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAdd((-326.12609544003));
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector((-2219), (-686));
      double double0 = openMapRealVector0.getSparsity();
      assertEquals((-2219), openMapRealVector0.getDimension());
      assertEquals(-0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1702.7613074781));
      // Undeclared exception!
      try { 
        openMapRealVector0.setSubVector((-5893), (RealVector) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // index (-5,893)
         //
         verifyException("org.apache.commons.math.linear.AbstractRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(openMapRealVector0);
      double double0 = openMapRealVector0.getDistance((RealVector) arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(Double.NaN, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
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
  public void test12()  throws Throwable  {
      Double[] doubleArray0 = new Double[1];
      doubleArray0[0] = (Double) 1.0E-12;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1.0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Double[] doubleArray0 = new Double[1];
      doubleArray0[0] = (Double) 1.0E-12;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1.0E-12);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      double[] doubleArray0 = new double[21];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector((RealVector) openMapRealVector0);
      double double0 = openMapRealVector1.getLInfDistance((RealVector) openMapRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertEquals(0.0, openMapRealVector1.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1702.7613074781));
      openMapRealVector0.unitize();
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector((RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      double[] doubleArray0 = new double[21];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(openMapRealVector0);
      RealVector realVector0 = openMapRealVector0.add((RealVector) arrayRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertTrue(realVector0.equals((Object)arrayRealVector0));
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      double[] doubleArray0 = new double[25];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      OpenMapRealVector openMapRealVector1 = (OpenMapRealVector)openMapRealVector0.add((RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
      assertTrue(openMapRealVector1.equals((Object)openMapRealVector0));
      assertNotSame(openMapRealVector1, openMapRealVector0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      double[] doubleArray0 = new double[25];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector((RealVector) openMapRealVector0);
      RealVector realVector0 = openMapRealVector0.add((RealVector) openMapRealVector1);
      assertTrue(realVector0.equals((Object)openMapRealVector0));
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertNotSame(realVector0, openMapRealVector0);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      double[] doubleArray0 = new double[13];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.append(openMapRealVector0);
      assertEquals(26, openMapRealVector1.getDimension());
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      double[] doubleArray0 = new double[21];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      // Undeclared exception!
      try { 
        openMapRealVector0.append((RealVector) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.linear.OpenMapRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      double[] doubleArray0 = new double[65];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.append(doubleArray0);
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
      assertNotSame(openMapRealVector1, openMapRealVector0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(1.0E-12);
      OpenMapRealVector openMapRealVector2 = openMapRealVector1.subtract(openMapRealVector0);
      RealVector realVector0 = openMapRealVector2.projection((RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
      assertTrue(realVector0.equals((Object)openMapRealVector2));
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      double[] doubleArray0 = new double[21];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      openMapRealVector0.dotProduct(doubleArray0);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      double[] doubleArray0 = new double[21];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.ebeDivide((RealVector) openMapRealVector0);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertNotSame(openMapRealVector1, openMapRealVector0);
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      double[] doubleArray0 = new double[8];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.ebeDivide(doubleArray0);
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertNotSame(openMapRealVector1, openMapRealVector0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      double[] doubleArray0 = new double[21];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.ebeMultiply((RealVector) openMapRealVector0);
      assertNotSame(openMapRealVector1, openMapRealVector0);
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
      assertTrue(openMapRealVector1.equals((Object)openMapRealVector0));
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      double[] doubleArray0 = new double[21];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.ebeMultiply(doubleArray0);
      assertTrue(openMapRealVector1.equals((Object)openMapRealVector0));
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
      assertNotSame(openMapRealVector1, openMapRealVector0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      double[] doubleArray0 = new double[21];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.getSubVector(19, 1);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertEquals(0.0, openMapRealVector1.getSparsity(), 0.01);
      assertEquals(1, openMapRealVector1.getDimension());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      double[] doubleArray0 = new double[19];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      double double0 = openMapRealVector0.getDistance((RealVector) openMapRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      double[] doubleArray0 = new double[19];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector((RealVector) openMapRealVector0);
      double double0 = openMapRealVector1.getDistance((RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertEquals(0.0, openMapRealVector1.getSparsity(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      double[] doubleArray0 = new double[19];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      double double0 = openMapRealVector0.getDistance(doubleArray0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      double[] doubleArray0 = new double[21];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      double double0 = openMapRealVector0.getL1Distance((RealVector) openMapRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(1.0E-12);
      OpenMapRealVector openMapRealVector2 = openMapRealVector1.subtract(openMapRealVector0);
      double double0 = openMapRealVector2.getL1Distance(openMapRealVector1);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertEquals(2.0E-12, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      assertEquals(Double.NaN, openMapRealVector0.getSparsity(), 0.01);
      
      Double[] doubleArray0 = new Double[0];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      double double0 = openMapRealVector0.getL1Distance((RealVector) arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      double[] doubleArray0 = new double[21];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      double double0 = openMapRealVector0.getL1Distance(doubleArray0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      double[] doubleArray0 = new double[21];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      double double0 = openMapRealVector0.getLInfDistance((RealVector) openMapRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = 744.7152735936;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 744.7152735936);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(openMapRealVector0);
      assertTrue(openMapRealVector1.equals((Object)openMapRealVector0));
      
      openMapRealVector1.mapMultiplyToSelf(2065.83579);
      double double0 = openMapRealVector1.getLInfDistance((RealVector) openMapRealVector0);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertEquals(1537714.7502757073, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      doubleArray0[0] = 1787.6464967;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1012.386);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.subtract(doubleArray0);
      double double0 = openMapRealVector1.getLInfDistance((RealVector) openMapRealVector0);
      assertEquals(1787.6464967, double0, 0.01);
      assertEquals(0.0, openMapRealVector1.getSparsity(), 0.01);
      assertEquals(0.3333333333333333, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      Double[] doubleArray0 = new Double[0];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      double double0 = openMapRealVector0.getLInfDistance((RealVector) arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      openMapRealVector0.mapAddToSelf(1.0E-12);
      double double0 = openMapRealVector0.getLInfDistance(doubleArray0);
      assertEquals(1.0E-12, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      double[] doubleArray0 = new double[21];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      boolean boolean0 = openMapRealVector0.isInfinite();
      assertFalse(boolean0);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      double[] doubleArray0 = new double[21];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      openMapRealVector0.unitize();
      boolean boolean0 = openMapRealVector0.isInfinite();
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      double[] doubleArray0 = new double[21];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1758.65427798248);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      Inverse inverse0 = new Inverse();
      openMapRealVector0.mapToSelf(inverse0);
      boolean boolean0 = openMapRealVector0.isInfinite();
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      boolean boolean0 = openMapRealVector0.isNaN();
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1702.7613074781));
      openMapRealVector0.unitize();
      boolean boolean0 = openMapRealVector0.isNaN();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      double[] doubleArray0 = new double[21];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      assertEquals(21, realMatrix0.getColumnDimension());
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertEquals(21, realMatrix0.getRowDimension());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(1);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(openMapRealVector0, 715827883);
      openMapRealVector1.setSubVector(1191, doubleArray0);
      assertEquals(715827884, openMapRealVector1.getDimension());
      assertEquals(0.0, openMapRealVector1.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      double[] doubleArray0 = new double[21];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      openMapRealVector0.set((-2637.86106850386));
      assertEquals(21, openMapRealVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(4055.606);
      OpenMapRealVector openMapRealVector2 = openMapRealVector0.subtract(openMapRealVector1);
      OpenMapRealVector openMapRealVector3 = openMapRealVector2.subtract(openMapRealVector1);
      assertEquals(1.0, openMapRealVector3.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(openMapRealVector0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.subtract((RealVector) arrayRealVector0);
      assertNotSame(openMapRealVector1, openMapRealVector0);
      assertTrue(openMapRealVector1.equals((Object)openMapRealVector0));
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      // Undeclared exception!
      try { 
        openMapRealVector0.unitize();
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // zero norm
         //
         verifyException("org.apache.commons.math.linear.OpenMapRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1702.7613074781));
      openMapRealVector0.hashCode();
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector0);
      assertTrue(boolean0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1012.386);
      boolean boolean0 = openMapRealVector0.equals((Object) null);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(2132973623, 2132973623);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(openMapRealVector0, 1);
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector1);
      assertEquals(2132973624, openMapRealVector1.getDimension());
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      Double[] doubleArray0 = new Double[0];
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray0, 1916.19556497884);
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector1);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(1309.8295528822);
      OpenMapRealVector openMapRealVector2 = openMapRealVector0.projection(doubleArray0);
      boolean boolean0 = openMapRealVector1.equals(openMapRealVector2);
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = (double) 1196;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.subtract(openMapRealVector0);
      boolean boolean0 = openMapRealVector1.equals(openMapRealVector0);
      assertFalse(boolean0);
      assertEquals(0.0, openMapRealVector1.getSparsity(), 0.01);
  }
}
