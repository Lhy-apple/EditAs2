/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:28:58 GMT 2023
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
      double[] doubleArray0 = new double[3];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.subtract((RealVector) arrayRealVector0);
      assertEquals(6, openMapRealVector1.getDimension());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(6, 21);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(openMapRealVector0);
      String string0 = arrayRealVector0.toString();
      assertEquals("{0; 0; 0; 0; 0; 0}", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      double double0 = arrayRealVector0.getL1Distance((RealVector) arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(8, arrayRealVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      double[] doubleArray0 = new double[10];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, true);
      RealVector realVector0 = arrayRealVector0.projection(doubleArray0);
      assertTrue(realVector0.isNaN());
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      ArrayRealVector arrayRealVector1 = arrayRealVector0.append(arrayRealVector0);
      assertEquals(12, arrayRealVector1.getDimension());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Double[] doubleArray0 = new Double[0];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      double double0 = arrayRealVector0.getDistance((RealVector) arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      double double0 = arrayRealVector0.getLInfDistance(arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(8, arrayRealVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.ebeMultiply((RealVector) arrayRealVector0);
      assertEquals(8, realVector0.getDimension());
      assertEquals(0.0, realVector0.getL1Norm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      RealVector realVector0 = openMapRealVector0.add((RealVector) arrayRealVector0);
      assertEquals(8, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(604, 604);
      ArrayRealVector arrayRealVector1 = new ArrayRealVector(arrayRealVector0, arrayRealVector0);
      try { 
        arrayRealVector1.dotProduct((RealVector) arrayRealVector0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // vector length mismatch: got 1,208 but expected 604
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      ArrayRealVector arrayRealVector1 = new ArrayRealVector(doubleArray0, arrayRealVector0);
      assertEquals(12, arrayRealVector1.getDimension());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.append((-133.2186666));
      assertEquals(133.2186666, realVector0.getLInfNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      ArrayRealVector arrayRealVector1 = new ArrayRealVector(arrayRealVector0);
      assertEquals(8, arrayRealVector1.getDimension());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      // Undeclared exception!
      try { 
        arrayRealVector0.setSubVector(3, (RealVector) arrayRealVector0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // index 3 out of allowed range [0, -1]
         //
         verifyException("org.apache.commons.math.linear.AbstractRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.ebeDivide((RealVector) arrayRealVector0);
      assertEquals(8, realVector0.getDimension());
      assertEquals(Double.NaN, realVector0.getL1Norm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        arrayRealVector0.getSubVector(286, 286);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // index 286 out of allowed range [0, 3]
         //
         verifyException("org.apache.commons.math.linear.AbstractRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.append(doubleArray0);
      assertEquals(12, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      OpenMapRealVector openMapRealVector1 = (OpenMapRealVector)arrayRealVector0.projection((RealVector) openMapRealVector0);
      assertEquals(Double.NaN, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      // Undeclared exception!
      try { 
        arrayRealVector0.setEntry((-59), (-59));
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // index -59 out of allowed range [0, -1]
         //
         verifyException("org.apache.commons.math.linear.AbstractRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.append((RealVector) arrayRealVector0);
      assertEquals(16, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.add((RealVector) arrayRealVector0);
      assertEquals(0.0, realVector0.getL1Norm(), 0.01);
      assertEquals(8, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.subtract((RealVector) arrayRealVector0);
      assertEquals(0.0, realVector0.getNorm(), 0.01);
      assertEquals(8, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      arrayRealVector0.set(1.0E-12);
      assertEquals(8.0E-12, arrayRealVector0.getLInfNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
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
  public void test25()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      try { 
        arrayRealVector0.add((RealVector) openMapRealVector0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // vector must have at least one element
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = null;
      try {
        arrayRealVector0 = new ArrayRealVector(doubleArray0, 3, (-1725));
        fail("Expecting exception: NegativeArraySizeException");
      
      } catch(NegativeArraySizeException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.linear.ArrayRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = null;
      try {
        arrayRealVector0 = new ArrayRealVector(doubleArray0, 1775, 1775);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // position 1,775 and size 1,775 dont fit to the size of the input array {2}
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Double[] doubleArray0 = new Double[1];
      ArrayRealVector arrayRealVector0 = null;
      try {
        arrayRealVector0 = new ArrayRealVector(doubleArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.linear.ArrayRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Double[] doubleArray0 = new Double[3];
      ArrayRealVector arrayRealVector0 = null;
      try {
        arrayRealVector0 = new ArrayRealVector(doubleArray0, 3387, 3387);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // position 3,387 and size 3,387 dont fit to the size of the input array {2}
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Double[] doubleArray0 = new Double[5];
      doubleArray0[1] = (Double) 1.0E-12;
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, 1, 1);
      assertEquals(1.0E-12, arrayRealVector0.getL1Norm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      ArrayRealVector arrayRealVector1 = new ArrayRealVector(arrayRealVector0, false);
      assertEquals(0, arrayRealVector1.getDimension());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      double[] doubleArray0 = new double[14];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      ArrayRealVector arrayRealVector1 = new ArrayRealVector((RealVector) arrayRealVector0, arrayRealVector0);
      assertEquals(56, arrayRealVector1.getDimension());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      openMapRealVector0.mapLog10ToSelf();
      RealVector realVector0 = arrayRealVector0.add((RealVector) openMapRealVector0);
      assertEquals(8, realVector0.getDimension());
      assertEquals(Double.POSITIVE_INFINITY, realVector0.getLInfNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      openMapRealVector0.mapLog10ToSelf();
      RealVector realVector0 = arrayRealVector0.subtract((RealVector) openMapRealVector0);
      boolean boolean0 = realVector0.isInfinite();
      assertTrue(boolean0);
      assertEquals(8, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapAdd((-66.6093333));
      assertEquals(188.39964506697964, realVector0.getNorm(), 0.01);
      assertEquals(0.0, arrayRealVector0.getLInfNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapSubtract(1.0E-12);
      assertEquals(8.0E-12, realVector0.getL1Norm(), 0.01);
      assertEquals(0.0, arrayRealVector0.getNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      double[] doubleArray0 = new double[10];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, true);
      RealVector realVector0 = arrayRealVector0.mapPow((-1194.19186512));
      assertFalse(realVector0.equals((Object)arrayRealVector0));
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapExp();
      assertEquals(0.0, arrayRealVector0.getL1Norm(), 0.01);
      assertEquals(8.0, realVector0.getL1Norm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapExpm1();
      assertEquals(8, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapLog();
      assertFalse(realVector0.equals((Object)arrayRealVector0));
      assertEquals(8, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapLog10();
      assertEquals(4, realVector0.getDimension());
      assertFalse(realVector0.equals((Object)arrayRealVector0));
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapCosh();
      assertEquals(8.0, realVector0.getL1Norm(), 0.01);
      assertEquals(0.0, arrayRealVector0.getLInfNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapSinh();
      assertEquals(8, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapTanh();
      assertEquals(12, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapCos();
      assertFalse(realVector0.equals((Object)arrayRealVector0));
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapSin();
      assertEquals(4, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapTan();
      assertEquals(8, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(3794);
      RealVector realVector0 = arrayRealVector0.mapAcos();
      assertEquals(96.75391348381304, realVector0.getNorm(), 0.01);
      assertEquals(0.0, arrayRealVector0.getLInfNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapAtan();
      assertEquals(4, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapInv();
      assertEquals(8, realVector0.getDimension());
      assertTrue(realVector0.isInfinite());
      assertEquals(0.0, arrayRealVector0.getNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapAbs();
      assertEquals(4, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapSqrt();
      assertEquals(8, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapCbrt();
      assertEquals(4, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapCeil();
      assertEquals(8, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapFloor();
      assertEquals(4, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapRint();
      assertEquals(8, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapSignum();
      assertEquals(8, realVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, false);
      RealVector realVector0 = arrayRealVector0.mapUlp();
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0}, doubleArray0, 0.01);
      assertEquals(2.0E-323, realVector0.getL1Norm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      RealVector realVector0 = arrayRealVector0.ebeMultiply((RealVector) openMapRealVector0);
      assertEquals(4, realVector0.getDimension());
      assertFalse(realVector0.isNaN());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      RealVector realVector0 = arrayRealVector0.ebeDivide((RealVector) openMapRealVector0);
      assertEquals(8, realVector0.getDimension());
      assertTrue(realVector0.isNaN());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      double double0 = arrayRealVector0.dotProduct((RealVector) arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(8, arrayRealVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      openMapRealVector0.mapLog10ToSelf();
      double double0 = arrayRealVector0.dotProduct((RealVector) openMapRealVector0);
      assertEquals(Double.NaN, double0, 0.01);
      assertEquals(8, openMapRealVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(604, 604);
      arrayRealVector0.unitize();
      assertEquals(1.3507311517786376E180, arrayRealVector0.getLInfNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, false);
      double double0 = arrayRealVector0.getL1Norm();
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(1311);
      double double0 = arrayRealVector0.getLInfNorm();
      assertEquals(1311, arrayRealVector0.getDimension());
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      double[] doubleArray0 = new double[10];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      double double0 = arrayRealVector0.getDistance((RealVector) openMapRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(20, openMapRealVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      double double0 = arrayRealVector0.getDistance(doubleArray0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      double double0 = arrayRealVector0.getL1Distance((RealVector) openMapRealVector0);
      assertEquals(8, openMapRealVector0.getDimension());
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      double double0 = arrayRealVector0.getLInfDistance((RealVector) openMapRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(4, openMapRealVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      double double0 = arrayRealVector0.getLInfDistance((RealVector) arrayRealVector0);
      assertEquals(0, arrayRealVector0.getDimension());
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      doubleArray0[1] = (-66.6093333);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.unitVector();
      assertEquals(94.19982253348984, arrayRealVector0.getNorm(), 0.01);
      assertEquals(2896.3093757400984, realVector0.getLInfNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
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
  public void test73()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
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
  public void test74()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      RealMatrix realMatrix0 = arrayRealVector0.outerProduct((RealVector) openMapRealVector0);
      assertEquals(8, realMatrix0.getColumnDimension());
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealMatrix realMatrix0 = arrayRealVector0.outerProduct((RealVector) arrayRealVector0);
      assertEquals(8, realMatrix0.getColumnDimension());
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      arrayRealVector0.setSubVector(10, (RealVector) openMapRealVector0);
      assertEquals(0, arrayRealVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      // Undeclared exception!
      try { 
        arrayRealVector0.setSubVector((-3225), (RealVector) openMapRealVector0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // index -3,225 out of allowed range [0, 7]
         //
         verifyException("org.apache.commons.math.linear.AbstractRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test78()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      boolean boolean0 = arrayRealVector0.isInfinite();
      assertFalse(boolean0);
      assertEquals(8, arrayRealVector0.getDimension());
      assertFalse(arrayRealVector0.isNaN());
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      doubleArray0[1] = (-66.6093333);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapAsin();
      boolean boolean0 = realVector0.isInfinite();
      assertFalse(boolean0);
      assertTrue(realVector0.isNaN());
      assertEquals(4262.9973312, arrayRealVector0.getLInfNorm(), 0.01);
      assertFalse(arrayRealVector0.isNaN());
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, true);
      boolean boolean0 = arrayRealVector0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test81()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(6, 21);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(openMapRealVector0);
      boolean boolean0 = arrayRealVector0.equals(arrayRealVector0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test82()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      Object object0 = new Object();
      boolean boolean0 = arrayRealVector0.equals(object0);
      assertFalse(boolean0);
      assertEquals(0, arrayRealVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test83()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      boolean boolean0 = arrayRealVector0.equals(openMapRealVector0);
      assertFalse(boolean0);
      assertEquals(4, arrayRealVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test84()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      doubleArray0[1] = (-66.6093333);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      RealVector realVector0 = arrayRealVector0.mapLog1p();
      boolean boolean0 = arrayRealVector0.equals(realVector0);
      assertEquals(1065.7493328, arrayRealVector0.getLInfNorm(), 0.01);
      assertFalse(realVector0.equals((Object)arrayRealVector0));
      assertTrue(realVector0.isNaN());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test85()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      boolean boolean0 = arrayRealVector0.equals(openMapRealVector0);
      assertTrue(boolean0);
      assertEquals(4, openMapRealVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test86()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[1] = (-66.6093333);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      openMapRealVector0.mapFloorToSelf();
      boolean boolean0 = arrayRealVector0.equals(openMapRealVector0);
      assertEquals(0.5, openMapRealVector0.getSparcity(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test87()  throws Throwable  {
      Double[] doubleArray0 = new Double[0];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      arrayRealVector0.hashCode();
      assertFalse(arrayRealVector0.isNaN());
  }

  @Test(timeout = 4000)
  public void test88()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      doubleArray0[1] = (-66.6093333);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      ArrayRealVector arrayRealVector1 = (ArrayRealVector)arrayRealVector0.mapAsin();
      arrayRealVector1.hashCode();
      assertEquals(4262.9973312, arrayRealVector0.getLInfNorm(), 0.01);
      assertFalse(arrayRealVector0.isNaN());
      assertTrue(arrayRealVector1.isNaN());
  }
}