/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:30:09 GMT 2023
 */

package org.apache.commons.math3.linear;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.OpenMapRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class OpenMapRealVector_ESTest extends OpenMapRealVector_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      // Undeclared exception!
      try { 
        openMapRealVector0.dotProduct((RealVector) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math3.linear.RealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAdd(1.0E-12);
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.append(1.0E-12);
      assertEquals(0.14285714285714285, openMapRealVector1.getSparsity(), 0.01);
      assertEquals(6, openMapRealVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      double double0 = openMapRealVector0.getSparsity();
      assertEquals(6, openMapRealVector0.getDimension());
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(4094, 4094);
      // Undeclared exception!
      try { 
        openMapRealVector0.getSubVector(1627, (-1001));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // number of elements should be positive (-1,001)
         //
         verifyException("org.apache.commons.math3.linear.OpenMapRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      // Undeclared exception!
      try { 
        openMapRealVector0.unitVector();
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // zero norm
         //
         verifyException("org.apache.commons.math3.linear.OpenMapRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(1002);
      RealVector realVector0 = openMapRealVector0.projection(openMapRealVector0);
      boolean boolean0 = realVector0.isNaN();
      assertEquals(1002, openMapRealVector0.getDimension());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      double[] doubleArray0 = new double[12];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(1.0E-12);
      openMapRealVector1.outerProduct(openMapRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(1.0E-12);
      RealVector realVector0 = openMapRealVector1.mapMultiply(1.0E-12);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertFalse(realVector0.equals((Object)openMapRealVector0));
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      double[] doubleArray0 = new double[8];
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
         verifyException("org.apache.commons.math3.linear.OpenMapRealVector$OpenMapSparseIterator", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Double[] doubleArray0 = new Double[2];
      doubleArray0[0] = (Double) 1.0E-12;
      OpenMapRealVector openMapRealVector0 = null;
      try {
        openMapRealVector0 = new OpenMapRealVector(doubleArray0, 3271.0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math3.linear.OpenMapRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(9, openMapRealVector0.getDimension());
      
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      openMapRealVector0.add((RealVector) arrayRealVector0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(1.0E-12);
      OpenMapRealVector openMapRealVector2 = openMapRealVector0.subtract(openMapRealVector1);
      openMapRealVector1.add((RealVector) openMapRealVector2);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertEquals(0.0, openMapRealVector2.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      double[] doubleArray0 = new double[14];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(1.0E-12);
      openMapRealVector1.append((RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, false);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.append((RealVector) arrayRealVector0);
      assertEquals(0.0, openMapRealVector1.getSparsity(), 0.01);
      assertEquals(12, openMapRealVector1.getDimension());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      double[] doubleArray0 = new double[12];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray0, (-1774.2690505));
      double double0 = openMapRealVector0.dotProduct((RealVector) openMapRealVector1);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(1.0E-12);
      double double0 = openMapRealVector0.dotProduct(openMapRealVector1);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertEquals(1.0E-24, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(1.0E-12);
      OpenMapRealVector openMapRealVector2 = openMapRealVector1.ebeDivide(openMapRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertEquals(1.0, openMapRealVector2.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(1.0E-12);
      OpenMapRealVector openMapRealVector2 = openMapRealVector1.ebeMultiply(openMapRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertEquals(0.0, openMapRealVector2.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Double double0 = new Double(733.71893083);
      Double[] doubleArray0 = new Double[9];
      doubleArray0[0] = double0;
      doubleArray0[1] = double0;
      doubleArray0[2] = double0;
      doubleArray0[3] = double0;
      doubleArray0[4] = double0;
      doubleArray0[5] = double0;
      doubleArray0[6] = double0;
      doubleArray0[7] = double0;
      doubleArray0[8] = double0;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.getSubVector(4, 4);
      assertEquals(4, openMapRealVector1.getDimension());
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      double[] doubleArray0 = new double[12];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(1.0E-12);
      double double0 = openMapRealVector1.getDistance((RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      double[] doubleArray0 = new double[12];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray0, (-1168.62988));
      double double0 = openMapRealVector0.getDistance((RealVector) openMapRealVector1);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      double[] doubleArray0 = new double[0];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1146.1368267158);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      double double0 = openMapRealVector0.getDistance((RealVector) arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      double[] doubleArray0 = new double[12];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(1.0E-12);
      double double0 = openMapRealVector1.getL1Distance((RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(openMapRealVector0);
      openMapRealVector1.mapAddToSelf(1.0E-12);
      double double0 = openMapRealVector0.getL1Distance(openMapRealVector1);
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertEquals(2.0E-12, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      double[] doubleArray0 = new double[8];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      RealVector realVector0 = RealVector.unmodifiableRealVector(openMapRealVector0);
      openMapRealVector0.getL1Distance(realVector0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(1.0E-12);
      double double0 = openMapRealVector0.getLInfDistance(openMapRealVector1);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(1.0E-12);
      OpenMapRealVector openMapRealVector2 = openMapRealVector0.add(openMapRealVector1);
      double double0 = openMapRealVector2.getLInfDistance(openMapRealVector1);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertEquals(1.0E-12, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      double[] doubleArray0 = new double[12];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray0, (-1168.62988));
      double double0 = openMapRealVector0.getLInfDistance(openMapRealVector1);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      openMapRealVector0.mapAddToSelf(1.0E-12);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray0);
      double double0 = openMapRealVector1.getLInfDistance(openMapRealVector0);
      assertEquals(0.0, openMapRealVector1.getSparsity(), 0.01);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertEquals(1.0E-12, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      double[] doubleArray0 = new double[12];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      double double0 = openMapRealVector0.getLInfDistance(arrayRealVector0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf((-1095.4285));
      boolean boolean0 = openMapRealVector1.isInfinite();
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Double[] doubleArray0 = new Double[3];
      Double double0 = new Double(11950.3368697197);
      doubleArray0[0] = double0;
      Double double1 = new Double(0.0);
      doubleArray0[1] = double1;
      doubleArray0[2] = double1;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (double) doubleArray0[2]);
      openMapRealVector0.mapDivideToSelf((double) doubleArray0[2]);
      boolean boolean0 = openMapRealVector0.isInfinite();
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(1.0E-12);
      boolean boolean0 = openMapRealVector1.isNaN();
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      double[] doubleArray0 = new double[8];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      openMapRealVector0.setSubVector(0, openMapRealVector0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      double[] doubleArray0 = new double[12];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      openMapRealVector0.set((-1817.011));
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      Double[] doubleArray1 = new Double[1];
      doubleArray1[0] = (Double) 1.0E-12;
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray1, 0.0);
      OpenMapRealVector openMapRealVector2 = openMapRealVector0.subtract(openMapRealVector1);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      assertEquals(1.0, openMapRealVector2.getSparsity(), 0.01);
      assertNotSame(openMapRealVector2, openMapRealVector1);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      double[] doubleArray0 = new double[11];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0);
      openMapRealVector0.subtract((RealVector) arrayRealVector0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      double[] doubleArray0 = new double[19];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      RealVector realVector0 = openMapRealVector0.subtract((RealVector) openMapRealVector0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      assertTrue(realVector0.equals((Object)openMapRealVector0));
      assertNotSame(realVector0, openMapRealVector0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(1.0E-12);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector2 = openMapRealVector0.unitVector();
      assertFalse(openMapRealVector2.equals((Object)openMapRealVector1));
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      double[] doubleArray0 = new double[10];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(1.0E-12);
      double[] doubleArray1 = openMapRealVector1.toArray();
      assertArrayEquals(new double[] {1.0E-12, 1.0E-12, 1.0E-12, 1.0E-12, 1.0E-12, 1.0E-12, 1.0E-12, 1.0E-12, 1.0E-12, 1.0E-12}, doubleArray1, 0.01);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      double[] doubleArray0 = new double[12];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(1.0E-12);
      openMapRealVector1.hashCode();
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      double[] doubleArray0 = new double[8];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector((RealVector) openMapRealVector0);
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector1);
      assertTrue(boolean0);
      assertEquals(0.0, openMapRealVector1.getSparsity(), 0.01);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector0);
      assertTrue(boolean0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      Object object0 = new Object();
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1078.52918));
      boolean boolean0 = openMapRealVector0.equals(object0);
      assertEquals(1.0, openMapRealVector0.getSparsity(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector((-1801), (-1801));
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector1);
      assertFalse(boolean0);
      assertEquals((-1801), openMapRealVector1.getDimension());
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray0, (-1095.4285));
      boolean boolean0 = openMapRealVector1.equals(openMapRealVector0);
      assertFalse(boolean0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      assertEquals(1.0, openMapRealVector1.getSparsity(), 0.01);
      assertFalse(openMapRealVector0.equals((Object)openMapRealVector1));
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      double[] doubleArray0 = new double[8];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(703.19);
      OpenMapRealVector openMapRealVector2 = new OpenMapRealVector((RealVector) openMapRealVector1);
      boolean boolean0 = openMapRealVector1.equals(openMapRealVector2);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      double[] doubleArray0 = new double[14];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparsity(), 0.01);
      
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAddToSelf(1.0E-12);
      OpenMapRealVector openMapRealVector2 = new OpenMapRealVector(doubleArray0);
      boolean boolean0 = openMapRealVector1.equals(openMapRealVector2);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(1347);
      RealVector realVector0 = openMapRealVector0.projection(openMapRealVector0);
      boolean boolean0 = openMapRealVector0.equals(realVector0);
      assertFalse(boolean0);
      assertEquals(1347, realVector0.getDimension());
  }
}
