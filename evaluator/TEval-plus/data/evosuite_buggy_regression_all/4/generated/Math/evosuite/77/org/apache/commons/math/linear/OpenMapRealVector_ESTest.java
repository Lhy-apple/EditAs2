/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:13:16 GMT 2023
 */

package org.apache.commons.math.linear;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.linear.ArrayRealVector;
import org.apache.commons.math.linear.OpenMapRealVector;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.linear.RealVector;
import org.apache.commons.math.util.OpenIntToDoubleHashMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class OpenMapRealVector_ESTest extends OpenMapRealVector_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(1762, (-1269));
      assertEquals(1762, openMapRealVector0.getDimension());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.append(1.0E-12);
      assertEquals(Double.NaN, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(6);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.add(openMapRealVector0);
      openMapRealVector0.set((-2715.297));
      openMapRealVector1.subtract((RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertNotSame(openMapRealVector0, openMapRealVector1);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2324.1180398127));
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.unitVector();
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertNotSame(openMapRealVector1, openMapRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      double double0 = openMapRealVector0.getSparcity();
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2324.1180398127));
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.mapAdd(0.0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
      assertTrue(openMapRealVector1.equals((Object)openMapRealVector0));
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Double double0 = new Double((-3573));
      Double[] doubleArray0 = new Double[1];
      doubleArray0[0] = double0;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      try { 
        openMapRealVector0.setSubVector((-3573), (RealVector) openMapRealVector0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // index -3,573 out of allowed range [0, 0]
         //
         verifyException("org.apache.commons.math.linear.AbstractRealVector", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(0, 0);
      assertEquals(0, openMapRealVector0.getDimension());
      
      double[] doubleArray0 = new double[0];
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.projection(doubleArray0);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2324.1180398127));
      RealVector realVector0 = openMapRealVector0.add(doubleArray0);
      assertEquals(0.0, realVector0.getLInfNorm(), 0.01);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Double[] doubleArray0 = new Double[4];
      Double double0 = new Double(0.0);
      doubleArray0[0] = double0;
      doubleArray0[1] = doubleArray0[0];
      doubleArray0[2] = doubleArray0[0];
      doubleArray0[3] = doubleArray0[0];
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
  public void test10()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = (-2324.1180398127);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2324.1180398127));
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector((RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Double[] doubleArray0 = new Double[0];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2651.672581717253));
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, (-1190), 0);
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
  public void test12()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      RealVector realVector0 = openMapRealVector0.add((RealVector) openMapRealVector0);
      assertNotSame(realVector0, openMapRealVector0);
      assertTrue(realVector0.equals((Object)openMapRealVector0));
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Double[] doubleArray0 = new Double[4];
      Double double0 = new Double(0.0);
      doubleArray0[0] = double0;
      doubleArray0[1] = doubleArray0[0];
      doubleArray0[2] = doubleArray0[0];
      doubleArray0[3] = double0;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      double[] doubleArray1 = new double[4];
      doubleArray1[0] = 1.0E-12;
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray1);
      OpenMapRealVector openMapRealVector2 = openMapRealVector1.add(openMapRealVector0);
      assertTrue(openMapRealVector2.equals((Object)openMapRealVector1));
      assertEquals(0.25, openMapRealVector2.getSparcity(), 0.01);
      assertNotSame(openMapRealVector2, openMapRealVector1);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2324.1180398127));
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.add(openMapRealVector0);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
      assertTrue(openMapRealVector1.equals((Object)openMapRealVector0));
      assertNotSame(openMapRealVector1, openMapRealVector0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Double[] doubleArray0 = new Double[4];
      Double double0 = new Double(0.0);
      doubleArray0[0] = double0;
      Double double1 = new Double(1303.767);
      doubleArray0[1] = doubleArray0[0];
      doubleArray0[2] = double1;
      doubleArray0[3] = doubleArray0[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      double[] doubleArray1 = new double[4];
      doubleArray1[0] = 1.0E-12;
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray1);
      OpenMapRealVector openMapRealVector2 = openMapRealVector1.add(openMapRealVector0);
      assertFalse(openMapRealVector2.equals((Object)openMapRealVector0));
      assertEquals(0.75, openMapRealVector2.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
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
  public void test17()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.append((RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(8, openMapRealVector1.getDimension());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-3864.812217989));
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.append(doubleArray0);
      assertEquals(8, openMapRealVector1.getDimension());
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Double[] doubleArray0 = new Double[4];
      Double double0 = new Double(0.0);
      doubleArray0[0] = double0;
      Double double1 = new Double(1303.767);
      doubleArray0[1] = doubleArray0[0];
      doubleArray0[2] = double1;
      doubleArray0[3] = double1;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.subtract((RealVector) openMapRealVector0);
      OpenMapRealVector openMapRealVector2 = (OpenMapRealVector)openMapRealVector1.projection((RealVector) openMapRealVector0);
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(0.5, openMapRealVector2.getSparcity(), 0.01);
      assertTrue(openMapRealVector2.equals((Object)openMapRealVector1));
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector((-1903), 2324.1180398127);
      try { 
        openMapRealVector0.dotProduct(doubleArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // vector length mismatch: got -1,903 but expected 1
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Double[] doubleArray0 = new Double[4];
      Double double0 = new Double(0.0);
      doubleArray0[0] = double0;
      Double double1 = new Double(1310.1734305092612);
      doubleArray0[1] = double0;
      doubleArray0[2] = double1;
      doubleArray0[3] = double0;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.ebeDivide((RealVector) openMapRealVector0);
      assertNotSame(openMapRealVector1, openMapRealVector0);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertEquals(0.25, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = (-2324.1180398127);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1414.0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.ebeDivide(doubleArray0);
      assertEquals(0.5, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = (-2324.1180398127);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1414.0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.ebeMultiply((RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
      assertNotSame(openMapRealVector1, openMapRealVector0);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = (-2324.1180398127);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1414.0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.ebeMultiply(doubleArray0);
      assertNotSame(openMapRealVector1, openMapRealVector0);
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.getSubVector(1, 1);
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(1, openMapRealVector1.getDimension());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Double double0 = new Double((-3573));
      Double[] doubleArray0 = new Double[1];
      doubleArray0[0] = double0;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      double[] doubleArray1 = openMapRealVector0.toArray();
      assertArrayEquals(new double[] {(-3573.0)}, doubleArray1, 0.01);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = (-2324.1180398127);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1414.0);
      double double0 = openMapRealVector0.getDistance((RealVector) openMapRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1414.0);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector((RealVector) openMapRealVector0);
      OpenMapRealVector openMapRealVector2 = (OpenMapRealVector)openMapRealVector1.mapCos();
      double double0 = openMapRealVector0.getDistance(openMapRealVector2);
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
      assertEquals(1.4142135623730951, double0, 0.01);
      assertEquals(0.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(47);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(arrayRealVector0);
      double double0 = openMapRealVector0.getDistance((RealVector) arrayRealVector0);
      assertEquals(0.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2324.1180398127));
      double double0 = openMapRealVector0.getL1Distance((RealVector) openMapRealVector0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      Double[] doubleArray0 = new Double[4];
      doubleArray0[0] = (Double) 1.0E-12;
      doubleArray0[1] = (Double) 1.0E-12;
      doubleArray0[2] = (Double) 1.0E-12;
      doubleArray0[3] = (Double) openMapRealVector0.DEFAULT_ZERO_TOLERANCE;
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray0);
      double double0 = openMapRealVector0.getL1Distance(openMapRealVector1);
      assertEquals(4.0E-12, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(0, 0);
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      double double0 = openMapRealVector0.getL1Distance((RealVector) arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-3573.0));
      double double0 = openMapRealVector0.getL1Distance(doubleArray0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = (-2324.1180398127);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1414.0);
      double double0 = openMapRealVector0.getLInfNorm();
      assertEquals((-2324.1180398127), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2324.1180398127));
      double double0 = openMapRealVector0.getLInfDistance((RealVector) openMapRealVector0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2324.1180398127));
      RealVector realVector0 = openMapRealVector0.mapCos();
      double double0 = realVector0.getLInfDistance((RealVector) openMapRealVector0);
      assertEquals(1.0, double0, 0.01);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(openMapRealVector0);
      OpenIntToDoubleHashMap.Iterator openIntToDoubleHashMap_Iterator0 = mock(OpenIntToDoubleHashMap.Iterator.class, new ViolatedAssumptionAnswer());
      doReturn(0).when(openIntToDoubleHashMap_Iterator0).key();
      OpenMapRealVector.OpenMapEntry openMapRealVector_OpenMapEntry0 = openMapRealVector1.new OpenMapEntry(openIntToDoubleHashMap_Iterator0);
      assertTrue(openMapRealVector1.equals((Object)openMapRealVector0));
      
      openMapRealVector_OpenMapEntry0.setValue(1.0E-12);
      double double0 = openMapRealVector0.getLInfDistance((RealVector) openMapRealVector1);
      assertEquals(1.0E-12, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector();
      OpenIntToDoubleHashMap.Iterator openIntToDoubleHashMap_Iterator0 = mock(OpenIntToDoubleHashMap.Iterator.class, new ViolatedAssumptionAnswer());
      doReturn((-1)).when(openIntToDoubleHashMap_Iterator0).key();
      OpenMapRealVector.OpenMapEntry openMapRealVector_OpenMapEntry0 = openMapRealVector0.new OpenMapEntry(openIntToDoubleHashMap_Iterator0);
      openMapRealVector_OpenMapEntry0.setValue(0.0);
      double double0 = openMapRealVector1.getLInfDistance((RealVector) openMapRealVector0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      double double0 = openMapRealVector0.getLInfDistance((RealVector) arrayRealVector0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1414.0);
      double double0 = openMapRealVector0.getLInfDistance(doubleArray0);
      assertEquals(0.0, double0, 0.01);
      assertEquals(0.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = (-2324.1180398127);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1414.0);
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      
      openMapRealVector0.unitize();
      double double0 = openMapRealVector0.getLInfDistance(doubleArray0);
      assertEquals(2323.1180398127, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Double[] doubleArray0 = new Double[4];
      Double double0 = new Double(0.0);
      doubleArray0[0] = double0;
      Double double1 = new Double(1303.767);
      doubleArray0[1] = doubleArray0[0];
      doubleArray0[2] = double1;
      doubleArray0[3] = double1;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      boolean boolean0 = openMapRealVector0.isInfinite();
      assertEquals(0.5, openMapRealVector0.getSparcity(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, Double.NEGATIVE_INFINITY);
      openMapRealVector0.unitize();
      boolean boolean0 = openMapRealVector0.isInfinite();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      doubleArray0[4] = Double.NEGATIVE_INFINITY;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, Double.NEGATIVE_INFINITY);
      boolean boolean0 = openMapRealVector0.isInfinite();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector((-3573), (-3573), (-1899.017963836));
      boolean boolean0 = openMapRealVector0.isNaN();
      assertEquals((-3573), openMapRealVector0.getDimension());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      doubleArray0[4] = (-1.0);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      boolean boolean0 = openMapRealVector0.isNaN();
      assertFalse(boolean0);
      assertEquals(0.14285714285714285, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Double double0 = new Double(3588.753524221921);
      Double[] doubleArray0 = new Double[5];
      doubleArray0[0] = (Double) (-2324.1180398127);
      doubleArray0[1] = double0;
      doubleArray0[2] = (Double) (-2324.1180398127);
      doubleArray0[3] = (Double) 0.0;
      doubleArray0[4] = doubleArray0[0];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0);
      openMapRealVector0.mapSqrtToSelf();
      boolean boolean0 = openMapRealVector0.isNaN();
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = (-2324.1180398127);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1414.0);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      assertEquals(0.5, openMapRealVector0.getSparcity(), 0.01);
      assertEquals(2, realMatrix0.getRowDimension());
      assertEquals(2, realMatrix0.getColumnDimension());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      openMapRealVector0.setSubVector(0, doubleArray0);
      assertEquals(0.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(6);
      openMapRealVector0.set((-2715.297));
      assertEquals(1.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      double[] doubleArray0 = new double[8];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2324.622243628538));
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(openMapRealVector0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.subtract((RealVector) arrayRealVector0);
      assertEquals(1.0, openMapRealVector1.getSparcity(), 0.01);
      assertTrue(openMapRealVector1.equals((Object)openMapRealVector0));
      assertNotSame(openMapRealVector1, openMapRealVector0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1414.0);
      OpenMapRealVector openMapRealVector1 = openMapRealVector0.subtract(doubleArray0);
      assertNotSame(openMapRealVector1, openMapRealVector0);
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
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
  public void test54()  throws Throwable  {
      double[] doubleArray0 = new double[8];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2324.622243628538));
      openMapRealVector0.hashCode();
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      boolean boolean0 = openMapRealVector0.equals("cannot normalize a zero norm vector");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2875.81209373));
      boolean boolean0 = openMapRealVector0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(openMapRealVector0, 1457);
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector1);
      assertEquals(1457, openMapRealVector1.getDimension());
      assertFalse(openMapRealVector1.equals((Object)openMapRealVector0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1414.0);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(doubleArray0, (-2324.1180398127));
      boolean boolean0 = openMapRealVector1.equals(openMapRealVector0);
      assertFalse(openMapRealVector0.equals((Object)openMapRealVector1));
      assertFalse(boolean0);
      assertEquals(0.0, openMapRealVector0.getSparcity(), 0.01);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = (-2324.1180398127);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1414.0);
      RealVector realVector0 = openMapRealVector0.mapTan();
      boolean boolean0 = openMapRealVector0.equals(realVector0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = (-2324.1180398127);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      OpenMapRealVector openMapRealVector1 = new OpenMapRealVector(openMapRealVector0);
      boolean boolean0 = openMapRealVector0.equals(openMapRealVector1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = (-2324.1180398127);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1414.0);
      OpenMapRealVector openMapRealVector1 = (OpenMapRealVector)openMapRealVector0.mapCos();
      OpenMapRealVector openMapRealVector2 = (OpenMapRealVector)openMapRealVector0.mapTan();
      boolean boolean0 = openMapRealVector1.equals(openMapRealVector2);
      assertEquals(0.0, openMapRealVector1.getSparcity(), 0.01);
      assertFalse(boolean0);
      assertFalse(openMapRealVector2.equals((Object)openMapRealVector1));
  }
}
