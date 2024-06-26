/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 18:12:53 GMT 2023
 */

package org.apache.commons.math.linear;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.linear.ArrayRealVector;
import org.apache.commons.math.linear.EigenDecompositionImpl;
import org.apache.commons.math.linear.OpenMapRealMatrix;
import org.apache.commons.math.linear.OpenMapRealVector;
import org.apache.commons.math.linear.RealMatrix;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class EigenDecompositionImpl_ESTest extends EigenDecompositionImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      doubleArray0[0] = 233.497;
      doubleArray0[1] = 233.497;
      doubleArray0[2] = 233.497;
      doubleArray0[3] = 233.497;
      doubleArray0[5] = 233.497;
      doubleArray0[6] = 233.497;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 233.497);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, (-1.0));
      eigenDecompositionImpl0.getImagEigenvalues();
      assertEquals(-0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, (-25.774355357943));
      eigenDecompositionImpl0.getImagEigenvalue(0);
      assertEquals(0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      doubleArray0[0] = 233.497;
      doubleArray0[1] = 233.497;
      doubleArray0[2] = 233.497;
      doubleArray0[3] = 233.497;
      doubleArray0[5] = 233.497;
      doubleArray0[6] = 233.497;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 233.497);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, (-1.0));
      try { 
        eigenDecompositionImpl0.getRealEigenvalue((-2144));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -2144
         //
         verifyException("org.apache.commons.math.linear.EigenDecompositionImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, (-25.774355357943));
      double[] doubleArray1 = eigenDecompositionImpl0.getRealEigenvalues();
      assertArrayEquals(new double[] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, doubleArray1, 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      doubleArray0[5] = 1309.5758208680322;
      doubleArray0[1] = 1.2E-8;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1309.5758208680322);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = null;
      try {
        eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, 1.0E-12);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // eigen decomposition of assymetric matrices not supported yet
         //
         verifyException("org.apache.commons.math.linear.EigenDecompositionImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-1.979714354613011));
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, (-1.979714354613011));
      eigenDecompositionImpl0.getV();
      eigenDecompositionImpl0.getV();
      assertEquals(0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 2.2250738585072014E-308);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, 1.0E-12);
      eigenDecompositionImpl0.getVT();
      RealMatrix realMatrix1 = eigenDecompositionImpl0.getV();
      assertEquals(7, realMatrix1.getColumnDimension());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, Double.NaN);
      OpenMapRealMatrix openMapRealMatrix0 = openMapRealVector0.outerproduct(openMapRealVector0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(openMapRealMatrix0, 1.0E-12);
      eigenDecompositionImpl0.getD();
      eigenDecompositionImpl0.getD();
      assertEquals(0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 2823.979);
      OpenMapRealMatrix openMapRealMatrix0 = openMapRealVector0.outerproduct(openMapRealVector0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(openMapRealMatrix0, 2823.979);
      eigenDecompositionImpl0.getVT();
      RealMatrix realMatrix0 = eigenDecompositionImpl0.getVT();
      assertNotNull(realMatrix0);
      assertEquals(0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 3351.153341920391);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, 3351.153341920391);
      RealMatrix realMatrix1 = eigenDecompositionImpl0.getV();
      RealMatrix realMatrix2 = eigenDecompositionImpl0.getVT();
      assertNotSame(realMatrix2, realMatrix1);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, Double.NaN);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, 1.0E-12);
      eigenDecompositionImpl0.getSolver();
      try { 
        eigenDecompositionImpl0.getEigenvector(15);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 15
         //
         verifyException("org.apache.commons.math.linear.EigenDecompositionImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-536.61924));
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, (-536.61924));
      try { 
        eigenDecompositionImpl0.getEigenvector((-2147444091));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -2147444091
         //
         verifyException("org.apache.commons.math.linear.EigenDecompositionImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 60.58675580667592);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, 0.29412889137505926);
      double double0 = eigenDecompositionImpl0.getDeterminant();
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 3351.153341920391);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, 3351.153341920391);
      eigenDecompositionImpl0.getVT();
      eigenDecompositionImpl0.getSolver();
      assertEquals(0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, Double.NaN);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, Double.NaN);
      assertEquals(Double.NaN, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      double[] doubleArray1 = new double[3];
      EigenDecompositionImpl eigenDecompositionImpl0 = null;
      try {
        eigenDecompositionImpl0 = new EigenDecompositionImpl(doubleArray1, doubleArray0, Double.NaN);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // cannot solve degree 3 equation
         //
         verifyException("org.apache.commons.math.linear.EigenDecompositionImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      double[] doubleArray0 = new double[18];
      doubleArray0[12] = 1309.5758208680322;
      doubleArray0[0] = 1.2332050199959558E-8;
      doubleArray0[9] = (-64.08393407840208);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1.2332050199959558E-8);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, 1.2332050199959558E-8);
      assertEquals(-0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      double[] doubleArray1 = new double[3];
      doubleArray1[1] = Double.NaN;
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(doubleArray1, doubleArray0, Double.NaN);
      eigenDecompositionImpl0.getVT();
      assertEquals(Double.NaN, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      doubleArray0[3] = (-2685.6497272705947);
      doubleArray0[6] = 1390.344936142925;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 0.0);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, 0.0);
      assertEquals(-0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      doubleArray0[0] = 233.497;
      doubleArray0[1] = 233.497;
      doubleArray0[2] = 233.497;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-4111.0));
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, Double.POSITIVE_INFINITY);
      assertEquals(0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-66.175829672931));
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = null;
      try {
        eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, Double.NEGATIVE_INFINITY);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // 
         //
         verifyException("org.apache.commons.math.linear.EigenDecompositionImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      doubleArray0[1] = (-2685.6497272705947);
      doubleArray0[0] = (-66.175829672931);
      doubleArray0[4] = (-2685.6497272705947);
      doubleArray0[5] = 1309.8931429937;
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, true);
      RealMatrix realMatrix0 = arrayRealVector0.outerProduct(arrayRealVector0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, (-66.175829672931));
      assertEquals(0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      doubleArray0[0] = (-66.175829672931);
      doubleArray0[1] = (-66.175829672931);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-66.175829672931));
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = null;
      try {
        eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, Double.NEGATIVE_INFINITY);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // 
         //
         verifyException("org.apache.commons.math.linear.EigenDecompositionImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      doubleArray0[1] = (-2685.6497272705947);
      doubleArray0[2] = (-66.175829672931);
      doubleArray0[4] = (-2685.6497272705947);
      doubleArray0[5] = 1309.8931429937;
      doubleArray0[6] = (-66.175829672931);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2685.6497272705947));
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, Double.NaN);
      assertEquals(-0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      double[] doubleArray0 = new double[10];
      doubleArray0[0] = (-66.175829672931);
      doubleArray0[2] = 233.497;
      doubleArray0[3] = (-2685.6497272705947);
      doubleArray0[5] = 1309.8931429937;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-858.3462758570914));
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, 8.950284666938726E-40);
      assertEquals(-0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      doubleArray0[1] = (-2685.6497272705947);
      doubleArray0[2] = (-66.175829672931);
      doubleArray0[4] = (-2685.6497272705947);
      doubleArray0[5] = 1293.6943123238998;
      doubleArray0[6] = (-66.175829672931);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-66.175829672931));
      OpenMapRealMatrix openMapRealMatrix0 = openMapRealVector0.outerproduct(openMapRealVector0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(openMapRealMatrix0, (-66.175829672931));
      assertEquals(0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      doubleArray0[5] = 1309.5758208680322;
      doubleArray0[0] = 1.2332050199959558E-8;
      doubleArray0[3] = (-64.08393407840208);
      doubleArray0[4] = (-550.2782768817898);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1.2332050199959558E-8);
      OpenMapRealMatrix openMapRealMatrix0 = openMapRealVector0.outerproduct(openMapRealVector0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(openMapRealMatrix0, (-64.08393407840208));
      assertEquals(-0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      double[] doubleArray0 = new double[18];
      doubleArray0[5] = 1309.5758208680322;
      doubleArray0[0] = 1.2332050199959558E-8;
      doubleArray0[3] = (-64.08393407840208);
      doubleArray0[4] = (-64.08393407840208);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1.2332050199959558E-8);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, (-64.08393407840208));
      assertEquals(0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      double[] doubleArray0 = new double[18];
      doubleArray0[2] = 1310.2385698756993;
      doubleArray0[0] = 1.2332050199959558E-8;
      doubleArray0[3] = (-70.01908305681151);
      doubleArray0[4] = (-70.01908305681151);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1.2332050199959558E-8);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, 1.2332050199959558E-8);
      assertEquals(-0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      double[] doubleArray0 = new double[25];
      doubleArray0[0] = (-66.175829672931);
      doubleArray0[17] = (-66.175829672931);
      doubleArray0[2] = 1.23E-8;
      doubleArray0[7] = 1309.5758208680322;
      doubleArray0[4] = 1309.5758208680322;
      doubleArray0[5] = (-66.175829672931);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-66.175829672931));
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, 1.0E-12);
      assertEquals(-0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      doubleArray0[1] = (-2685.6497272705947);
      doubleArray0[2] = (-66.175829672931);
      doubleArray0[5] = 1309.8931429937;
      doubleArray0[6] = (-66.175829672931);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-2685.6497272705947));
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, 1.0E-12);
      assertEquals(-0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      double[] doubleArray0 = new double[25];
      doubleArray0[0] = (-66.175829672931);
      doubleArray0[17] = (-66.175829672931);
      doubleArray0[2] = 1.23E-8;
      doubleArray0[3] = 1309.5758208680322;
      doubleArray0[4] = 1309.5758208680322;
      doubleArray0[5] = (-66.175829672931);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-66.175829672931));
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, (-66.175829672931));
      assertEquals(0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      doubleArray0[0] = (-2685.6497272705947);
      doubleArray0[1] = (-2685.6497272705947);
      doubleArray0[2] = (-81.53546487896779);
      doubleArray0[3] = 1309.8931429937;
      doubleArray0[4] = (-2685.6497272705947);
      doubleArray0[5] = 1309.8931429937;
      doubleArray0[6] = (-81.53546487896779);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-81.53546487896779));
      OpenMapRealMatrix openMapRealMatrix0 = openMapRealVector0.outerproduct(openMapRealVector0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(openMapRealMatrix0, (-81.53546487896779));
      assertEquals((-3.552593067940438E-53), eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      double[] doubleArray0 = new double[25];
      doubleArray0[0] = (-66.175829672931);
      doubleArray0[17] = (-66.175829672931);
      doubleArray0[4] = 1309.5758208680322;
      doubleArray0[5] = (-66.175829672931);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-66.175829672931));
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, 1.0E-12);
      assertEquals(0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      doubleArray0[0] = (-2685.6497272705947);
      doubleArray0[1] = (-2685.6497272705947);
      doubleArray0[3] = 1309.8931429937;
      doubleArray0[4] = (-2685.6497272705947);
      doubleArray0[5] = 1309.8931429937;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, (-2475.3791054392505));
      assertEquals(-0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      doubleArray0[0] = (-66.175829672931);
      doubleArray0[1] = (-66.175829672931);
      doubleArray0[2] = 1.2332050199959558E-8;
      doubleArray0[3] = 1309.5758208680322;
      doubleArray0[4] = 1309.5758208680322;
      doubleArray0[5] = (-66.175829672931);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-66.175829672931));
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, 1.0E-12);
      assertEquals(4.907718913844518E-65, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      double[] doubleArray0 = new double[25];
      doubleArray0[0] = (-66.175829672931);
      doubleArray0[17] = (-66.175829672931);
      doubleArray0[2] = 1.23E-8;
      doubleArray0[4] = 1309.5758208680322;
      doubleArray0[5] = (-66.175829672931);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-66.175829672931));
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, 1.0E-12);
      assertEquals(0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      double[] doubleArray0 = new double[25];
      doubleArray0[0] = (-66.175829672931);
      doubleArray0[17] = (-66.175829672931);
      doubleArray0[2] = (-66.175829672931);
      doubleArray0[3] = 1309.5758208680322;
      doubleArray0[4] = 1309.5758208680322;
      doubleArray0[5] = (-66.175829672931);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-66.175829672931));
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, (-66.175829672931));
      assertEquals(0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      double[] doubleArray0 = new double[18];
      doubleArray0[12] = 1309.5758208680322;
      doubleArray0[0] = (-11.297976836783002);
      doubleArray0[3] = (-64.08393407840208);
      doubleArray0[9] = (-64.08393407840208);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-11.297976836783002));
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, (-11.297976836783002));
      assertEquals(0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      doubleArray0[0] = (-66.175829672931);
      doubleArray0[1] = 233.497;
      doubleArray0[2] = 233.497;
      doubleArray0[3] = 233.497;
      doubleArray0[4] = 1309.8931429937;
      doubleArray0[5] = 1309.8931429937;
      doubleArray0[6] = 1390.344936142925;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 1.1102230246251565E-16);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, (-66.175829672931));
      assertEquals(1.2119369560654845E-87, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      doubleArray0[1] = (-2685.6497272705947);
      doubleArray0[2] = (-66.175829672931);
      doubleArray0[4] = (-2685.6497272705947);
      doubleArray0[5] = 1309.8931429937;
      doubleArray0[6] = (-66.175829672931);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, (-66.175829672931));
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, (-66.175829672931));
      assertEquals(0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      doubleArray0[0] = 233.497;
      doubleArray0[1] = 233.497;
      doubleArray0[2] = 233.497;
      doubleArray0[3] = 233.497;
      doubleArray0[5] = 233.497;
      doubleArray0[6] = 233.497;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(doubleArray0, 233.497);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct(doubleArray0);
      EigenDecompositionImpl eigenDecompositionImpl0 = new EigenDecompositionImpl(realMatrix0, (-1.0));
      eigenDecompositionImpl0.getSolver();
      assertEquals(-0.0, eigenDecompositionImpl0.getDeterminant(), 0.01);
  }
}
