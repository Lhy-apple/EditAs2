/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:34:10 GMT 2023
 */

package org.apache.commons.math.linear;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.ArrayRealVector;
import org.apache.commons.math.linear.DecompositionSolver;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.linear.SingularValueDecompositionImpl;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SingularValueDecompositionImpl_ESTest extends SingularValueDecompositionImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, true);
      RealMatrix realMatrix0 = arrayRealVector0.outerProduct(doubleArray0);
      SingularValueDecompositionImpl singularValueDecompositionImpl0 = new SingularValueDecompositionImpl(realMatrix0);
      // Undeclared exception!
      try { 
        singularValueDecompositionImpl0.getConditionNumber();
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 0
         //
         verifyException("org.apache.commons.math.linear.SingularValueDecompositionImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Array2DRowRealMatrix array2DRowRealMatrix0 = new Array2DRowRealMatrix(1, 1);
      SingularValueDecompositionImpl singularValueDecompositionImpl0 = new SingularValueDecompositionImpl(array2DRowRealMatrix0, 1);
      double[] doubleArray0 = singularValueDecompositionImpl0.getSingularValues();
      assertEquals(0, doubleArray0.length);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      Array2DRowRealMatrix array2DRowRealMatrix0 = new Array2DRowRealMatrix(doubleArray0);
      SingularValueDecompositionImpl singularValueDecompositionImpl0 = new SingularValueDecompositionImpl(array2DRowRealMatrix0);
      // Undeclared exception!
      try { 
        singularValueDecompositionImpl0.getNorm();
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 0
         //
         verifyException("org.apache.commons.math.linear.SingularValueDecompositionImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      doubleArray0[5] = 1042.8362758492394;
      Array2DRowRealMatrix array2DRowRealMatrix0 = new Array2DRowRealMatrix(doubleArray0);
      SingularValueDecompositionImpl singularValueDecompositionImpl0 = new SingularValueDecompositionImpl(array2DRowRealMatrix0);
      singularValueDecompositionImpl0.getCovariance(1042.8362758492394);
      RealMatrix realMatrix0 = singularValueDecompositionImpl0.getVT();
      assertFalse(realMatrix0.equals((Object)array2DRowRealMatrix0));
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      doubleArray0[0] = (double) 1;
      Array2DRowRealMatrix array2DRowRealMatrix0 = new Array2DRowRealMatrix(doubleArray0);
      SingularValueDecompositionImpl singularValueDecompositionImpl0 = new SingularValueDecompositionImpl(array2DRowRealMatrix0);
      singularValueDecompositionImpl0.getSolver();
      singularValueDecompositionImpl0.getU();
      assertEquals(1, singularValueDecompositionImpl0.getRank());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      Array2DRowRealMatrix array2DRowRealMatrix0 = new Array2DRowRealMatrix(doubleArray0);
      RealMatrix realMatrix0 = array2DRowRealMatrix0.transpose();
      SingularValueDecompositionImpl singularValueDecompositionImpl0 = new SingularValueDecompositionImpl(realMatrix0);
      // Undeclared exception!
      try { 
        singularValueDecompositionImpl0.getSolver();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // column index -1 out of allowed range [0, 0]
         //
         verifyException("org.apache.commons.math.linear.MatrixUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = Double.NaN;
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, true);
      RealMatrix realMatrix0 = arrayRealVector0.outerProduct(doubleArray0);
      SingularValueDecompositionImpl singularValueDecompositionImpl0 = new SingularValueDecompositionImpl(realMatrix0);
      RealMatrix realMatrix1 = singularValueDecompositionImpl0.getU();
      assertEquals(2, realMatrix1.getColumnDimension());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      double[] doubleArray0 = new double[8];
      doubleArray0[5] = 1049.1888008815;
      Array2DRowRealMatrix array2DRowRealMatrix0 = new Array2DRowRealMatrix(doubleArray0);
      SingularValueDecompositionImpl singularValueDecompositionImpl0 = new SingularValueDecompositionImpl(array2DRowRealMatrix0);
      singularValueDecompositionImpl0.getUT();
      DecompositionSolver decompositionSolver0 = singularValueDecompositionImpl0.getSolver();
      assertEquals(1, singularValueDecompositionImpl0.getRank());
      assertFalse(decompositionSolver0.isNonSingular());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      doubleArray0[0] = (double) 1;
      Array2DRowRealMatrix array2DRowRealMatrix0 = new Array2DRowRealMatrix(doubleArray0);
      SingularValueDecompositionImpl singularValueDecompositionImpl0 = new SingularValueDecompositionImpl(array2DRowRealMatrix0);
      singularValueDecompositionImpl0.getS();
      RealMatrix realMatrix0 = singularValueDecompositionImpl0.getS();
      assertEquals(1, realMatrix0.getRowDimension());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      doubleArray0[0] = (double) 1;
      Array2DRowRealMatrix array2DRowRealMatrix0 = new Array2DRowRealMatrix(doubleArray0);
      SingularValueDecompositionImpl singularValueDecompositionImpl0 = new SingularValueDecompositionImpl(array2DRowRealMatrix0);
      DecompositionSolver decompositionSolver0 = singularValueDecompositionImpl0.getSolver();
      assertFalse(decompositionSolver0.isNonSingular());
      
      singularValueDecompositionImpl0.getV();
      assertEquals(1, singularValueDecompositionImpl0.getRank());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      doubleArray0[0] = (double) 1;
      Array2DRowRealMatrix array2DRowRealMatrix0 = new Array2DRowRealMatrix(doubleArray0);
      RealMatrix realMatrix0 = array2DRowRealMatrix0.transpose();
      SingularValueDecompositionImpl singularValueDecompositionImpl0 = new SingularValueDecompositionImpl(realMatrix0);
      DecompositionSolver decompositionSolver0 = singularValueDecompositionImpl0.getSolver();
      assertEquals(1, singularValueDecompositionImpl0.getRank());
      assertFalse(decompositionSolver0.isNonSingular());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      double[] doubleArray0 = new double[8];
      doubleArray0[5] = 1049.1888008815;
      Array2DRowRealMatrix array2DRowRealMatrix0 = new Array2DRowRealMatrix(doubleArray0);
      Array2DRowRealMatrix array2DRowRealMatrix1 = (Array2DRowRealMatrix)array2DRowRealMatrix0.transpose();
      double[][] doubleArray1 = new double[6][3];
      doubleArray1[0] = doubleArray0;
      doubleArray1[2] = doubleArray0;
      doubleArray1[3] = doubleArray0;
      doubleArray1[4] = doubleArray0;
      array2DRowRealMatrix1.data = doubleArray1;
      SingularValueDecompositionImpl singularValueDecompositionImpl0 = new SingularValueDecompositionImpl(array2DRowRealMatrix1);
      Array2DRowRealMatrix array2DRowRealMatrix2 = (Array2DRowRealMatrix)singularValueDecompositionImpl0.getV();
      assertEquals(2098.377601763, singularValueDecompositionImpl0.getNorm(), 0.01);
      assertEquals(2, array2DRowRealMatrix2.getColumnDimension());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      double[] doubleArray0 = new double[8];
      doubleArray0[5] = 1049.1888008815;
      Array2DRowRealMatrix array2DRowRealMatrix0 = new Array2DRowRealMatrix(doubleArray0);
      SingularValueDecompositionImpl singularValueDecompositionImpl0 = new SingularValueDecompositionImpl(array2DRowRealMatrix0);
      // Undeclared exception!
      try { 
        singularValueDecompositionImpl0.getCovariance(2777.2073862487055);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // cutoff singular value is 2,777.207, should be at most 1,049.189
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      double[] doubleArray0 = new double[8];
      doubleArray0[2] = Double.POSITIVE_INFINITY;
      Array2DRowRealMatrix array2DRowRealMatrix0 = new Array2DRowRealMatrix(doubleArray0);
      SingularValueDecompositionImpl singularValueDecompositionImpl0 = new SingularValueDecompositionImpl(array2DRowRealMatrix0);
      DecompositionSolver decompositionSolver0 = singularValueDecompositionImpl0.getSolver();
      assertEquals(0, singularValueDecompositionImpl0.getRank());
      assertFalse(decompositionSolver0.isNonSingular());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = (double) 1;
      Array2DRowRealMatrix array2DRowRealMatrix0 = new Array2DRowRealMatrix(doubleArray0);
      SingularValueDecompositionImpl singularValueDecompositionImpl0 = new SingularValueDecompositionImpl(array2DRowRealMatrix0);
      DecompositionSolver decompositionSolver0 = singularValueDecompositionImpl0.getSolver();
      assertTrue(decompositionSolver0.isNonSingular());
  }
}