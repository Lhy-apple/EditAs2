/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:37:04 GMT 2023
 */

package org.apache.commons.math.linear;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.math.BigDecimal;
import java.math.BigInteger;
import org.apache.commons.math.linear.BigMatrix;
import org.apache.commons.math.linear.BigMatrixImpl;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BigMatrixImpl_ESTest extends BigMatrixImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      double[][] doubleArray0 = new double[1][0];
      double[] doubleArray1 = new double[2];
      doubleArray0[0] = doubleArray1;
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(doubleArray0);
      assertEquals(1, bigMatrixImpl0.getRowDimension());
      
      BigMatrixImpl bigMatrixImpl1 = (BigMatrixImpl)bigMatrixImpl0.copy();
      assertEquals(4, bigMatrixImpl1.getRoundingMode());
      assertEquals(64, bigMatrixImpl1.getScale());
      assertEquals(4, bigMatrixImpl0.getRoundingMode());
      assertEquals(2, bigMatrixImpl1.getColumnDimension());
      assertEquals(64, bigMatrixImpl0.getScale());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl();
      // Undeclared exception!
      try { 
        bigMatrixImpl0.getEntryAsDouble(1, 1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BigDecimal[][] bigDecimalArray0 = new BigDecimal[1][3];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      // Undeclared exception!
      try { 
        bigMatrixImpl0.getPermutation();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      double[][] doubleArray0 = new double[1][7];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(doubleArray0);
      bigMatrixImpl0.setRoundingMode((-1907));
      assertEquals((-1907), bigMatrixImpl0.getRoundingMode());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      double[][] doubleArray0 = new double[1][0];
      double[] doubleArray1 = new double[2];
      doubleArray0[0] = doubleArray1;
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(doubleArray0);
      int int0 = bigMatrixImpl0.getScale();
      assertFalse(bigMatrixImpl0.isSquare());
      assertEquals(4, bigMatrixImpl0.getRoundingMode());
      assertEquals(64, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BigDecimal[] bigDecimalArray0 = new BigDecimal[6];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      assertEquals(64, bigMatrixImpl0.getScale());
      
      bigMatrixImpl0.setScale((byte)0);
      assertEquals(4, bigMatrixImpl0.getRoundingMode());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl();
      // Undeclared exception!
      try { 
        bigMatrixImpl0.preMultiply((BigMatrix) bigMatrixImpl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BigDecimal[] bigDecimalArray0 = new BigDecimal[6];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      try { 
        bigMatrixImpl0.inverse();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // coefficient matrix is not square
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BigMatrixImpl bigMatrixImpl0 = null;
      try {
        bigMatrixImpl0 = new BigMatrixImpl((-1), (-1));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // row and column dimensions must be positive
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(3568, 3568);
      // Undeclared exception!
      try { 
        bigMatrixImpl0.getColumnAsDoubleArray(564);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BigMatrixImpl bigMatrixImpl0 = null;
      try {
        bigMatrixImpl0 = new BigMatrixImpl(7, 0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // row and column dimensions must be positive
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BigDecimal[][] bigDecimalArray0 = new BigDecimal[1][4];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0, true);
      int[] intArray0 = new int[2];
      BigMatrixImpl bigMatrixImpl1 = (BigMatrixImpl)bigMatrixImpl0.getSubMatrix(intArray0, intArray0);
      assertEquals(4, bigMatrixImpl1.getRoundingMode());
      assertEquals(64, bigMatrixImpl1.getScale());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BigMatrixImpl bigMatrixImpl0 = null;
      try {
        bigMatrixImpl0 = new BigMatrixImpl((BigDecimal[][]) null, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BigDecimal[][] bigDecimalArray0 = new BigDecimal[0][9];
      BigMatrixImpl bigMatrixImpl0 = null;
      try {
        bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0, false);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Matrix must have at least one row.
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BigDecimal[] bigDecimalArray0 = new BigDecimal[0];
      BigDecimal[][] bigDecimalArray1 = new BigDecimal[7][9];
      bigDecimalArray1[0] = bigDecimalArray0;
      BigMatrixImpl bigMatrixImpl0 = null;
      try {
        bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray1, false);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Matrix must have at least one column.
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BigDecimal[][] bigDecimalArray0 = new BigDecimal[2][4];
      BigDecimal[] bigDecimalArray1 = new BigDecimal[1];
      bigDecimalArray0[0] = bigDecimalArray1;
      BigMatrixImpl bigMatrixImpl0 = null;
      try {
        bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0, false);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // All input rows must have the same length.
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      double[][] doubleArray0 = new double[0][0];
      BigMatrixImpl bigMatrixImpl0 = null;
      try {
        bigMatrixImpl0 = new BigMatrixImpl(doubleArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Matrix must have at least one row.
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      double[][] doubleArray0 = new double[1][2];
      double[] doubleArray1 = new double[0];
      doubleArray0[0] = doubleArray1;
      BigMatrixImpl bigMatrixImpl0 = null;
      try {
        bigMatrixImpl0 = new BigMatrixImpl(doubleArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Matrix must have at least one column.
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      double[][] doubleArray0 = new double[2][9];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(doubleArray0);
      assertEquals(4, bigMatrixImpl0.getRoundingMode());
      assertEquals(64, bigMatrixImpl0.getScale());
      assertEquals(9, bigMatrixImpl0.getColumnDimension());
      assertEquals(2, bigMatrixImpl0.getRowDimension());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      double[][] doubleArray1 = new double[9][1];
      doubleArray1[0] = doubleArray0;
      BigMatrixImpl bigMatrixImpl0 = null;
      try {
        bigMatrixImpl0 = new BigMatrixImpl(doubleArray1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // All input rows must have the same length.
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      String[][] stringArray0 = new String[6][6];
      BigMatrixImpl bigMatrixImpl0 = null;
      try {
        bigMatrixImpl0 = new BigMatrixImpl(stringArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.math.BigDecimal", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      String[][] stringArray0 = new String[0][2];
      BigMatrixImpl bigMatrixImpl0 = null;
      try {
        bigMatrixImpl0 = new BigMatrixImpl(stringArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Matrix must have at least one row.
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      String[][] stringArray0 = new String[2][9];
      String[] stringArray1 = new String[0];
      stringArray0[0] = stringArray1;
      BigMatrixImpl bigMatrixImpl0 = null;
      try {
        bigMatrixImpl0 = new BigMatrixImpl(stringArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Matrix must have at least one column.
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      String[][] stringArray0 = new String[3][3];
      String[] stringArray1 = new String[2];
      stringArray0[1] = stringArray1;
      BigMatrixImpl bigMatrixImpl0 = null;
      try {
        bigMatrixImpl0 = new BigMatrixImpl(stringArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // All input rows must have the same length.
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BigDecimal[] bigDecimalArray0 = new BigDecimal[6];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      BigMatrix bigMatrix0 = bigMatrixImpl0.transpose();
      try { 
        bigMatrix0.subtract((BigMatrix) bigMatrixImpl0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // matrix dimension mismatch
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      BigInteger bigInteger0 = new BigInteger(byteArray0);
      BigDecimal bigDecimal0 = new BigDecimal(bigInteger0);
      BigDecimal[] bigDecimalArray0 = new BigDecimal[1];
      bigDecimalArray0[0] = bigDecimal0;
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      BigMatrixImpl bigMatrixImpl1 = (BigMatrixImpl)bigMatrixImpl0.subtract((BigMatrix) bigMatrixImpl0);
      assertEquals(64, bigMatrixImpl0.getScale());
      assertEquals(4, bigMatrixImpl0.getRoundingMode());
      assertEquals(64, bigMatrixImpl1.getScale());
      assertEquals(4, bigMatrixImpl1.getRoundingMode());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      BigInteger bigInteger0 = new BigInteger(byteArray0);
      BigDecimal bigDecimal0 = new BigDecimal(bigInteger0);
      BigDecimal[] bigDecimalArray0 = new BigDecimal[6];
      bigDecimalArray0[0] = bigDecimal0;
      bigDecimalArray0[1] = bigDecimal0;
      bigDecimalArray0[2] = bigDecimal0;
      bigDecimalArray0[3] = bigDecimal0;
      bigDecimalArray0[4] = bigDecimal0;
      bigDecimalArray0[5] = bigDecimal0;
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      BigMatrixImpl bigMatrixImpl1 = (BigMatrixImpl)bigMatrixImpl0.scalarAdd(bigDecimal0);
      assertEquals(64, bigMatrixImpl0.getScale());
      assertEquals(64, bigMatrixImpl1.getScale());
      assertTrue(bigMatrixImpl1.equals((Object)bigMatrixImpl0));
      assertEquals(4, bigMatrixImpl0.getRoundingMode());
      assertEquals(4, bigMatrixImpl1.getRoundingMode());
      assertEquals(6, bigMatrixImpl1.getRowDimension());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      double[][] doubleArray0 = new double[1][0];
      double[] doubleArray1 = new double[2];
      doubleArray0[0] = doubleArray1;
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(doubleArray0);
      assertFalse(bigMatrixImpl0.isSquare());
      
      BigDecimal bigDecimal0 = BigMatrixImpl.ONE;
      BigMatrixImpl bigMatrixImpl1 = (BigMatrixImpl)bigMatrixImpl0.scalarMultiply(bigDecimal0);
      assertEquals(4, bigMatrixImpl1.getRoundingMode());
      assertEquals(4, bigMatrixImpl0.getRoundingMode());
      assertFalse(bigMatrixImpl1.isSquare());
      assertEquals(64, bigMatrixImpl1.getScale());
      assertEquals(64, bigMatrixImpl0.getScale());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      BigDecimal[] bigDecimalArray0 = new BigDecimal[10];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      try { 
        bigMatrixImpl0.multiply((BigMatrix) bigMatrixImpl0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Matrices are not multiplication compatible.
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      BigDecimal[] bigDecimalArray0 = new BigDecimal[3];
      byte[] byteArray0 = new byte[7];
      BigInteger bigInteger0 = new BigInteger(byteArray0);
      BigDecimal bigDecimal0 = new BigDecimal(bigInteger0);
      bigDecimalArray0[0] = bigDecimal0;
      bigDecimalArray0[1] = bigDecimal0;
      bigDecimalArray0[2] = bigDecimal0;
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      double[][] doubleArray0 = bigMatrixImpl0.getDataAsDoubleArray();
      assertEquals(3, doubleArray0.length);
      assertEquals(4, bigMatrixImpl0.getRoundingMode());
      assertEquals(64, bigMatrixImpl0.getScale());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      double[][] doubleArray0 = new double[1][0];
      double[] doubleArray1 = new double[2];
      doubleArray0[0] = doubleArray1;
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(doubleArray0);
      BigDecimal bigDecimal0 = bigMatrixImpl0.getNorm();
      assertEquals(64, bigMatrixImpl0.getScale());
      assertEquals((short)0, bigDecimal0.shortValue());
      assertEquals(4, bigMatrixImpl0.getRoundingMode());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      BigDecimal[] bigDecimalArray0 = new BigDecimal[5];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      try { 
        bigMatrixImpl0.getSubMatrix((int) (byte) (-34), (int) (byte)79, (int) (byte) (-34), (int) (byte) (-34));
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // invalid row or column index selection
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(2354, 2354);
      try { 
        bigMatrixImpl0.getSubMatrix(2354, 2354, 3568, 2354);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // invalid row or column index selection
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl();
      try { 
        bigMatrixImpl0.getSubMatrix(2354, 2332, 2325, 9);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // invalid row or column index selection
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      BigDecimal[] bigDecimalArray0 = new BigDecimal[0];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      try { 
        bigMatrixImpl0.getSubMatrix(120, 120, 120, 120);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // invalid row or column index selection
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      BigDecimal[][] bigDecimalArray0 = new BigDecimal[1][4];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0, true);
      int[] intArray0 = new int[8];
      int[] intArray1 = new int[0];
      try { 
        bigMatrixImpl0.getSubMatrix(intArray0, intArray1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // selected row and column index arrays must be non-empty
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl();
      try { 
        bigMatrixImpl0.setSubMatrix((BigDecimal[][]) null, (-1), 1253);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // invalid row or column index selection
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      BigDecimal[][] bigDecimalArray0 = new BigDecimal[0][8];
      BigMatrixImpl bigMatrixImpl0 = null;
      try {
        bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Matrix must have at least one row.
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      BigDecimal[][] bigDecimalArray0 = new BigDecimal[3][0];
      BigMatrixImpl bigMatrixImpl0 = null;
      try {
        bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Matrix must have at least one column.
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      BigDecimal[][] bigDecimalArray0 = new BigDecimal[3][7];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0, true);
      assertEquals(64, bigMatrixImpl0.getScale());
      assertEquals(4, bigMatrixImpl0.getRoundingMode());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      BigDecimal[][] bigDecimalArray0 = new BigDecimal[3][7];
      BigDecimal[] bigDecimalArray1 = new BigDecimal[18];
      bigDecimalArray0[0] = bigDecimalArray1;
      BigMatrixImpl bigMatrixImpl0 = null;
      try {
        bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0, true);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // All input rows must have the same length.
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(4921, 4921);
      BigMatrixImpl bigMatrixImpl1 = (BigMatrixImpl)bigMatrixImpl0.getColumnMatrix((byte)0);
      assertEquals(4921, bigMatrixImpl1.getRowDimension());
      assertEquals(64, bigMatrixImpl0.getScale());
      assertEquals(64, bigMatrixImpl1.getScale());
      assertEquals(4, bigMatrixImpl1.getRoundingMode());
      assertTrue(bigMatrixImpl0.isSquare());
      assertEquals(4, bigMatrixImpl0.getRoundingMode());
      assertEquals(1, bigMatrixImpl1.getColumnDimension());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(1768, 2306);
      BigDecimal[] bigDecimalArray0 = bigMatrixImpl0.getRow(2);
      assertEquals(2306, bigDecimalArray0.length);
      assertEquals(64, bigMatrixImpl0.getScale());
      assertEquals(1768, bigMatrixImpl0.getRowDimension());
      assertEquals(4, bigMatrixImpl0.getRoundingMode());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      BigDecimal[] bigDecimalArray0 = new BigDecimal[1];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      try { 
        bigMatrixImpl0.getRow((-1225));
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // illegal row argument
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      BigDecimal[] bigDecimalArray0 = new BigDecimal[9];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      try { 
        bigMatrixImpl0.getRowAsDoubleArray((byte)64);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // illegal row argument
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      BigInteger bigInteger0 = new BigInteger(byteArray0);
      BigDecimal bigDecimal0 = new BigDecimal(bigInteger0);
      BigDecimal[] bigDecimalArray0 = new BigDecimal[6];
      bigDecimalArray0[0] = bigDecimal0;
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      double[] doubleArray0 = bigMatrixImpl0.getRowAsDoubleArray((byte)0);
      assertEquals(4, bigMatrixImpl0.getRoundingMode());
      assertEquals(1, doubleArray0.length);
      assertEquals(64, bigMatrixImpl0.getScale());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      BigDecimal[][] bigDecimalArray0 = new BigDecimal[1][4];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0, true);
      try { 
        bigMatrixImpl0.getColumnAsDoubleArray(852);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // illegal column argument
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(3568, 3568);
      // Undeclared exception!
      try { 
        bigMatrixImpl0.getTrace();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      BigDecimal[] bigDecimalArray0 = new BigDecimal[3];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      try { 
        bigMatrixImpl0.getTrace();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // matrix is not square
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      double[][] doubleArray0 = new double[1][0];
      double[] doubleArray1 = new double[2];
      doubleArray0[0] = doubleArray1;
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(doubleArray0);
      bigMatrixImpl0.operate(doubleArray1);
      assertEquals(4, bigMatrixImpl0.getRoundingMode());
      assertEquals(64, bigMatrixImpl0.getScale());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(3568, 3568);
      BigDecimal[] bigDecimalArray0 = new BigDecimal[6];
      try { 
        bigMatrixImpl0.operate(bigDecimalArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // vector has wrong length
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      BigDecimal[] bigDecimalArray0 = new BigDecimal[6];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      try { 
        bigMatrixImpl0.solve(bigDecimalArray0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // coefficient matrix is not square
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      BigDecimal[] bigDecimalArray0 = new BigDecimal[6];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      double[] doubleArray0 = new double[9];
      try { 
        bigMatrixImpl0.solve(doubleArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // constant vector has wrong length
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      BigDecimal[] bigDecimalArray0 = new BigDecimal[2];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      BigMatrix bigMatrix0 = bigMatrixImpl0.transpose();
      try { 
        bigMatrixImpl0.solve(bigMatrix0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Incorrect row dimension
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[4] = (byte)36;
      BigInteger bigInteger0 = new BigInteger(byteArray0);
      BigDecimal bigDecimal0 = new BigDecimal(bigInteger0);
      BigDecimal[] bigDecimalArray0 = new BigDecimal[1];
      bigDecimalArray0[0] = bigDecimal0;
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      assertTrue(bigMatrixImpl0.isSquare());
      
      BigMatrixImpl bigMatrixImpl1 = (BigMatrixImpl)bigMatrixImpl0.solve((BigMatrix) bigMatrixImpl0);
      assertEquals(64, bigMatrixImpl0.getScale());
      assertEquals(64, bigMatrixImpl1.getScale());
      assertEquals(4, bigMatrixImpl0.getRoundingMode());
      assertEquals(4, bigMatrixImpl1.getRoundingMode());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(1768, 2306);
      try { 
        bigMatrixImpl0.getLUMatrix();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // LU decomposition requires that the matrix be square.
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[0] = (byte)6;
      BigInteger bigInteger0 = new BigInteger(byteArray0);
      BigDecimal bigDecimal0 = new BigDecimal(bigInteger0);
      BigDecimal[] bigDecimalArray0 = new BigDecimal[6];
      bigDecimalArray0[0] = bigDecimal0;
      bigDecimalArray0[1] = bigDecimal0;
      bigDecimalArray0[2] = bigDecimal0;
      bigDecimalArray0[3] = bigDecimal0;
      bigDecimalArray0[4] = bigDecimal0;
      bigDecimalArray0[5] = bigDecimal0;
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      BigMatrix bigMatrix0 = bigMatrixImpl0.transpose();
      BigMatrix bigMatrix1 = bigMatrixImpl0.multiply(bigMatrix0);
      try { 
        bigMatrix1.solve(bigDecimalArray0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Matrix is singular.
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl();
      String string0 = bigMatrixImpl0.toString();
      assertEquals(4, bigMatrixImpl0.getRoundingMode());
      assertEquals(64, bigMatrixImpl0.getScale());
      assertEquals("BigMatrixImpl{}", string0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      BigDecimal[][] bigDecimalArray0 = new BigDecimal[2][5];
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0, false);
      String string0 = bigMatrixImpl0.toString();
      assertEquals(64, bigMatrixImpl0.getScale());
      assertEquals(4, bigMatrixImpl0.getRoundingMode());
      assertEquals("BigMatrixImpl{{null,null,null,null,null},{null,null,null,null,null}}", string0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl();
      boolean boolean0 = bigMatrixImpl0.equals("");
      assertFalse(boolean0);
      assertEquals(4, bigMatrixImpl0.getRoundingMode());
      assertEquals(64, bigMatrixImpl0.getScale());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      BigInteger bigInteger0 = new BigInteger(byteArray0);
      BigDecimal bigDecimal0 = new BigDecimal(bigInteger0);
      BigDecimal[] bigDecimalArray0 = new BigDecimal[6];
      bigDecimalArray0[0] = bigDecimal0;
      bigDecimalArray0[1] = bigDecimal0;
      bigDecimalArray0[2] = bigDecimal0;
      bigDecimalArray0[3] = bigDecimal0;
      bigDecimalArray0[4] = bigDecimal0;
      bigDecimalArray0[5] = bigDecimal0;
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(bigDecimalArray0);
      bigMatrixImpl0.hashCode();
      assertEquals(4, bigMatrixImpl0.getRoundingMode());
      assertEquals(64, bigMatrixImpl0.getScale());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      BigMatrixImpl bigMatrixImpl0 = new BigMatrixImpl(2354, 2354);
      try { 
        bigMatrixImpl0.getColumnMatrix((-1513));
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // illegal column argument
         //
         verifyException("org.apache.commons.math.linear.BigMatrixImpl", e);
      }
  }
}
