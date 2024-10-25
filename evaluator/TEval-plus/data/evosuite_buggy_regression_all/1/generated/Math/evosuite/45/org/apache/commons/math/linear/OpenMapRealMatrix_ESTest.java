/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:05:26 GMT 2023
 */

package org.apache.commons.math.linear;

import org.junit.Test;
import static org.junit.Assert.*;
import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.OpenMapRealMatrix;
import org.apache.commons.math.linear.RealMatrix;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class OpenMapRealMatrix_ESTest extends OpenMapRealMatrix_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      OpenMapRealMatrix openMapRealMatrix0 = new OpenMapRealMatrix(1167, 1167);
      openMapRealMatrix0.addToEntry(52, 52, 6.944444444437549E-5);
      assertEquals(1167, openMapRealMatrix0.getRowDimension());
      assertEquals(1167, openMapRealMatrix0.getColumnDimension());
      
      OpenMapRealMatrix openMapRealMatrix1 = openMapRealMatrix0.multiply(openMapRealMatrix0);
      assertEquals(1167, openMapRealMatrix1.getColumnDimension());
      assertEquals(1167, openMapRealMatrix1.getRowDimension());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      OpenMapRealMatrix openMapRealMatrix0 = new OpenMapRealMatrix(983, 983);
      OpenMapRealMatrix openMapRealMatrix1 = openMapRealMatrix0.copy();
      assertEquals(983, openMapRealMatrix1.getColumnDimension());
      assertEquals(983, openMapRealMatrix1.getRowDimension());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      OpenMapRealMatrix openMapRealMatrix0 = new OpenMapRealMatrix(1167, 1167);
      OpenMapRealMatrix openMapRealMatrix1 = openMapRealMatrix0.createMatrix(1843, 1775);
      assertEquals(1843, openMapRealMatrix1.getRowDimension());
      assertEquals(1775, openMapRealMatrix1.getColumnDimension());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      OpenMapRealMatrix openMapRealMatrix0 = new OpenMapRealMatrix(1167, 1167);
      openMapRealMatrix0.setEntry(74, 74, 0.01745329052209854);
      OpenMapRealMatrix openMapRealMatrix1 = openMapRealMatrix0.add(openMapRealMatrix0);
      assertEquals(1167, openMapRealMatrix0.getColumnDimension());
      assertEquals(1167, openMapRealMatrix1.getColumnDimension());
      assertNotSame(openMapRealMatrix1, openMapRealMatrix0);
      assertEquals(1167, openMapRealMatrix1.getRowDimension());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      OpenMapRealMatrix openMapRealMatrix0 = new OpenMapRealMatrix(1167, 1167);
      openMapRealMatrix0.setEntry(74, 74, 0.01745329052209854);
      OpenMapRealMatrix openMapRealMatrix1 = openMapRealMatrix0.subtract((RealMatrix) openMapRealMatrix0);
      assertEquals(1167, openMapRealMatrix1.getRowDimension());
      assertEquals(1167, openMapRealMatrix1.getColumnDimension());
      assertNotSame(openMapRealMatrix1, openMapRealMatrix0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      OpenMapRealMatrix openMapRealMatrix0 = new OpenMapRealMatrix(1156, 1156);
      BlockRealMatrix blockRealMatrix0 = new BlockRealMatrix(1156, 1156);
      openMapRealMatrix0.addToEntry(52, 52, 52);
      assertEquals(1156, openMapRealMatrix0.getColumnDimension());
      
      RealMatrix realMatrix0 = openMapRealMatrix0.multiply((RealMatrix) blockRealMatrix0);
      assertEquals(1156, realMatrix0.getRowDimension());
      assertEquals(1156, realMatrix0.getColumnDimension());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      OpenMapRealMatrix openMapRealMatrix0 = new OpenMapRealMatrix(1167, 1167);
      assertEquals(1167, openMapRealMatrix0.getRowDimension());
      
      OpenMapRealMatrix openMapRealMatrix1 = openMapRealMatrix0.multiply(openMapRealMatrix0);
      assertEquals(1167, openMapRealMatrix1.getColumnDimension());
      assertEquals(1167, openMapRealMatrix1.getRowDimension());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      OpenMapRealMatrix openMapRealMatrix0 = new OpenMapRealMatrix(2573, 1167);
      openMapRealMatrix0.addToEntry(52, 224, 0.0);
      assertEquals(2573, openMapRealMatrix0.getRowDimension());
      assertEquals(1167, openMapRealMatrix0.getColumnDimension());
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      OpenMapRealMatrix openMapRealMatrix0 = new OpenMapRealMatrix(1167, 2073);
      openMapRealMatrix0.addToEntry(756, 0, 0.008333333333329196);
      openMapRealMatrix0.multiplyEntry(756, 0, 0.008333333333329196);
      assertEquals(1167, openMapRealMatrix0.getRowDimension());
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      OpenMapRealMatrix openMapRealMatrix0 = new OpenMapRealMatrix(1167, 1167);
      openMapRealMatrix0.multiplyEntry(572, 572, 20.130795466893055);
      assertEquals(1167, openMapRealMatrix0.getColumnDimension());
  }
}
