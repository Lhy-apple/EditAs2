/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:19:47 GMT 2023
 */

package org.jfree.data.category;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.lang.reflect.Array;
import java.util.List;
import javax.swing.JLayeredPane;
import javax.swing.table.DefaultTableModel;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.jfree.chart.util.TableOrder;
import org.jfree.data.category.CategoryToPieDataset;
import org.jfree.data.category.DefaultIntervalCategoryDataset;
import org.jfree.data.time.Day;
import org.jfree.data.time.Quarter;
import org.jfree.data.time.RegularTimePeriod;
import org.jfree.data.time.Week;
import org.jfree.data.xy.XYDataItem;
import org.junit.runner.RunWith;
import sun.util.calendar.ZoneInfo;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DefaultIntervalCategoryDataset_ESTest extends DefaultIntervalCategoryDataset_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      double[][] doubleArray0 = new double[7][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      TableOrder tableOrder0 = TableOrder.BY_ROW;
      CategoryToPieDataset categoryToPieDataset0 = new CategoryToPieDataset(defaultIntervalCategoryDataset0, tableOrder0, 4);
      Comparable comparable0 = categoryToPieDataset0.getKey(4);
      assertNotNull(comparable0);
      assertEquals("Category 5", comparable0);
      assertEquals(7, defaultIntervalCategoryDataset0.getRowCount());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      double[][] doubleArray0 = new double[9][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getValue(3, (-1));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.getValue(): category index out of range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      double[][] doubleArray0 = new double[1][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset1 = (DefaultIntervalCategoryDataset)defaultIntervalCategoryDataset0.clone();
      boolean boolean0 = defaultIntervalCategoryDataset1.equals(defaultIntervalCategoryDataset0);
      assertEquals(1, defaultIntervalCategoryDataset1.getRowCount());
      assertNotSame(defaultIntervalCategoryDataset1, defaultIntervalCategoryDataset0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      double[][] doubleArray0 = new double[8][2];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Day day0 = new Day();
      int int0 = defaultIntervalCategoryDataset0.getRowIndex(day0);
      assertEquals(8, defaultIntervalCategoryDataset0.getRowCount());
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      double[][] doubleArray0 = new double[12][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Comparable comparable0 = defaultIntervalCategoryDataset0.getRowKey(0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getEndValue(comparable0, comparable0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unknown 'category' key.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      String[] stringArray0 = new String[2];
      Number[][] numberArray0 = new Number[0][9];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(stringArray0, numberArray0, numberArray0);
      int int0 = defaultIntervalCategoryDataset0.getCategoryCount();
      assertEquals(0, int0);
      assertEquals(0, defaultIntervalCategoryDataset0.getSeriesCount());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset((Number[][]) null, (Number[][]) null);
      int int0 = defaultIntervalCategoryDataset0.getCategoryCount();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Number[][] numberArray0 = new Number[0][2];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, (Number[][]) null);
      assertEquals(0, defaultIntervalCategoryDataset0.getCategoryCount());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      double[][] doubleArray0 = new double[5][7];
      double[][] doubleArray1 = new double[0][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = null;
      try {
        defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray1, doubleArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset: the number of series in the start value dataset does not match the number of series in the end value dataset.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Comparable<DefaultIntervalCategoryDataset>[] comparableArray0 = (Comparable<DefaultIntervalCategoryDataset>[]) Array.newInstance(Comparable.class, 2);
      Number[][] numberArray0 = new Number[1][9];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = null;
      try {
        defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(comparableArray0, comparableArray0, numberArray0, numberArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The number of series keys does not match the number of series in the data.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Comparable<DefaultTableModel>[] comparableArray0 = (Comparable<DefaultTableModel>[]) Array.newInstance(Comparable.class, 1);
      Number[][] numberArray0 = new Number[1][5];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = null;
      try {
        defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(comparableArray0, comparableArray0, numberArray0, numberArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The number of category keys does not match the number of categories in the data.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      double[][] doubleArray0 = new double[1][2];
      double[][] doubleArray1 = new double[1][2];
      double[] doubleArray2 = new double[0];
      doubleArray1[0] = doubleArray2;
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = null;
      try {
        defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset: the number of categories in the start value dataset does not match the number of categories in the end value dataset.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Comparable<Integer>[] comparableArray0 = (Comparable<Integer>[]) Array.newInstance(Comparable.class, 1);
      Number[][] numberArray0 = new Number[1][1];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(comparableArray0, comparableArray0, numberArray0, numberArray0);
      assertEquals(1, defaultIntervalCategoryDataset0.getColumnCount());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Week week0 = new Week();
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset((Number[][]) null, (Number[][]) null);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.setEndValue(1, week0, 53);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.setValue: series outside valid range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      double[][] doubleArray0 = new double[5][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getSeriesKey(53);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // No such series : 53
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      double[][] doubleArray0 = new double[10][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Comparable comparable0 = defaultIntervalCategoryDataset0.getSeriesKey(1);
      assertEquals(10, defaultIntervalCategoryDataset0.getSeriesCount());
      assertEquals("Series 2", comparable0);
      assertEquals(10, defaultIntervalCategoryDataset0.getRowCount());
      assertNotNull(comparable0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      double[][] doubleArray0 = new double[9][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getSeriesKey((-127));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // No such series : -127
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      double[][] doubleArray0 = new double[31][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.setSeriesKeys((Comparable[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'seriesKeys' argument.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      double[][] doubleArray0 = new double[3][9];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Comparable<Object>[] comparableArray0 = (Comparable<Object>[]) Array.newInstance(Comparable.class, 0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.setSeriesKeys(comparableArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The number of series keys does not match the data.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      double[][] doubleArray0 = new double[14][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      defaultIntervalCategoryDataset0.getValue(1, 1);
      assertEquals(14, defaultIntervalCategoryDataset0.getRowCount());
      assertEquals(7, defaultIntervalCategoryDataset0.getCategoryCount());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      double[][] doubleArray0 = new double[20][2];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      List list0 = defaultIntervalCategoryDataset0.getColumnKeys();
      assertFalse(list0.isEmpty());
      assertEquals(20, defaultIntervalCategoryDataset0.getRowCount());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Number[][] numberArray0 = new Number[0][5];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      List list0 = defaultIntervalCategoryDataset0.getColumnKeys();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      double[][] doubleArray0 = new double[6][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Comparable<DefaultIntervalCategoryDataset>[] comparableArray0 = (Comparable<DefaultIntervalCategoryDataset>[]) Array.newInstance(Comparable.class, 7);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.setCategoryKeys(comparableArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.setCategoryKeys(): null category not permitted.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      double[][] doubleArray0 = new double[1][2];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.setCategoryKeys((Comparable[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'categoryKeys' argument.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      double[][] doubleArray0 = new double[2][2];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Comparable<String>[] comparableArray0 = (Comparable<String>[]) Array.newInstance(Comparable.class, 0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.setCategoryKeys(comparableArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The number of categories does not match the data.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      double[][] doubleArray0 = new double[6][7];
      Week week0 = new Week();
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Comparable<DefaultIntervalCategoryDataset>[] comparableArray0 = (Comparable<DefaultIntervalCategoryDataset>[]) Array.newInstance(Comparable.class, 7);
      comparableArray0[0] = (Comparable<DefaultIntervalCategoryDataset>) week0;
      comparableArray0[1] = (Comparable<DefaultIntervalCategoryDataset>) week0;
      XYDataItem xYDataItem0 = new XYDataItem((Number) 1, (Number) 53);
      comparableArray0[2] = (Comparable<DefaultIntervalCategoryDataset>) xYDataItem0;
      comparableArray0[3] = (Comparable<DefaultIntervalCategoryDataset>) week0;
      MockDate mockDate0 = new MockDate(1);
      Quarter quarter0 = new Quarter(mockDate0);
      comparableArray0[4] = (Comparable<DefaultIntervalCategoryDataset>) quarter0;
      comparableArray0[5] = (Comparable<DefaultIntervalCategoryDataset>) week0;
      ZoneInfo zoneInfo0 = (ZoneInfo)RegularTimePeriod.DEFAULT_TIME_ZONE;
      Day day0 = new Day(mockDate0, zoneInfo0);
      comparableArray0[6] = (Comparable<DefaultIntervalCategoryDataset>) day0;
      defaultIntervalCategoryDataset0.setCategoryKeys(comparableArray0);
      defaultIntervalCategoryDataset0.setStartValue(0, week0, 1);
      assertEquals(6, defaultIntervalCategoryDataset0.getRowCount());
      assertEquals(6, defaultIntervalCategoryDataset0.getSeriesCount());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      double[][] doubleArray0 = new double[7][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Comparable comparable0 = defaultIntervalCategoryDataset0.getRowKey(0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getValue(comparable0, comparable0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unknown 'category' key.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      double[][] doubleArray0 = new double[2][2];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getValue((Comparable) integer0, (Comparable) integer0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unknown 'series' key.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      double[][] doubleArray0 = new double[19][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Comparable comparable0 = defaultIntervalCategoryDataset0.getRowKey(0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getStartValue(comparable0, comparable0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unknown 'category' key.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      double[][] doubleArray0 = new double[7][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Week week0 = new Week();
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getStartValue((Comparable) week0, (Comparable) week0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unknown 'series' key.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      double[][] doubleArray0 = new double[1][2];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getStartValue((-584), (-584));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.getValue(): series index out of range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      double[][] doubleArray0 = new double[40][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getStartValue(33, 33);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.getValue(): category index out of range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      double[][] doubleArray0 = new double[2][2];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getStartValue(2105, 0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.getValue(): series index out of range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      double[][] doubleArray0 = new double[2][2];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getStartValue(0, (-1817));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.getValue(): category index out of range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      double[][] doubleArray0 = new double[9][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      defaultIntervalCategoryDataset0.getStartValue(1, 1);
      assertEquals(9, defaultIntervalCategoryDataset0.getSeriesCount());
      assertEquals(9, defaultIntervalCategoryDataset0.getRowCount());
      assertEquals(7, defaultIntervalCategoryDataset0.getCategoryCount());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      double[][] doubleArray0 = new double[12][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Integer integer0 = JLayeredPane.DRAG_LAYER;
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getEndValue((Comparable) integer0, (Comparable) integer0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unknown 'series' key.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      double[][] doubleArray0 = new double[1][2];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getEndValue((-1028), (-1028));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.getValue(): series index out of range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      double[][] doubleArray0 = new double[7][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getEndValue(8, 8);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.getValue(): series index out of range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      double[][] doubleArray0 = new double[40][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getEndValue(32, 32);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.getValue(): category index out of range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      double[][] doubleArray0 = new double[12][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Week week0 = new Week((-5), (-5));
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.setStartValue((-5), week0, 53);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.setValue: series outside valid range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      double[][] doubleArray0 = new double[6][7];
      Week week0 = new Week();
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.setStartValue(0, week0, 1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.setValue: unrecognised category.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      double[][] doubleArray0 = new double[7][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Week week0 = new Week();
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.setStartValue(53, week0, 1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.setValue: series outside valid range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      double[][] doubleArray0 = new double[9][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Week week0 = new Week();
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.setEndValue((-1042), week0, 1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.setValue: series outside valid range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      double[][] doubleArray0 = new double[7][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Day day0 = new Day();
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.setEndValue(4, day0, (Number) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.setValue: unrecognised category.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      double[][] doubleArray0 = new double[2][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getColumnIndex((Comparable) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'columnKey' argument.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      double[][] doubleArray0 = new double[7][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      List list0 = defaultIntervalCategoryDataset0.getRowKeys();
      assertEquals(7, list0.size());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      double[][] doubleArray0 = new double[0][9];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      List list0 = defaultIntervalCategoryDataset0.getRowKeys();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      double[][] doubleArray0 = new double[5][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getRowKey(225);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The 'row' argument is out of bounds.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      double[][] doubleArray0 = new double[7][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getRowKey((-1419));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The 'row' argument is out of bounds.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      double[][] doubleArray0 = new double[1][2];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      boolean boolean0 = defaultIntervalCategoryDataset0.equals(defaultIntervalCategoryDataset0);
      assertEquals(1, defaultIntervalCategoryDataset0.getRowCount());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      double[][] doubleArray0 = new double[0][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Integer integer0 = JLayeredPane.FRAME_CONTENT_LAYER;
      boolean boolean0 = defaultIntervalCategoryDataset0.equals(integer0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      double[][] doubleArray0 = new double[1][7];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset1 = (DefaultIntervalCategoryDataset)defaultIntervalCategoryDataset0.clone();
      assertTrue(defaultIntervalCategoryDataset1.equals((Object)defaultIntervalCategoryDataset0));
      
      Comparable<Object>[] comparableArray0 = (Comparable<Object>[]) Array.newInstance(Comparable.class, 1);
      defaultIntervalCategoryDataset1.setSeriesKeys(comparableArray0);
      boolean boolean0 = defaultIntervalCategoryDataset1.equals(defaultIntervalCategoryDataset0);
      assertFalse(boolean0);
      assertFalse(defaultIntervalCategoryDataset1.equals((Object)defaultIntervalCategoryDataset0));
  }
}
