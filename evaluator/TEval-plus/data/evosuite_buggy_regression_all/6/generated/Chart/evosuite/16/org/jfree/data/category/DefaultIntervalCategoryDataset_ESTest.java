/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:46:22 GMT 2023
 */

package org.jfree.data.category;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.lang.reflect.Array;
import java.util.List;
import java.util.Stack;
import javax.swing.JLayeredPane;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jfree.data.category.DefaultIntervalCategoryDataset;
import org.jfree.data.time.Day;
import org.jfree.data.time.SerialDate;
import org.jfree.data.time.Week;
import org.jfree.data.time.Year;
import org.jfree.data.xy.XYDatasetTableModel;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DefaultIntervalCategoryDataset_ESTest extends DefaultIntervalCategoryDataset_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      double[][] doubleArray0 = new double[1][0];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      int int0 = defaultIntervalCategoryDataset0.getColumnCount();
      assertEquals(1, defaultIntervalCategoryDataset0.getRowCount());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Number[][] numberArray0 = new Number[3][3];
      String[] stringArray0 = new String[3];
      stringArray0[0] = "";
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(stringArray0, stringArray0, numberArray0, numberArray0);
      defaultIntervalCategoryDataset0.getValue((Comparable) "", (Comparable) "");
      assertEquals(3, defaultIntervalCategoryDataset0.getSeriesCount());
      assertEquals(3, defaultIntervalCategoryDataset0.getCategoryCount());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Number[][] numberArray0 = new Number[11][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset1 = (DefaultIntervalCategoryDataset)defaultIntervalCategoryDataset0.clone();
      boolean boolean0 = defaultIntervalCategoryDataset0.equals(defaultIntervalCategoryDataset1);
      assertEquals(11, defaultIntervalCategoryDataset1.getRowCount());
      assertTrue(boolean0);
      assertNotSame(defaultIntervalCategoryDataset1, defaultIntervalCategoryDataset0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Number[][] numberArray0 = new Number[8][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      Day day0 = new Day();
      int int0 = defaultIntervalCategoryDataset0.getRowIndex(day0);
      assertEquals(8, defaultIntervalCategoryDataset0.getRowCount());
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Number[][] numberArray0 = new Number[16][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getRowKey(2204);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The 'row' argument is out of bounds.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Number[][] numberArray0 = new Number[3][3];
      String[] stringArray0 = new String[3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(stringArray0, numberArray0, numberArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.setStartValue(1, "", (Number) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.setValue: unrecognised category.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset((Number[][]) null, (Number[][]) null);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getColumnKey(4612);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Number[][] numberArray0 = new Number[2][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, (Number[][]) null);
      assertEquals(2, defaultIntervalCategoryDataset0.getSeriesCount());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Number[][] numberArray0 = new Number[9][3];
      Number[][] numberArray1 = new Number[1][4];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = null;
      try {
        defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray1);
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
      Number[][] numberArray0 = new Number[12][3];
      Comparable<XYDatasetTableModel>[] comparableArray0 = (Comparable<XYDatasetTableModel>[]) Array.newInstance(Comparable.class, 9);
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
      Number[][] numberArray0 = new Number[1][3];
      Number[][] numberArray1 = new Number[1][4];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = null;
      try {
        defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset: the number of categories in the start value dataset does not match the number of categories in the end value dataset.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Number[][] numberArray0 = new Number[9][3];
      Comparable<XYDatasetTableModel>[] comparableArray0 = (Comparable<XYDatasetTableModel>[]) Array.newInstance(Comparable.class, 9);
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
  public void test12()  throws Throwable  {
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset((Number[][]) null, (Number[][]) null);
      Week week0 = new Week();
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.setStartValue(1, week0, 53);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.setValue: series outside valid range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Number[][] numberArray0 = new Number[8][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getSeriesKey(335);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // No such series : 335
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Number[][] numberArray0 = new Number[11][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      Comparable comparable0 = defaultIntervalCategoryDataset0.getSeriesKey(9);
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
  public void test15()  throws Throwable  {
      Number[][] numberArray0 = new Number[2][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getSeriesKey((-2391));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // No such series : -2391
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Number[][] numberArray0 = new Number[1][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      assertEquals(1, defaultIntervalCategoryDataset0.getSeriesCount());
      assertEquals(1, defaultIntervalCategoryDataset0.getRowCount());
      
      Comparable<Integer>[] comparableArray0 = (Comparable<Integer>[]) Array.newInstance(Comparable.class, 1);
      defaultIntervalCategoryDataset0.setSeriesKeys(comparableArray0);
      assertEquals(3, defaultIntervalCategoryDataset0.getColumnCount());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Number[][] numberArray0 = new Number[11][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
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
      double[][] doubleArray0 = new double[20][0];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Comparable<DefaultIntervalCategoryDataset>[] comparableArray0 = (Comparable<DefaultIntervalCategoryDataset>[]) Array.newInstance(Comparable.class, 0);
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
      String[] stringArray0 = new String[3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(stringArray0, (Number[][]) null, (Number[][]) null);
      int int0 = defaultIntervalCategoryDataset0.getCategoryCount();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      double[][] doubleArray0 = new double[0][0];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      int int0 = defaultIntervalCategoryDataset0.getCategoryCount();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Number[][] numberArray0 = new Number[8][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      List list0 = defaultIntervalCategoryDataset0.getColumnKeys();
      assertEquals(8, defaultIntervalCategoryDataset0.getRowCount());
      assertFalse(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      double[][] doubleArray0 = new double[0][2];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      List list0 = defaultIntervalCategoryDataset0.getColumnKeys();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Number[][] numberArray0 = new Number[11][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      Comparable<String>[] comparableArray0 = (Comparable<String>[]) Array.newInstance(Comparable.class, 3);
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
  public void test24()  throws Throwable  {
      Number[][] numberArray0 = new Number[32][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
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
  public void test25()  throws Throwable  {
      Number[][] numberArray0 = new Number[4][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      Comparable<Object>[] comparableArray0 = (Comparable<Object>[]) Array.newInstance(Comparable.class, 0);
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
  public void test26()  throws Throwable  {
      Number[][] numberArray0 = new Number[1][3];
      Integer integer0 = JLayeredPane.DEFAULT_LAYER;
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
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
  public void test27()  throws Throwable  {
      Number[][] numberArray0 = new Number[8][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      Comparable comparable0 = defaultIntervalCategoryDataset0.getSeriesKey(1);
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
  public void test28()  throws Throwable  {
      Number[][] numberArray0 = new Number[3][3];
      String[] stringArray0 = new String[3];
      stringArray0[1] = "";
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(stringArray0, numberArray0, numberArray0);
      Comparable comparable0 = defaultIntervalCategoryDataset0.getSeriesKey(1);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getStartValue((Comparable) comparable0, (Comparable) "");
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
      Number[][] numberArray0 = new Number[1][3];
      Integer integer0 = JLayeredPane.DEFAULT_LAYER;
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getStartValue((Comparable) integer0, (Comparable) integer0);
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
      Number[][] numberArray0 = new Number[3][3];
      String[] stringArray0 = new String[3];
      stringArray0[0] = "";
      stringArray0[1] = "";
      stringArray0[2] = "52@Ua";
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(stringArray0, numberArray0, numberArray0);
      defaultIntervalCategoryDataset0.setCategoryKeys(stringArray0);
      Comparable comparable0 = defaultIntervalCategoryDataset0.getSeriesKey(1);
      Number number0 = defaultIntervalCategoryDataset0.getStartValue((Comparable) comparable0, (Comparable) "");
      assertNull(number0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Number[][] numberArray0 = new Number[19][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getStartValue((-1140), (-1140));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.getValue(): series index out of range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      double[][] doubleArray0 = new double[1][0];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getStartValue(366, 1);
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
      Number[][] numberArray0 = new Number[2][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getStartValue(0, (-1520));
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
      double[][] doubleArray0 = new double[1][0];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getStartValue(0, 0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.getValue(): category index out of range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Number[][] numberArray0 = new Number[3][3];
      String[] stringArray0 = new String[3];
      stringArray0[0] = "";
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(stringArray0, numberArray0, numberArray0);
      Comparable comparable0 = defaultIntervalCategoryDataset0.getSeriesKey(1);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getEndValue((Comparable) "", (Comparable) comparable0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'columnKey' argument.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Number[][] numberArray0 = new Number[1][3];
      Integer integer0 = JLayeredPane.DEFAULT_LAYER;
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
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
  public void test37()  throws Throwable  {
      Number[][] numberArray0 = new Number[3][3];
      String[] stringArray0 = new String[3];
      stringArray0[0] = "";
      stringArray0[1] = "";
      stringArray0[2] = "52@Ua";
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(stringArray0, numberArray0, numberArray0);
      defaultIntervalCategoryDataset0.setCategoryKeys(stringArray0);
      Comparable comparable0 = defaultIntervalCategoryDataset0.getSeriesKey(1);
      Number number0 = defaultIntervalCategoryDataset0.getEndValue((Comparable) "", (Comparable) comparable0);
      assertNull(number0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Number[][] numberArray0 = new Number[6][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getEndValue((-853), (-853));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.getValue(): series index out of range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      double[][] doubleArray0 = new double[20][0];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getValue(53, 53);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.getValue(): series index out of range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Number[][] numberArray0 = new Number[8][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getValue(1, (-1546));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.getValue(): category index out of range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      double[][] doubleArray0 = new double[5][0];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getEndValue(1, 53);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.getValue(): category index out of range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      double[][] doubleArray0 = new double[20][0];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Week week0 = new Week();
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.setStartValue((-1176), week0, 1);
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
      Number[][] numberArray0 = new Number[3][3];
      String[] stringArray0 = new String[3];
      stringArray0[0] = "";
      stringArray0[1] = "";
      stringArray0[2] = "52@Ua";
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(stringArray0, numberArray0, numberArray0);
      defaultIntervalCategoryDataset0.setCategoryKeys(stringArray0);
      defaultIntervalCategoryDataset0.setStartValue(1, "", (Number) null);
      assertEquals(3, defaultIntervalCategoryDataset0.getColumnCount());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Number[][] numberArray0 = new Number[9][3];
      Integer integer0 = JLayeredPane.DEFAULT_LAYER;
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.setEndValue((-1898), integer0, integer0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.setValue: series outside valid range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Number[][] numberArray0 = new Number[3][3];
      String[] stringArray0 = new String[3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(stringArray0, numberArray0, numberArray0);
      Integer integer0 = new Integer(1);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.setEndValue(1, "", integer0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.setValue: unrecognised category.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      double[][] doubleArray0 = new double[1][0];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Week week0 = new Week();
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.setEndValue(1, week0, 1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // DefaultIntervalCategoryDataset.setValue: series outside valid range.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Number[][] numberArray0 = new Number[3][3];
      String[] stringArray0 = new String[3];
      stringArray0[0] = "";
      Integer integer0 = new Integer(1);
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(stringArray0, stringArray0, numberArray0, numberArray0);
      defaultIntervalCategoryDataset0.setEndValue(1, "", integer0);
      assertEquals(3, defaultIntervalCategoryDataset0.getSeriesCount());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      double[][] doubleArray0 = new double[3][8];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      List list0 = defaultIntervalCategoryDataset0.getRowKeys();
      assertEquals(3, list0.size());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      double[][] doubleArray0 = new double[0][0];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      List list0 = defaultIntervalCategoryDataset0.getRowKeys();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Number[][] numberArray0 = new Number[7][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      Comparable comparable0 = defaultIntervalCategoryDataset0.getRowKey(1);
      assertNotNull(comparable0);
      assertEquals(7, defaultIntervalCategoryDataset0.getRowCount());
      assertEquals("Series 2", comparable0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      double[][] doubleArray0 = new double[3][8];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        defaultIntervalCategoryDataset0.getRowKey((-518));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The 'row' argument is out of bounds.
         //
         verifyException("org.jfree.data.category.DefaultIntervalCategoryDataset", e);
      }
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      double[][] doubleArray0 = new double[20][0];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      boolean boolean0 = defaultIntervalCategoryDataset0.equals(defaultIntervalCategoryDataset0);
      assertEquals(20, defaultIntervalCategoryDataset0.getRowCount());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      double[][] doubleArray0 = new double[1][0];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      Stack<String> stack0 = new Stack<String>();
      boolean boolean0 = defaultIntervalCategoryDataset0.equals(stack0);
      assertFalse(boolean0);
      assertEquals(1, defaultIntervalCategoryDataset0.getRowCount());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Number[][] numberArray0 = new Number[8][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset1 = new DefaultIntervalCategoryDataset((Number[][]) null, numberArray0);
      boolean boolean0 = defaultIntervalCategoryDataset0.equals(defaultIntervalCategoryDataset1);
      assertEquals(8, defaultIntervalCategoryDataset0.getRowCount());
      assertFalse(boolean0);
      assertEquals(0, defaultIntervalCategoryDataset1.getSeriesCount());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Number[][] numberArray0 = new Number[11][3];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(numberArray0, numberArray0);
      Object object0 = defaultIntervalCategoryDataset0.clone();
      assertTrue(object0.equals((Object)defaultIntervalCategoryDataset0));
      
      Comparable<String>[] comparableArray0 = (Comparable<String>[]) Array.newInstance(Comparable.class, 3);
      SerialDate serialDate0 = SerialDate.createInstance(473);
      comparableArray0[0] = (Comparable<String>) serialDate0;
      Year year0 = new Year();
      comparableArray0[1] = (Comparable<String>) year0;
      Week week0 = new Week(1, year0);
      comparableArray0[2] = (Comparable<String>) week0;
      defaultIntervalCategoryDataset0.setCategoryKeys(comparableArray0);
      boolean boolean0 = defaultIntervalCategoryDataset0.equals(object0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset((Number[][]) null, (Number[][]) null);
      Number[][] numberArray0 = new Number[0][5];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset1 = new DefaultIntervalCategoryDataset((Number[][]) null, numberArray0);
      boolean boolean0 = defaultIntervalCategoryDataset0.equals(defaultIntervalCategoryDataset1);
      assertFalse(defaultIntervalCategoryDataset1.equals((Object)defaultIntervalCategoryDataset0));
      assertFalse(boolean0);
  }
}
