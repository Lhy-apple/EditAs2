/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:20:09 GMT 2023
 */

package org.jfree.data.statistics;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.math.BigInteger;
import java.time.LocalDate;
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;
import java.util.Vector;
import javax.swing.JLayeredPane;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.time.MockLocalDate;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.jfree.data.KeyedObjects2D;
import org.jfree.data.Range;
import org.jfree.data.statistics.BoxAndWhiskerItem;
import org.jfree.data.statistics.DefaultBoxAndWhiskerCategoryDataset;
import org.jfree.data.time.FixedMillisecond;
import org.jfree.data.time.Hour;
import org.jfree.data.time.Millisecond;
import org.jfree.data.time.Month;
import org.jfree.data.time.SimpleTimePeriod;
import org.jfree.data.time.Week;
import org.jfree.data.xy.XYDatasetTableModel;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DefaultBoxAndWhiskerCategoryDataset_ESTest extends DefaultBoxAndWhiskerCategoryDataset_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Stack<XYDatasetTableModel> stack0 = new Stack<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 0, (Number) 23, (Number) 23, (Number) 23, (Number) 0, (Number) 23, (Number) Double.NaN, (Number) 0, (List) stack0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      assertEquals(1, defaultBoxAndWhiskerCategoryDataset0.getColumnCount());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      int int0 = defaultBoxAndWhiskerCategoryDataset0.getColumnIndex((Comparable) null);
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      int int0 = defaultBoxAndWhiskerCategoryDataset0.getRowIndex((Comparable) null);
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      // Undeclared exception!
      try { 
        defaultBoxAndWhiskerCategoryDataset0.getRowKey(2399);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 2399, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      double double0 = defaultBoxAndWhiskerCategoryDataset0.getRangeLowerBound(false);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Month month0 = new Month();
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      // Undeclared exception!
      try { 
        defaultBoxAndWhiskerCategoryDataset0.getValue((Comparable) month0, (Comparable) month0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Row key (February 2014) not recognised.
         //
         verifyException("org.jfree.data.KeyedObjects2D", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      int int0 = defaultBoxAndWhiskerCategoryDataset0.getRowCount();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      List list0 = defaultBoxAndWhiskerCategoryDataset0.getColumnKeys();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      List list0 = defaultBoxAndWhiskerCategoryDataset0.getRowKeys();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      // Undeclared exception!
      try { 
        defaultBoxAndWhiskerCategoryDataset0.getValue(0, 59);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 0, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Range range0 = defaultBoxAndWhiskerCategoryDataset0.getRangeBounds(false);
      assertEquals(0.0, range0.getLength(), 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      int int0 = defaultBoxAndWhiskerCategoryDataset0.getColumnCount();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      // Undeclared exception!
      try { 
        defaultBoxAndWhiskerCategoryDataset0.getColumnKey(3062);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 3062, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      double double0 = defaultBoxAndWhiskerCategoryDataset0.getRangeUpperBound(false);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Object object0 = defaultBoxAndWhiskerCategoryDataset0.clone();
      assertNotSame(object0, defaultBoxAndWhiskerCategoryDataset0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      FixedMillisecond fixedMillisecond0 = new FixedMillisecond();
      // Undeclared exception!
      try { 
        defaultBoxAndWhiskerCategoryDataset0.add((List) linkedList0, (Comparable) fixedMillisecond0, (Comparable) fixedMillisecond0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Range(double, double): require lower (Infinity) <= upper (-Infinity).
         //
         verifyException("org.jfree.data.Range", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      // Undeclared exception!
      try { 
        defaultBoxAndWhiskerCategoryDataset0.getItem(0, 0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 0, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Stack<XYDatasetTableModel> stack0 = new Stack<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 0, (Number) 0, (Number) 23, (Number) 23, (Number) 0, (Number) 23, (Number) 0, (Number) 23, (List) stack0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) 0);
      assertEquals(1, defaultBoxAndWhiskerCategoryDataset0.getRowCount());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Stack<XYDatasetTableModel> stack0 = new Stack<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) null, 23, (Number) null, (Number) null, (Number) null, 0, (Number) null, 23, stack0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      assertEquals(1, defaultBoxAndWhiskerCategoryDataset0.getColumnCount());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Stack<XYDatasetTableModel> stack0 = new Stack<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem(0, 23, 23, 23, 23, 23, 0, (Number) null, stack0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      MockDate mockDate0 = new MockDate();
      SimpleTimePeriod simpleTimePeriod0 = new SimpleTimePeriod(mockDate0, mockDate0);
      List list0 = defaultBoxAndWhiskerCategoryDataset0.getOutliers((Comparable) hour0, (Comparable) simpleTimePeriod0);
      assertNull(list0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Vector<XYDatasetTableModel> vector0 = new Vector<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 23, (Number) 0, (Number) 23, (Number) 0, (Number) 23, (Number) 0, (Number) 0, (Number) 0, (List) vector0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      BigInteger bigInteger0 = BigInteger.TEN;
      BoxAndWhiskerItem boxAndWhiskerItem1 = new BoxAndWhiskerItem(23, 0, 0, 0, 0, 0, bigInteger0, 23, vector0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem1, (Comparable) bigInteger0, (Comparable) hour0);
      assertEquals(1, defaultBoxAndWhiskerCategoryDataset0.getColumnCount());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((double) 0, 790.1754397075808, 790.1754397075808, (double) 0, 790.1754397075808, (double) 0, (double) 0, (double) 0, (List) null);
      BigInteger bigInteger0 = BigInteger.ONE;
      BoxAndWhiskerItem boxAndWhiskerItem1 = new BoxAndWhiskerItem(bigInteger0, bigInteger0, bigInteger0, bigInteger0, bigInteger0, bigInteger0, bigInteger0, bigInteger0, (List) null);
      Week week0 = new Week();
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem1, (Comparable) week0, (Comparable) week0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) bigInteger0, (Comparable) bigInteger0);
      assertEquals((short)1, bigInteger0.shortValue());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Stack<XYDatasetTableModel> stack0 = new Stack<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 23, (Number) 0, (Number) 23, (Number) 0, (Number) 23, (Number) 0, (Number) 0, (Number) 0, (List) stack0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMeanValue(0, 0);
      assertEquals(23, number0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((double) 0, 790.1754397075808, 790.1754397075808, (double) 0, 790.1754397075808, 0.5063277786140752, 0.5063277786140752, 0.5063277786140752, (List) null);
      Millisecond millisecond0 = new Millisecond();
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) millisecond0, (Comparable) millisecond0);
      Double double0 = (Double)defaultBoxAndWhiskerCategoryDataset0.getMinRegularValue((Comparable) millisecond0, (Comparable) millisecond0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) double0, (Comparable) double0);
      Double double1 = (Double)defaultBoxAndWhiskerCategoryDataset0.getMeanValue((Comparable) double0, (Comparable) millisecond0);
      assertNull(double1);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      LinkedList<KeyedObjects2D> linkedList0 = new LinkedList<KeyedObjects2D>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 23, (Number) 0, (Number) 0, (Number) 23, (Number) 23, (Number) 0, (Number) 23, (Number) 23, (List) linkedList0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMeanValue((Comparable) hour0, (Comparable) hour0);
      assertEquals(23, number0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Stack<XYDatasetTableModel> stack0 = new Stack<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 0, (Number) 23, (Number) 23, (Number) 0, (Number) 23, (Number) 23, (Number) 0, (Number) 23, (List) stack0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMedianValue(0, 0);
      assertEquals(23, number0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((double) 0, 790.1754397075808, 790.1754397075808, (double) 0, 790.1754397075808, 790.1754397075808, 790.1754397075808, 790.1754397075808, (List) null);
      Millisecond millisecond0 = new Millisecond();
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) millisecond0, (Comparable) millisecond0);
      Double double0 = (Double)defaultBoxAndWhiskerCategoryDataset0.getMinRegularValue((Comparable) millisecond0, (Comparable) millisecond0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) double0, (Comparable) double0);
      Double double1 = (Double)defaultBoxAndWhiskerCategoryDataset0.getMedianValue((Comparable) millisecond0, (Comparable) double0);
      assertNull(double1);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((double) 0, (double) 0, (double) 0, (double) 0, (double) 0, (double) 0, (double) 0, (double) 0, (List) null);
      Millisecond millisecond0 = new Millisecond();
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) millisecond0, (Comparable) millisecond0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMedianValue((Comparable) millisecond0, (Comparable) millisecond0);
      assertEquals(0.0, number0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Vector<XYDatasetTableModel> vector0 = new Vector<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 23, (Number) 0, (Number) 23, (Number) 0, (Number) 23, (Number) 0, (Number) 0, (Number) 0, (List) vector0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getQ1Value(0, 0);
      assertEquals(23, number0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Vector<XYDatasetTableModel> vector0 = new Vector<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 0, (Number) 23, (Number) 0, (Number) 23, (Number) 23, (Number) 0, (Number) 0, (Number) 23, (List) vector0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0.FIRST_HOUR_IN_DAY, (Comparable) hour0.FIRST_HOUR_IN_DAY);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getQ1Value((Comparable) hour0.FIRST_HOUR_IN_DAY, (Comparable) hour0);
      assertNull(number0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Vector<XYDatasetTableModel> vector0 = new Vector<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 0, (Number) 0, (Number) 0, (Number) 0, (Number) 23, (Number) 23, (Number) 0, (Number) 0, (List) vector0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getQ1Value((Comparable) hour0, (Comparable) hour0);
      assertEquals(0, number0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Stack<XYDatasetTableModel> stack0 = new Stack<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 23, (Number) 0, (Number) 23, (Number) 0, (Number) 23, (Number) 0, (Number) 0, (Number) 0, (List) stack0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getQ3Value(0, 0);
      assertEquals(0, number0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((double) 0, 790.1754397075808, 790.1754397075808, (double) 0, 790.1754397075808, 790.1754397075808, 790.1754397075808, 790.1754397075808, (List) null);
      Millisecond millisecond0 = new Millisecond();
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) millisecond0, (Comparable) millisecond0);
      Double double0 = (Double)defaultBoxAndWhiskerCategoryDataset0.getMinRegularValue((Comparable) millisecond0, (Comparable) millisecond0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) double0, (Comparable) double0);
      Double double1 = (Double)defaultBoxAndWhiskerCategoryDataset0.getQ3Value((Comparable) double0, (Comparable) millisecond0);
      assertNull(double1);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      LinkedList<KeyedObjects2D> linkedList0 = new LinkedList<KeyedObjects2D>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 23, (Number) 0, (Number) 0, (Number) 23, (Number) 23, (Number) 0, (Number) 23, (Number) 23, (List) linkedList0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getQ3Value((Comparable) hour0, (Comparable) hour0);
      assertEquals(23, number0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Vector<XYDatasetTableModel> vector0 = new Vector<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 0, (Number) 23, (Number) 0, (Number) 23, (Number) 23, (Number) 0, (Number) 0, (Number) 23, (List) vector0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMinRegularValue(0, 0);
      assertEquals(23, number0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Stack<XYDatasetTableModel> stack0 = new Stack<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 0, (Number) 0, (Number) 23, (Number) 0, (Number) 0, (Number) 0, (Number) 0, (Number) 0, (List) stack0);
      Week week0 = new Week();
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) week0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) week0, (Comparable) hour0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMinRegularValue((Comparable) hour0, (Comparable) hour0);
      assertNull(number0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((double) 0, (double) 0, (double) 0, (double) 0, (double) 0, (double) 0, (double) 0, (double) 0, (List) null);
      Millisecond millisecond0 = new Millisecond();
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) millisecond0, (Comparable) millisecond0);
      Double double0 = (Double)defaultBoxAndWhiskerCategoryDataset0.getMaxRegularValue(0, 0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) double0, (Comparable) double0);
      Double double1 = (Double)defaultBoxAndWhiskerCategoryDataset0.getMaxOutlier((Comparable) double0, (Comparable) millisecond0);
      assertNull(double1);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Stack<XYDatasetTableModel> stack0 = new Stack<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 0, (Number) 23, (Number) 0, (Number) 23, (Number) 23, (Number) 0, (Number) 0, (Number) 23, (List) stack0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      Integer integer0 = JLayeredPane.DRAG_LAYER;
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMaxRegularValue((Comparable) hour0, (Comparable) integer0);
      assertNull(number0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Vector<XYDatasetTableModel> vector0 = new Vector<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 0, (Number) 23, (Number) 0, (Number) 23, (Number) 23, (Number) 0, (Number) 0, (Number) 23, (List) vector0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMaxRegularValue((Comparable) hour0, (Comparable) hour0);
      assertEquals(0, number0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Stack<XYDatasetTableModel> stack0 = new Stack<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 23, (Number) 0, (Number) 23, (Number) 0, (Number) 23, (Number) 0, (Number) 0, (Number) 0, (List) stack0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMinOutlier(0, 0);
      assertEquals(0, number0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Vector<XYDatasetTableModel> vector0 = new Vector<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 0, (Number) 23, (Number) 0, (Number) 23, (Number) 23, (Number) 0, (Number) 0, (Number) 23, (List) vector0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMinOutlier((Comparable) hour0, (Comparable) hour0);
      assertEquals(0, number0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Vector<XYDatasetTableModel> vector0 = new Vector<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 0, (Number) 23, (Number) 0, (Number) 23, (Number) 23, (Number) 0, (Number) 0, (Number) 23, (List) vector0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMaxOutlier(0, 0);
      assertEquals(23, number0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Vector<XYDatasetTableModel> vector0 = new Vector<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 0, (Number) 23, (Number) 0, (Number) 23, (Number) 23, (Number) 0, (Number) 0, (Number) 23, (List) vector0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMaxOutlier((Comparable) hour0, (Comparable) hour0);
      assertEquals(23, number0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((double) 0, 790.1754397075808, 790.1754397075808, (double) 0, 790.1754397075808, (double) 0, (double) 0, (double) 0, (List) null);
      Millisecond millisecond0 = new Millisecond();
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) millisecond0, (Comparable) millisecond0);
      Double double0 = (Double)defaultBoxAndWhiskerCategoryDataset0.getMinRegularValue((Comparable) millisecond0, (Comparable) millisecond0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) double0, (Comparable) double0);
      List list0 = defaultBoxAndWhiskerCategoryDataset0.getOutliers(0, 1);
      assertNull(list0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Hour hour0 = new Hour();
      Vector<XYDatasetTableModel> vector0 = new Vector<XYDatasetTableModel>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) 23, (Number) 0, (Number) 23, (Number) 0, (Number) 23, (Number) 0, (Number) 0, (Number) 0, (List) vector0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) hour0, (Comparable) hour0);
      List list0 = defaultBoxAndWhiskerCategoryDataset0.getOutliers(0, 0);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((double) 0, 790.1754397075808, 790.1754397075808, (double) 0, 790.1754397075808, 790.1754397075808, 790.1754397075808, 790.1754397075808, (List) null);
      Millisecond millisecond0 = new Millisecond();
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) millisecond0, (Comparable) millisecond0);
      Double double0 = (Double)defaultBoxAndWhiskerCategoryDataset0.getMinRegularValue((Comparable) millisecond0, (Comparable) millisecond0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) double0, (Comparable) double0);
      List list0 = defaultBoxAndWhiskerCategoryDataset0.getOutliers((Comparable) double0, (Comparable) double0);
      assertNull(list0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset1 = new DefaultBoxAndWhiskerCategoryDataset();
      boolean boolean0 = defaultBoxAndWhiskerCategoryDataset0.equals(defaultBoxAndWhiskerCategoryDataset1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      boolean boolean0 = defaultBoxAndWhiskerCategoryDataset0.equals(defaultBoxAndWhiskerCategoryDataset0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      LocalDate localDate0 = MockLocalDate.now();
      boolean boolean0 = defaultBoxAndWhiskerCategoryDataset0.equals(localDate0);
      assertFalse(boolean0);
  }
}