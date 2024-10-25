/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:02:43 GMT 2023
 */

package org.jfree.data.statistics;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.awt.AWTKeyStroke;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Stack;
import java.util.Vector;
import javax.swing.JLayeredPane;
import javax.swing.JScrollPane;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jfree.data.KeyedObjects2D;
import org.jfree.data.Range;
import org.jfree.data.statistics.BoxAndWhiskerItem;
import org.jfree.data.statistics.DefaultBoxAndWhiskerCategoryDataset;
import org.jfree.data.time.FixedMillisecond;
import org.jfree.data.time.Month;
import org.jfree.data.time.Quarter;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DefaultBoxAndWhiskerCategoryDataset_ESTest extends DefaultBoxAndWhiskerCategoryDataset_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
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
      Integer integer0 = JLayeredPane.POPUP_LAYER;
      int int0 = defaultBoxAndWhiskerCategoryDataset0.getRowIndex(integer0);
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      // Undeclared exception!
      try { 
        defaultBoxAndWhiskerCategoryDataset0.getRowKey(1);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 1, Size: 0
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
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      // Undeclared exception!
      try { 
        defaultBoxAndWhiskerCategoryDataset0.getValue((Comparable) "JyvO';;B3-jWW]Up", (Comparable) "JyvO';;B3-jWW]Up");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Row key (JyvO';;B3-jWW]Up) not recognised.
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
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      List list0 = defaultBoxAndWhiskerCategoryDataset0.getRowKeys();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      // Undeclared exception!
      try { 
        defaultBoxAndWhiskerCategoryDataset0.getValue(3111, 3111);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 3111, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Range range0 = defaultBoxAndWhiskerCategoryDataset0.getRangeBounds(true);
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
        defaultBoxAndWhiskerCategoryDataset0.getColumnKey(483);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 483, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      double double0 = defaultBoxAndWhiskerCategoryDataset0.getRangeUpperBound(true);
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
      FixedMillisecond fixedMillisecond0 = new FixedMillisecond();
      ArrayList<Integer> arrayList0 = new ArrayList<Integer>();
      // Undeclared exception!
      try { 
        defaultBoxAndWhiskerCategoryDataset0.add((List) arrayList0, (Comparable) fixedMillisecond0, (Comparable) fixedMillisecond0);
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
        defaultBoxAndWhiskerCategoryDataset0.getItem((-86), (-86));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]");
      assertEquals(1, defaultBoxAndWhiskerCategoryDataset0.getRowCount());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.DEFAULT_LAYER;
      Vector<KeyedObjects2D> vector0 = new Vector<KeyedObjects2D>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem(integer0, (Number) null, integer0, integer0, integer0, integer0, (Number) null, (Number) null, vector0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      assertEquals(1, defaultBoxAndWhiskerCategoryDataset0.getRowCount());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      ArrayList<JScrollPane> arrayList0 = new ArrayList<JScrollPane>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) arrayList0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Integer integer1 = JLayeredPane.MODAL_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem1 = new BoxAndWhiskerItem((Number) integer1, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer1, (Number) integer1, (List) arrayList0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem1, (Comparable) integer1, (Comparable) integer1);
      assertEquals(2, defaultBoxAndWhiskerCategoryDataset0.getColumnCount());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.DRAG_LAYER;
      Vector<KeyedObjects2D> vector0 = new Vector<KeyedObjects2D>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) vector0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Integer integer1 = JLayeredPane.MODAL_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem1 = new BoxAndWhiskerItem((Number) integer1, (Number) integer0, (Number) integer1, (Number) integer1, (Number) integer0, (Number) integer0, (Number) integer1, (Number) integer0, (List) vector0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem1, (Comparable) integer1, (Comparable) integer0);
      assertEquals(1, defaultBoxAndWhiskerCategoryDataset0.getColumnCount());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]", (Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]");
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMeanValue(0, 1);
      assertNull(number0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMeanValue(0, 0);
      assertEquals(100, number0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]", (Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]");
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMeanValue((Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]", (Comparable) integer0);
      assertNull(number0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMeanValue((Comparable) integer0, (Comparable) integer0);
      assertEquals(100, number0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMedianValue(0, 0);
      assertEquals(100, number0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.MODAL_LAYER;
      Stack<AWTKeyStroke> stack0 = new Stack<AWTKeyStroke>();
      Integer integer1 = JLayeredPane.FRAME_CONTENT_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem(integer0, integer0, integer0, integer0, (Number) null, (Number) null, integer1, integer1, stack0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Quarter quarter0 = new Quarter();
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer1, (Comparable) quarter0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMedianValue((Comparable) integer1, (Comparable) integer0);
      assertNull(number0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.FRAME_CONTENT_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMedianValue((Comparable) integer0, (Comparable) integer0);
      assertEquals((-30000), number0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      Vector<KeyedObjects2D> vector0 = new Vector<KeyedObjects2D>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) vector0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getQ1Value(0, 0);
      assertEquals(100, number0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]", (Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]");
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getQ1Value((Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]", (Comparable) integer0);
      assertNull(number0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getQ1Value((Comparable) integer0, (Comparable) integer0);
      assertEquals(100, number0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getQ3Value(0, 0);
      assertEquals(100, number0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.POPUP_LAYER;
      Stack<KeyedObjects2D> stack0 = new Stack<KeyedObjects2D>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) stack0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Integer integer1 = JLayeredPane.FRAME_CONTENT_LAYER;
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer1, (Comparable) integer1);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getQ3Value((Comparable) integer0, (Comparable) integer1);
      assertNull(number0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]", (Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]");
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getQ3Value((Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]", (Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]");
      assertEquals(100, number0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]", (Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]");
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMinRegularValue(1, 0);
      assertNull(number0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMinRegularValue(0, 0);
      assertEquals(100, number0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.MODAL_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Integer integer1 = JLayeredPane.DRAG_LAYER;
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer1, (Comparable) integer1);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMinRegularValue((Comparable) integer1, (Comparable) integer0);
      assertNull(number0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      Stack<KeyedObjects2D> stack0 = new Stack<KeyedObjects2D>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) stack0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMinRegularValue((Comparable) integer0, (Comparable) integer0);
      assertEquals(100, number0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      Vector<KeyedObjects2D> vector0 = new Vector<KeyedObjects2D>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) vector0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMaxRegularValue(0, 0);
      assertEquals(100, number0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      Locale.FilteringMode locale_FilteringMode0 = Locale.FilteringMode.MAP_EXTENDED_RANGES;
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) locale_FilteringMode0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) locale_FilteringMode0, (Comparable) integer0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMaxRegularValue((Comparable) integer0, (Comparable) integer0);
      assertNull(number0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.MODAL_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMaxRegularValue((Comparable) integer0, (Comparable) integer0);
      assertEquals(200, number0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      Vector<KeyedObjects2D> vector0 = new Vector<KeyedObjects2D>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) vector0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMinOutlier(0, 0);
      assertEquals(100, number0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]", (Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]");
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMinOutlier((Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]", (Comparable) integer0);
      assertNull(number0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.FRAME_CONTENT_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMinOutlier((Comparable) integer0, (Comparable) integer0);
      assertEquals((-30000), number0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      Vector<KeyedObjects2D> vector0 = new Vector<KeyedObjects2D>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) vector0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMaxOutlier(0, 0);
      assertEquals(100, number0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]", (Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]");
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMaxOutlier((Comparable) integer0, (Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]");
      assertNull(number0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.FRAME_CONTENT_LAYER;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      Number number0 = defaultBoxAndWhiskerCategoryDataset0.getMaxOutlier((Comparable) integer0, (Comparable) integer0);
      assertEquals((-30000), number0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      Stack<KeyedObjects2D> stack0 = new Stack<KeyedObjects2D>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) stack0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      List list0 = defaultBoxAndWhiskerCategoryDataset0.getOutliers(0, 0);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      BigInteger bigInteger0 = BigInteger.ONE;
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem(integer0, integer0, integer0, integer0, bigInteger0, integer0, bigInteger0, bigInteger0, (List) null);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) bigInteger0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) "org.jfree.data.statistics.BoxAndWhiskerItem@5[mean=100,median=100,q1=100,q3=100]", (Comparable) integer0);
      List list0 = defaultBoxAndWhiskerCategoryDataset0.getOutliers((Comparable) integer0, (Comparable) integer0);
      assertNull(list0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      Vector<Locale.Category> vector0 = new Vector<Locale.Category>();
      BoxAndWhiskerItem boxAndWhiskerItem0 = new BoxAndWhiskerItem((Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (Number) integer0, (List) vector0);
      defaultBoxAndWhiskerCategoryDataset0.add(boxAndWhiskerItem0, (Comparable) integer0, (Comparable) integer0);
      List list0 = defaultBoxAndWhiskerCategoryDataset0.getOutliers((Comparable) integer0, (Comparable) integer0);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset1 = new DefaultBoxAndWhiskerCategoryDataset();
      boolean boolean0 = defaultBoxAndWhiskerCategoryDataset0.equals(defaultBoxAndWhiskerCategoryDataset1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      boolean boolean0 = defaultBoxAndWhiskerCategoryDataset0.equals(defaultBoxAndWhiskerCategoryDataset0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Month month0 = new Month();
      DefaultBoxAndWhiskerCategoryDataset defaultBoxAndWhiskerCategoryDataset0 = new DefaultBoxAndWhiskerCategoryDataset();
      boolean boolean0 = defaultBoxAndWhiskerCategoryDataset0.equals(month0);
      assertFalse(boolean0);
  }
}
