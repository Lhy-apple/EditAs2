/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 01:53:29 GMT 2023
 */

package org.jfree.data;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.jfree.chart.util.SortOrder;
import org.jfree.data.DefaultKeyedValues;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DefaultKeyedValues_ESTest extends DefaultKeyedValues_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      Comparable<Object> comparable0 = (Comparable<Object>) mock(Comparable.class, new ViolatedAssumptionAnswer());
      doReturn((String) null, (String) null).when(comparable0).toString();
      defaultKeyedValues0.setValue(comparable0, 1.0);
      Object object0 = defaultKeyedValues0.clone();
      boolean boolean0 = defaultKeyedValues0.equals(object0);
      assertNotSame(object0, defaultKeyedValues0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      Float float0 = Float.valueOf((float) 0);
      defaultKeyedValues0.insertValue(0, (Comparable) float0, (double) 0);
      assertEquals(1, defaultKeyedValues0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      List list0 = defaultKeyedValues0.getKeys();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      // Undeclared exception!
      try { 
        defaultKeyedValues0.removeValue((Comparable) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'key' argument.
         //
         verifyException("org.jfree.data.DefaultKeyedValues", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      Double double0 = new Double((-1.0));
      defaultKeyedValues0.setValue((Comparable) double0, (Number) double0);
      Number number0 = defaultKeyedValues0.getValue((Comparable) double0);
      assertEquals((-1.0), number0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      // Undeclared exception!
      try { 
        defaultKeyedValues0.getValue((Comparable) 1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Key not found: 1
         //
         verifyException("org.jfree.data.DefaultKeyedValues", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      // Undeclared exception!
      try { 
        defaultKeyedValues0.addValue((Comparable) null, 2472.6924449);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'key' argument.
         //
         verifyException("org.jfree.data.DefaultKeyedValues", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      Long long0 = Long.valueOf((long) (-15));
      // Undeclared exception!
      try { 
        defaultKeyedValues0.insertValue((-15), (Comparable) long0, (double) (-15));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 'position' out of bounds.
         //
         verifyException("org.jfree.data.DefaultKeyedValues", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      Float float0 = Float.valueOf((float) 2076);
      // Undeclared exception!
      try { 
        defaultKeyedValues0.insertValue(2076, (Comparable) float0, (double) 2076);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 'position' out of bounds.
         //
         verifyException("org.jfree.data.DefaultKeyedValues", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      // Undeclared exception!
      try { 
        defaultKeyedValues0.insertValue(0, (Comparable) null, (double) 0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'key' argument.
         //
         verifyException("org.jfree.data.DefaultKeyedValues", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      Double double0 = new Double(0.0);
      defaultKeyedValues0.setValue((Comparable) double0, (Number) double0);
      defaultKeyedValues0.insertValue(0, (Comparable) double0, (Number) double0);
      assertEquals(1, defaultKeyedValues0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      Double double0 = new Double(0.0);
      defaultKeyedValues0.setValue((Comparable) double0, (Number) double0);
      // Undeclared exception!
      try { 
        defaultKeyedValues0.insertValue(1, (Comparable) double0, 0.0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 1, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      Long long0 = new Long((-1L));
      defaultKeyedValues0.setValue((Comparable) long0, (Number) long0);
      defaultKeyedValues0.removeValue((Comparable) long0);
      assertEquals(1, defaultKeyedValues0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      Byte byte0 = new Byte((byte)94);
      defaultKeyedValues0.setValue((Comparable) byte0, (double) (byte)94);
      Double double0 = new Double(411.22580580173);
      defaultKeyedValues0.setValue((Comparable) double0, (Number) byte0);
      defaultKeyedValues0.removeValue((Comparable) byte0);
      assertEquals(1, defaultKeyedValues0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      Long long0 = new Long((-304L));
      defaultKeyedValues0.removeValue((Comparable) long0);
      assertEquals(0, defaultKeyedValues0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      SortOrder sortOrder0 = SortOrder.DESCENDING;
      defaultKeyedValues0.setValue((Comparable) 0, 970.94401760092);
      defaultKeyedValues0.sortByKeys(sortOrder0);
      assertEquals(1, defaultKeyedValues0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      Double double0 = new Double(0.0);
      defaultKeyedValues0.setValue((Comparable) double0, (Number) double0);
      SortOrder sortOrder0 = SortOrder.ASCENDING;
      defaultKeyedValues0.sortByValues(sortOrder0);
      assertEquals(1, defaultKeyedValues0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      boolean boolean0 = defaultKeyedValues0.equals(defaultKeyedValues0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      Long long0 = new Long((-304L));
      boolean boolean0 = defaultKeyedValues0.equals(long0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      Double double0 = new Double(0.0);
      defaultKeyedValues0.setValue((Comparable) double0, (Number) double0);
      DefaultKeyedValues defaultKeyedValues1 = new DefaultKeyedValues();
      boolean boolean0 = defaultKeyedValues0.equals(defaultKeyedValues1);
      assertFalse(boolean0);
      assertFalse(defaultKeyedValues1.equals((Object)defaultKeyedValues0));
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      Comparable<Object> comparable0 = (Comparable<Object>) mock(Comparable.class, new ViolatedAssumptionAnswer());
      doReturn((String) null).when(comparable0).toString();
      defaultKeyedValues0.setValue(comparable0, 1.0);
      Double double0 = new Double(1.0);
      DefaultKeyedValues defaultKeyedValues1 = new DefaultKeyedValues();
      defaultKeyedValues1.addValue((Comparable) double0, (Number) double0);
      boolean boolean0 = defaultKeyedValues1.equals(defaultKeyedValues0);
      assertFalse(defaultKeyedValues1.equals((Object)defaultKeyedValues0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      Double double0 = new Double((-1.0));
      defaultKeyedValues0.setValue((Comparable) double0, (Number) null);
      Object object0 = defaultKeyedValues0.clone();
      boolean boolean0 = defaultKeyedValues0.equals(object0);
      assertNotSame(object0, defaultKeyedValues0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      Double double0 = new Double((-1.0));
      defaultKeyedValues0.setValue((Comparable) double0, (Number) null);
      DefaultKeyedValues defaultKeyedValues1 = (DefaultKeyedValues)defaultKeyedValues0.clone();
      assertTrue(defaultKeyedValues1.equals((Object)defaultKeyedValues0));
      
      defaultKeyedValues1.addValue((Comparable) double0, (Number) double0);
      boolean boolean0 = defaultKeyedValues0.equals(defaultKeyedValues1);
      assertFalse(boolean0);
      assertFalse(defaultKeyedValues1.equals((Object)defaultKeyedValues0));
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      Double double0 = new Double(0.0);
      defaultKeyedValues0.setValue((Comparable) double0, (Number) double0);
      DefaultKeyedValues defaultKeyedValues1 = (DefaultKeyedValues)defaultKeyedValues0.clone();
      assertTrue(defaultKeyedValues1.equals((Object)defaultKeyedValues0));
      
      defaultKeyedValues1.setValue((Comparable) double0, (-1106.6546));
      boolean boolean0 = defaultKeyedValues0.equals(defaultKeyedValues1);
      assertFalse(defaultKeyedValues1.equals((Object)defaultKeyedValues0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      DefaultKeyedValues defaultKeyedValues0 = new DefaultKeyedValues();
      defaultKeyedValues0.hashCode();
  }
}