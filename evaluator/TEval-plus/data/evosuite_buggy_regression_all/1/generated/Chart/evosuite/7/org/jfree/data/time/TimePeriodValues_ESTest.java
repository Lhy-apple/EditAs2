/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 12:57:00 GMT 2023
 */

package org.jfree.data.time;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.System;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.jfree.data.time.FixedMillisecond;
import org.jfree.data.time.Quarter;
import org.jfree.data.time.TimePeriod;
import org.jfree.data.time.TimePeriodValue;
import org.jfree.data.time.TimePeriodValues;
import org.jfree.data.time.Week;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TimePeriodValues_ESTest extends TimePeriodValues_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TimePeriodValues timePeriodValues0 = new TimePeriodValues("Q1/2014");
      TimePeriodValues timePeriodValues1 = (TimePeriodValues)timePeriodValues0.clone();
      boolean boolean0 = timePeriodValues0.equals(timePeriodValues1);
      assertEquals("Value", timePeriodValues1.getRangeDescription());
      assertEquals((-1), timePeriodValues1.getMinEndIndex());
      assertEquals("Time", timePeriodValues1.getDomainDescription());
      assertEquals((-1), timePeriodValues1.getMinMiddleIndex());
      assertEquals((-1), timePeriodValues1.getMinStartIndex());
      assertTrue(boolean0);
      assertEquals((-1), timePeriodValues1.getMaxMiddleIndex());
      assertEquals((-1), timePeriodValues1.getMaxEndIndex());
      assertEquals((-1), timePeriodValues1.getMaxStartIndex());
      assertNotSame(timePeriodValues1, timePeriodValues0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TimePeriodValues timePeriodValues0 = new TimePeriodValues("Q1/2014");
      int int0 = timePeriodValues0.getMaxMiddleIndex();
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals("Time", timePeriodValues0.getDomainDescription());
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals("Value", timePeriodValues0.getRangeDescription());
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
      assertEquals((-1), int0);
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues("Q1/2014");
      timePeriodValues0.add((TimePeriod) quarter0, (double) 4);
      Object object0 = timePeriodValues0.clone();
      boolean boolean0 = timePeriodValues0.equals(object0);
      assertEquals(1, timePeriodValues0.getItemCount());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Week week0 = new Week();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(week0);
      // Undeclared exception!
      try { 
        timePeriodValues0.getValue((-2570));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(quarter0);
      timePeriodValues0.add((TimePeriod) quarter0, (Number) 4);
      timePeriodValues0.delete(4, 1);
      assertEquals(0, timePeriodValues0.getMinStartIndex());
      assertEquals(0, timePeriodValues0.getMaxEndIndex());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      FixedMillisecond fixedMillisecond0 = new FixedMillisecond();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(fixedMillisecond0);
      int int0 = timePeriodValues0.getMinEndIndex();
      assertEquals("Value", timePeriodValues0.getRangeDescription());
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals((-1), int0);
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
      assertEquals("Time", timePeriodValues0.getDomainDescription());
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TimePeriodValues timePeriodValues0 = new TimePeriodValues("v1/>8s *syPtjh");
      int int0 = timePeriodValues0.getMaxStartIndex();
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals((-1), int0);
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals("Value", timePeriodValues0.getRangeDescription());
      assertEquals("Time", timePeriodValues0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(quarter0);
      int int0 = timePeriodValues0.getMinStartIndex();
      assertEquals((-1), int0);
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
      assertEquals("Time", timePeriodValues0.getDomainDescription());
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
      assertEquals("Value", timePeriodValues0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(quarter0);
      // Undeclared exception!
      try { 
        timePeriodValues0.getTimePeriod(1);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 1, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(quarter0);
      int int0 = timePeriodValues0.getMaxEndIndex();
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
      assertEquals((-1), int0);
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
      assertEquals("Value", timePeriodValues0.getRangeDescription());
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals("Time", timePeriodValues0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MockDate mockDate0 = new MockDate();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(mockDate0);
      int int0 = timePeriodValues0.getMinMiddleIndex();
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals("Time", timePeriodValues0.getDomainDescription());
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
      assertEquals("Value", timePeriodValues0.getRangeDescription());
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(quarter0);
      // Undeclared exception!
      try { 
        timePeriodValues0.add((TimePeriodValue) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null item not allowed.
         //
         verifyException("org.jfree.data.time.TimePeriodValues", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(quarter0);
      timePeriodValues0.add((TimePeriod) quarter0, 1.0);
      System.setCurrentTimeMillis((-2435L));
      FixedMillisecond fixedMillisecond0 = new FixedMillisecond();
      timePeriodValues0.add((TimePeriod) fixedMillisecond0, (Number) 4);
      assertEquals(1, timePeriodValues0.getMinEndIndex());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues("Q1/2014");
      MockDate mockDate0 = new MockDate((-2241), 45, 1, (-1692), (-1692));
      FixedMillisecond fixedMillisecond0 = new FixedMillisecond(mockDate0);
      timePeriodValues0.add((TimePeriod) fixedMillisecond0, (Number) 1);
      timePeriodValues0.add((TimePeriod) quarter0, (Number) 4);
      assertEquals(1, timePeriodValues0.getMaxEndIndex());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(quarter0);
      timePeriodValues0.delete(4, 1);
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
      assertEquals("Value", timePeriodValues0.getRangeDescription());
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals("Time", timePeriodValues0.getDomainDescription());
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Week week0 = new Week();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(week0);
      // Undeclared exception!
      try { 
        timePeriodValues0.delete(1, 53);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 1, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Week week0 = new Week();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(week0);
      boolean boolean0 = timePeriodValues0.equals(timePeriodValues0);
      assertEquals("Value", timePeriodValues0.getRangeDescription());
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
      assertTrue(boolean0);
      assertEquals("Time", timePeriodValues0.getDomainDescription());
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues("Q1/2014");
      TimePeriodValue timePeriodValue0 = new TimePeriodValue((TimePeriod) quarter0, (Number) 1);
      boolean boolean0 = timePeriodValues0.equals(timePeriodValue0);
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
      assertEquals("Time", timePeriodValues0.getDomainDescription());
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertFalse(boolean0);
      assertEquals("Value", timePeriodValues0.getRangeDescription());
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues("Q1/2014");
      TimePeriodValues timePeriodValues1 = new TimePeriodValues(quarter0);
      boolean boolean0 = timePeriodValues1.equals(timePeriodValues0);
      assertFalse(boolean0);
      assertEquals((-1), timePeriodValues1.getMaxMiddleIndex());
      assertEquals((-1), timePeriodValues1.getMaxEndIndex());
      assertEquals((-1), timePeriodValues1.getMinEndIndex());
      assertEquals((-1), timePeriodValues1.getMaxStartIndex());
      assertEquals((-1), timePeriodValues1.getMinStartIndex());
      assertEquals("Time", timePeriodValues1.getDomainDescription());
      assertEquals("Value", timePeriodValues1.getRangeDescription());
      assertEquals((-1), timePeriodValues1.getMinMiddleIndex());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Week week0 = new Week();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(week0);
      TimePeriodValues timePeriodValues1 = new TimePeriodValues(week0);
      timePeriodValues1.setDomainDescription("");
      boolean boolean0 = timePeriodValues1.equals(timePeriodValues0);
      assertEquals("", timePeriodValues1.getDomainDescription());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Week week0 = new Week();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(week0);
      TimePeriodValues timePeriodValues1 = new TimePeriodValues(week0);
      timePeriodValues0.setRangeDescription("");
      boolean boolean0 = timePeriodValues1.equals(timePeriodValues0);
      assertEquals("", timePeriodValues0.getRangeDescription());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(quarter0);
      timePeriodValues0.add((TimePeriod) quarter0, (Number) 4);
      TimePeriodValues timePeriodValues1 = timePeriodValues0.createCopy(4, 1);
      boolean boolean0 = timePeriodValues0.equals(timePeriodValues1);
      assertEquals(1, timePeriodValues0.getItemCount());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues("Q1/2014");
      timePeriodValues0.add((TimePeriod) quarter0, (double) 4);
      TimePeriodValues timePeriodValues1 = (TimePeriodValues)timePeriodValues0.clone();
      timePeriodValues1.update(0, 1);
      boolean boolean0 = timePeriodValues1.equals(timePeriodValues0);
      assertEquals(1, timePeriodValues0.getItemCount());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(quarter0, (String) null, "Not enough valid columns where generated by query.");
      timePeriodValues0.hashCode();
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals("Not enough valid columns where generated by query.", timePeriodValues0.getRangeDescription());
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      TimePeriodValues timePeriodValues0 = new TimePeriodValues("Q1/2014");
      timePeriodValues0.hashCode();
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
      assertEquals("Time", timePeriodValues0.getDomainDescription());
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
      assertEquals("Value", timePeriodValues0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(quarter0, "Q1/2014", (String) null);
      timePeriodValues0.hashCode();
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals("Q1/2014", timePeriodValues0.getDomainDescription());
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
  }
}
