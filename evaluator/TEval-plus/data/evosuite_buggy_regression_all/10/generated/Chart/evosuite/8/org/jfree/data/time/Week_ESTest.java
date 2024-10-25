/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 12:19:03 GMT 2023
 */

package org.jfree.data.time;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Date;
import java.util.Locale;
import java.util.TimeZone;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.System;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.jfree.data.time.Month;
import org.jfree.data.time.RegularTimePeriod;
import org.jfree.data.time.Week;
import org.jfree.data.time.Year;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Week_ESTest extends Week_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Week week0 = new Week();
      MockDate mockDate0 = new MockDate(1);
      Week week1 = new Week(mockDate0, week0.DEFAULT_TIME_ZONE);
      int int0 = week0.compareTo(week1);
      assertEquals(1392409281319L, week0.getLastMillisecond());
      assertEquals(44, int0);
      assertEquals(1, week1.getWeek());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Week week0 = new Week();
      week0.hashCode();
      assertEquals(1392409281319L, week0.getLastMillisecond());
      assertEquals(106749L, week0.getSerialIndex());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Week week0 = new Week();
      int int0 = week0.compareTo(week0);
      assertEquals(0, int0);
      assertEquals(1392409281319L, week0.getLastMillisecond());
      assertEquals(7, week0.getWeek());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      // Undeclared exception!
      try { 
        Week.parseWeek("Week 7, 2014");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can't evaluate the week.
         //
         verifyException("org.jfree.data.time.Week", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Week week0 = new Week(0, 0);
      int int0 = week0.getYearValue();
      assertEquals(1392409281319L, week0.getLastMillisecond());
      assertEquals(0L, week0.getSerialIndex());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Week week0 = new Week();
      String string0 = week0.toString();
      assertEquals("Week 7, 2014", string0);
      assertEquals(1392409281319L, week0.getLastMillisecond());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Week week0 = new Week();
      long long0 = week0.getSerialIndex();
      assertEquals(1392409281319L, week0.getLastMillisecond());
      assertEquals(106749L, long0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Week week0 = new Week();
      long long0 = week0.getMiddleMillisecond();
      assertEquals(1392409281320L, long0);
      assertEquals(7, week0.getWeek());
      assertEquals(1392409281319L, week0.getLastMillisecond());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Week week0 = null;
      try {
        week0 = new Week(2958465, (Year) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jfree.data.time.Week", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MockDate mockDate0 = new MockDate(4069, 4069, 4069, 4069, 4069, 4069);
      Year year0 = new Year(mockDate0);
      Week week0 = new Week((-416), year0);
      assertEquals(335003L, week0.getSerialIndex());
      assertEquals(1392409281319L, week0.getLastMillisecond());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Week week0 = null;
      try {
        week0 = new Week((Date) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'time' argument.
         //
         verifyException("org.jfree.data.time.Week", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      MockDate mockDate0 = new MockDate(0L);
      Locale locale0 = Locale.PRC;
      Week week0 = null;
      try {
        week0 = new Week(mockDate0, (TimeZone) null, locale0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'zone' argument.
         //
         verifyException("org.jfree.data.time.Week", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MockDate mockDate0 = new MockDate();
      TimeZone timeZone0 = TimeZone.getDefault();
      Week week0 = null;
      try {
        week0 = new Week(mockDate0, timeZone0, (Locale) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'locale' argument.
         //
         verifyException("org.jfree.data.time.Week", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      MockDate mockDate0 = new MockDate(0, 0, 0, 0, 0, 0);
      Week week0 = new Week(mockDate0);
      assertEquals((-2209075200001L), week0.getLastMillisecond());
      assertEquals(100701L, week0.getSerialIndex());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      System.setCurrentTimeMillis(0);
      Week week0 = new Week();
      RegularTimePeriod regularTimePeriod0 = week0.previous();
      assertEquals(104409L, regularTimePeriod0.getSerialIndex());
      assertEquals((-1L), regularTimePeriod0.getLastMillisecond());
      assertEquals(1, week0.getWeek());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Week week0 = new Week();
      RegularTimePeriod regularTimePeriod0 = week0.previous();
      assertEquals(106748L, regularTimePeriod0.getSerialIndex());
      assertEquals(1392409281319L, regularTimePeriod0.getLastMillisecond());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Week week0 = new Week((-460), (-460));
      RegularTimePeriod regularTimePeriod0 = week0.next();
      RegularTimePeriod regularTimePeriod1 = regularTimePeriod0.previous();
      assertNull(regularTimePeriod1);
      assertEquals(1392409281319L, regularTimePeriod0.getLastMillisecond());
      assertEquals(52, week0.getWeek());
      assertEquals((-24326L), regularTimePeriod0.getSerialIndex());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Week week0 = new Week((-460), 0);
      RegularTimePeriod regularTimePeriod0 = week0.next();
      assertEquals(53L, regularTimePeriod0.getSerialIndex());
      assertEquals(1392409281319L, regularTimePeriod0.getLastMillisecond());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Week week0 = new Week(53, 9999);
      RegularTimePeriod regularTimePeriod0 = week0.next();
      assertEquals(530000L, week0.getSerialIndex());
      assertNull(regularTimePeriod0);
      assertEquals(1392409281319L, week0.getLastMillisecond());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Week week0 = new Week();
      boolean boolean0 = week0.equals("Quarter outside valid range.");
      assertEquals(1392409281319L, week0.getLastMillisecond());
      assertFalse(boolean0);
      assertEquals(106749L, week0.getSerialIndex());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Week week0 = new Week(3680, 3680);
      boolean boolean0 = week0.equals(week0);
      assertEquals(1392409281319L, week0.getLastMillisecond());
      assertTrue(boolean0);
      assertEquals(195136L, week0.getSerialIndex());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Week week0 = new Week(0, 0);
      RegularTimePeriod regularTimePeriod0 = week0.next();
      boolean boolean0 = week0.equals(regularTimePeriod0);
      assertEquals(1392409281319L, regularTimePeriod0.getLastMillisecond());
      assertEquals(1L, regularTimePeriod0.getSerialIndex());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Week week0 = new Week(0, 0);
      Week week1 = new Week(0, 53);
      boolean boolean0 = week1.equals(week0);
      assertEquals(2809L, week1.getSerialIndex());
      assertFalse(boolean0);
      assertEquals(1392409281319L, week1.getLastMillisecond());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Week week0 = new Week(394, 394);
      Week week1 = new Week(394, 394);
      boolean boolean0 = week0.equals(week1);
      assertEquals(1392409281319L, week1.getLastMillisecond());
      assertEquals(20764L, week1.getSerialIndex());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Week week0 = new Week(0, 0);
      Object object0 = new Object();
      int int0 = week0.compareTo(object0);
      assertEquals(0L, week0.getSerialIndex());
      assertEquals(1, int0);
      assertEquals(1392409281319L, week0.getLastMillisecond());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Week week0 = new Week(0, 0);
      Month month0 = new Month(1, 53);
      int int0 = week0.compareTo(month0);
      assertEquals(0L, week0.getSerialIndex());
      assertEquals(0, int0);
      assertEquals(1392409281319L, week0.getLastMillisecond());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Week week0 = Week.parseWeek((String) null);
      assertNull(week0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      // Undeclared exception!
      try { 
        Week.parseWeek("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Could not find separator.
         //
         verifyException("org.jfree.data.time.Week", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      // Undeclared exception!
      try { 
        Week.parseWeek("Quarter outside valid range.");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can't evaluate the year.
         //
         verifyException("org.jfree.data.time.Week", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      // Undeclared exception!
      try { 
        Week.parseWeek("The 'week' argument must be in the range 1 - 53.");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can't evaluate the year.
         //
         verifyException("org.jfree.data.time.Week", e);
      }
  }
}
