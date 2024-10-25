/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:30:15 GMT 2023
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
import org.evosuite.runtime.mock.java.util.MockDate;
import org.jfree.data.time.RegularTimePeriod;
import org.jfree.data.time.Week;
import org.jfree.data.time.Year;
import org.junit.runner.RunWith;
import sun.util.calendar.ZoneInfo;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Week_ESTest extends Week_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Week week0 = new Week();
      MockDate mockDate0 = new MockDate(53, 53, 1, 53, 53);
      Week week1 = new Week(mockDate0, week0.DEFAULT_TIME_ZONE);
      assertEquals((-396986820001L), week1.getLastMillisecond());
      assertEquals(23, week1.getWeek());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Week week0 = new Week(0, 0);
      week0.hashCode();
      assertEquals(1392409281319L, week0.getLastMillisecond());
      assertEquals(0L, week0.getSerialIndex());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Week week0 = new Week();
      int int0 = week0.compareTo(week0);
      assertEquals(0, int0);
      assertEquals(1392409281319L, week0.getLastMillisecond());
      assertEquals(106749L, week0.getSerialIndex());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Week week0 = new Week();
      long long0 = week0.getMiddleMillisecond();
      assertEquals(7, week0.getWeek());
      assertEquals(1392409281320L, long0);
      assertEquals(1392409281319L, week0.getLastMillisecond());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
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
  public void test05()  throws Throwable  {
      Year year0 = new Year();
      Week week0 = new Week((-489), year0);
      int int0 = week0.getYearValue();
      assertEquals(2014, int0);
      assertEquals(1392409281319L, week0.getLastMillisecond());
      assertEquals(23, week0.getWeek());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Week week0 = new Week();
      String string0 = week0.toString();
      assertEquals(1392409281319L, week0.getLastMillisecond());
      assertEquals("Week 7, 2014", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Year year0 = new Year();
      Week week0 = new Week((-489), year0);
      long long0 = week0.getSerialIndex();
      assertEquals(1392409281319L, week0.getLastMillisecond());
      assertEquals(106765L, long0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Year year0 = new Year();
      Week week0 = new Week(1, year0);
      RegularTimePeriod regularTimePeriod0 = week0.previous();
      assertEquals(106741L, regularTimePeriod0.getSerialIndex());
      assertNotNull(regularTimePeriod0);
      assertEquals(1392409281319L, regularTimePeriod0.getLastMillisecond());
      assertEquals(1, week0.getWeek());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
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
  public void test10()  throws Throwable  {
      MockDate mockDate0 = new MockDate(53L);
      Locale locale0 = Locale.TAIWAN;
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
  public void test11()  throws Throwable  {
      MockDate mockDate0 = new MockDate((-9223372036854775808L));
      ZoneInfo zoneInfo0 = (ZoneInfo)RegularTimePeriod.DEFAULT_TIME_ZONE;
      Week week0 = null;
      try {
        week0 = new Week(mockDate0, zoneInfo0, (Locale) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'locale' argument.
         //
         verifyException("org.jfree.data.time.Week", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MockDate mockDate0 = new MockDate(0, 0, 0, 0, 0);
      Week week0 = new Week(mockDate0);
      RegularTimePeriod regularTimePeriod0 = week0.previous();
      assertNull(regularTimePeriod0);
      assertEquals((-2209075200001L), week0.getLastMillisecond());
      assertEquals(100701L, week0.getSerialIndex());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      MockDate mockDate0 = new MockDate(0L);
      Week week0 = new Week(mockDate0);
      assertEquals(104411L, week0.getSerialIndex());
      assertEquals((-1L), week0.getLastMillisecond());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Week week0 = new Week(0, 0);
      RegularTimePeriod regularTimePeriod0 = week0.previous();
      assertEquals((-1L), regularTimePeriod0.getSerialIndex());
      assertEquals(1392409281319L, regularTimePeriod0.getLastMillisecond());
      assertNotNull(regularTimePeriod0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Week week0 = new Week(120, 9999);
      RegularTimePeriod regularTimePeriod0 = week0.next();
      assertNull(regularTimePeriod0);
      assertEquals(1392409281319L, week0.getLastMillisecond());
      assertEquals(120, week0.getWeek());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Week week0 = new Week();
      RegularTimePeriod regularTimePeriod0 = week0.next();
      assertEquals(106750L, regularTimePeriod0.getSerialIndex());
      assertEquals(1392409281319L, regularTimePeriod0.getLastMillisecond());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Week week0 = new Week(1588, 1588);
      RegularTimePeriod regularTimePeriod0 = week0.next();
      assertEquals(84217L, regularTimePeriod0.getSerialIndex());
      assertEquals(1392409281319L, regularTimePeriod0.getLastMillisecond());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Week week0 = new Week((-459), 3584);
      Week week1 = (Week)week0.next();
      assertEquals(1392409281319L, week1.getLastMillisecond());
      assertEquals(1, week1.getWeek());
      assertNotNull(week1);
      assertEquals(190006L, week1.getSerialIndex());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Week week0 = new Week();
      Week week1 = new Week(1, 1);
      week0.equals(week1);
      assertEquals(106749L, week0.getSerialIndex());
      assertEquals(54L, week1.getSerialIndex());
      assertEquals(1392409281319L, week1.getLastMillisecond());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Week week0 = new Week();
      week0.equals(week0);
      assertEquals(106749L, week0.getSerialIndex());
      assertEquals(1392409281319L, week0.getLastMillisecond());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Week week0 = new Week();
      Object object0 = new Object();
      boolean boolean0 = week0.equals(object0);
      assertFalse(boolean0);
      assertEquals(7, week0.getWeek());
      assertEquals(1392409281319L, week0.getLastMillisecond());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Week week0 = new Week();
      Week week1 = new Week();
      boolean boolean0 = week0.equals(week1);
      assertTrue(boolean0);
      assertEquals(1392409281319L, week1.getLastMillisecond());
      assertEquals(7, week1.getWeek());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Week week0 = new Week((-459), 3584);
      Week week1 = new Week(53, 53);
      boolean boolean0 = week1.equals(week0);
      assertFalse(boolean0);
      assertEquals(190005L, week0.getSerialIndex());
      assertEquals(1392409281319L, week1.getLastMillisecond());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Year year0 = new Year();
      Week week0 = new Week();
      int int0 = week0.compareTo(year0);
      assertEquals(0, int0);
      assertEquals(1392409281319L, week0.getLastMillisecond());
      assertEquals(7, week0.getWeek());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Week week0 = new Week();
      Week week1 = new Week(53, 53);
      int int0 = week1.compareTo(week0);
      assertEquals(2862L, week1.getSerialIndex());
      assertEquals(1392409281319L, week1.getLastMillisecond());
      assertEquals((-1961), int0);
      assertEquals(7, week0.getWeek());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Week week0 = new Week((-1287), (-1287));
      Object object0 = new Object();
      int int0 = week0.compareTo(object0);
      assertEquals((-68218L), week0.getSerialIndex());
      assertEquals(1, int0);
      assertEquals(1392409281319L, week0.getLastMillisecond());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Week week0 = Week.parseWeek((String) null);
      assertNull(week0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
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
  public void test29()  throws Throwable  {
      // Undeclared exception!
      try { 
        Week.parseWeek("8#\",s+\"2[aHjMv2YJ");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can't evaluate the year.
         //
         verifyException("org.jfree.data.time.Week", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      // Undeclared exception!
      try { 
        Week.parseWeek("R2K`^aD,[h6d-Vh{3NN");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can't evaluate the year.
         //
         verifyException("org.jfree.data.time.Week", e);
      }
  }
}
