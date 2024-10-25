/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:42:52 GMT 2023
 */

package org.joda.time.base;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.joda.time.DateMidnight;
import org.joda.time.DurationFieldType;
import org.joda.time.Hours;
import org.joda.time.LocalDate;
import org.joda.time.Minutes;
import org.joda.time.MonthDay;
import org.joda.time.Months;
import org.joda.time.Period;
import org.joda.time.ReadableInstant;
import org.joda.time.ReadablePartial;
import org.joda.time.ReadablePeriod;
import org.joda.time.Seconds;
import org.joda.time.Weeks;
import org.joda.time.YearMonth;
import org.joda.time.Years;
import org.joda.time.base.BaseSingleFieldPeriod;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BaseSingleFieldPeriod_ESTest extends BaseSingleFieldPeriod_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Months months0 = Months.SEVEN;
      Period period0 = new Period(months0);
      boolean boolean0 = months0.equals(period0);
      assertEquals(7, months0.getMonths());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Months months0 = Months.FOUR;
      months0.hashCode();
      assertEquals(4, months0.getMonths());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Months months0 = Months.FOUR;
      months0.toMutablePeriod();
      assertEquals(4, months0.getMonths());
      assertEquals(1, months0.size());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      // Undeclared exception!
      try { 
        Months.monthsBetween((ReadableInstant) null, (ReadableInstant) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadableInstant objects must not be null
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DateMidnight dateMidnight0 = DateMidnight.now();
      Minutes minutes0 = Minutes.minutesBetween((ReadableInstant) dateMidnight0, (ReadableInstant) dateMidnight0);
      assertEquals(0, minutes0.getMinutes());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DateMidnight dateMidnight0 = DateMidnight.now();
      // Undeclared exception!
      try { 
        Minutes.minutesBetween((ReadableInstant) dateMidnight0, (ReadableInstant) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadableInstant objects must not be null
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      // Undeclared exception!
      try { 
        Months.monthsBetween((ReadablePartial) null, (ReadablePartial) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadablePartial objects must not be null
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MonthDay monthDay0 = new MonthDay();
      YearMonth yearMonth0 = new YearMonth(1, 1);
      // Undeclared exception!
      try { 
        Months.monthsBetween((ReadablePartial) monthDay0, (ReadablePartial) yearMonth0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadablePartial objects must have the same set of fields
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MonthDay monthDay0 = new MonthDay();
      // Undeclared exception!
      try { 
        Months.monthsBetween((ReadablePartial) monthDay0, (ReadablePartial) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadablePartial objects must not be null
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MonthDay monthDay0 = new MonthDay();
      LocalDate localDate0 = new LocalDate();
      // Undeclared exception!
      try { 
        Years.yearsBetween((ReadablePartial) monthDay0, (ReadablePartial) localDate0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadablePartial objects must have the same set of fields
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MonthDay monthDay0 = new MonthDay();
      Weeks weeks0 = Weeks.weeksBetween((ReadablePartial) monthDay0, (ReadablePartial) monthDay0);
      assertEquals(0, weeks0.getWeeks());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Months months0 = Months.TEN;
      Period period0 = months0.toPeriod();
      // Undeclared exception!
      try { 
        Seconds.standardSecondsIn(period0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Cannot convert period to duration as months is not precise in the period P10M
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Seconds seconds0 = Seconds.standardSecondsIn((ReadablePeriod) null);
      assertEquals(0, seconds0.getSeconds());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Hours hours0 = Hours.SEVEN;
      Minutes minutes0 = Minutes.standardMinutesIn(hours0);
      assertEquals(7, hours0.getHours());
      assertEquals(420, minutes0.getMinutes());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Months months0 = Months.FIVE;
      // Undeclared exception!
      try { 
        months0.getFieldType((-1545));
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // -1545
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Months months0 = Months.FOUR;
      // Undeclared exception!
      try { 
        months0.getValue(5069);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // 5069
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Months months0 = Months.SEVEN;
      DurationFieldType durationFieldType0 = DurationFieldType.millis();
      int int0 = months0.get(durationFieldType0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Months months0 = Months.SEVEN;
      DurationFieldType durationFieldType0 = DurationFieldType.months();
      int int0 = months0.get(durationFieldType0);
      assertEquals(7, int0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Months months0 = Months.EIGHT;
      DurationFieldType durationFieldType0 = DurationFieldType.days();
      boolean boolean0 = months0.isSupported(durationFieldType0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Months months0 = Months.ONE;
      DurationFieldType durationFieldType0 = months0.getFieldType();
      boolean boolean0 = months0.isSupported(durationFieldType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Months months0 = Months.FIVE;
      boolean boolean0 = months0.equals(months0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      MonthDay monthDay0 = new MonthDay();
      Months months0 = Months.EIGHT;
      boolean boolean0 = months0.equals(monthDay0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Months months0 = Months.EIGHT;
      Years years0 = Years.ZERO;
      boolean boolean0 = months0.equals(years0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Months months0 = Months.SEVEN;
      Months months1 = Months.EIGHT;
      boolean boolean0 = months1.equals(months0);
      assertEquals(8, months1.getMonths());
      assertFalse(boolean0);
      assertFalse(months0.equals((Object)months1));
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Years years0 = Years.ZERO;
      int int0 = years0.compareTo((BaseSingleFieldPeriod) years0);
      assertEquals(0, int0);
      assertEquals(0, years0.getYears());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Years years0 = Years.ZERO;
      Hours hours0 = Hours.ONE;
      // Undeclared exception!
      try { 
        years0.compareTo((BaseSingleFieldPeriod) hours0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // class org.joda.time.Years cannot be compared to class org.joda.time.Hours
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Years years0 = Years.ZERO;
      Years years1 = Years.MIN_VALUE;
      int int0 = years0.compareTo((BaseSingleFieldPeriod) years1);
      assertEquals(Integer.MIN_VALUE, years1.getYears());
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Years years0 = Years.ZERO;
      Years years1 = Years.MAX_VALUE;
      int int0 = years0.compareTo((BaseSingleFieldPeriod) years1);
      assertEquals(Integer.MAX_VALUE, years1.getYears());
      assertEquals((-1), int0);
  }
}
