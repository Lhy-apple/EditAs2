/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:39:08 GMT 2023
 */

package org.joda.time.base;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.joda.time.Chronology;
import org.joda.time.DateMidnight;
import org.joda.time.DateTime;
import org.joda.time.DateTimeZone;
import org.joda.time.Duration;
import org.joda.time.DurationFieldType;
import org.joda.time.LocalDateTime;
import org.joda.time.LocalTime;
import org.joda.time.MonthDay;
import org.joda.time.MutablePeriod;
import org.joda.time.Period;
import org.joda.time.PeriodType;
import org.joda.time.ReadableInstant;
import org.joda.time.ReadablePartial;
import org.joda.time.ReadablePeriod;
import org.joda.time.YearMonthDay;
import org.joda.time.Years;
import org.joda.time.chrono.BuddhistChronology;
import org.joda.time.chrono.GJChronology;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BasePeriod_ESTest extends BasePeriod_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Duration duration0 = Duration.standardHours(0L);
      Period period0 = duration0.toPeriodFrom((ReadableInstant) null);
      assertEquals(8, period0.size());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Duration duration0 = Duration.ZERO;
      PeriodType periodType0 = PeriodType.yearMonthDayTime();
      Period period0 = new Period(duration0, periodType0);
      assertEquals(7, period0.size());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      MonthDay monthDay0 = new MonthDay();
      PeriodType periodType0 = PeriodType.yearDayTime();
      Period period0 = new Period(monthDay0, monthDay0, periodType0);
      MutablePeriod mutablePeriod0 = period0.toMutablePeriod();
      // Undeclared exception!
      try { 
        mutablePeriod0.addWeeks(1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Period does not support field 'weeks'
         //
         verifyException("org.joda.time.base.BasePeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PeriodType periodType0 = PeriodType.years();
      Period period0 = null;
      try {
        period0 = new Period(1, 0, 1, 0, 1, 0, 0, 0, periodType0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Period does not support field 'weeks'
         //
         verifyException("org.joda.time.base.BasePeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Duration duration0 = Duration.standardSeconds(0L);
      MutablePeriod mutablePeriod0 = new MutablePeriod(duration0, (ReadableInstant) null, (PeriodType) null);
      assertEquals(8, mutablePeriod0.size());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MutablePeriod mutablePeriod0 = new MutablePeriod(1293L, 1293L);
      Duration duration0 = mutablePeriod0.toDurationTo((ReadableInstant) null);
      assertEquals(0L, duration0.getMillis());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PeriodType periodType0 = PeriodType.hours();
      MutablePeriod mutablePeriod0 = new MutablePeriod(periodType0);
      // Undeclared exception!
      try { 
        mutablePeriod0.set((DurationFieldType) null, 0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Period does not support field 'null'
         //
         verifyException("org.joda.time.base.BasePeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MutablePeriod mutablePeriod0 = new MutablePeriod(618L);
      Years years0 = Years.TWO;
      mutablePeriod0.mergePeriod(years0);
      assertEquals(1, years0.size());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PeriodType periodType0 = PeriodType.hours();
      MutablePeriod mutablePeriod0 = new MutablePeriod(periodType0);
      DateMidnight dateMidnight0 = new DateMidnight();
      Duration duration0 = mutablePeriod0.toDurationFrom(dateMidnight0);
      assertEquals(0L, duration0.getStandardSeconds());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MutablePeriod mutablePeriod0 = null;
      try {
        mutablePeriod0 = new MutablePeriod("America/Anchorage");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid format: \"America/Anchorage\"
         //
         verifyException("org.joda.time.format.PeriodFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PeriodType periodType0 = PeriodType.hours();
      MutablePeriod mutablePeriod0 = new MutablePeriod(periodType0);
      // Undeclared exception!
      try { 
        mutablePeriod0.add(2808, 2808, 2808, 2808, 2808, 2808, 2808, 2808);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Period does not support field 'years'
         //
         verifyException("org.joda.time.base.BasePeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Period period0 = Period.years(1);
      assertEquals(8, period0.size());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PeriodType periodType0 = PeriodType.yearWeekDayTime();
      DateMidnight dateMidnight0 = new DateMidnight();
      Period period0 = new Period(dateMidnight0, dateMidnight0, periodType0);
      assertEquals(7, period0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance((DateTimeZone) null);
      DateTimeZone dateTimeZone0 = gJChronology0.getZone();
      DateTime dateTime0 = new DateTime((long) 0, dateTimeZone0);
      Period period0 = new Period((ReadableInstant) null, dateTime0);
      assertEquals(8, period0.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PeriodType periodType0 = PeriodType.months();
      MutablePeriod mutablePeriod0 = new MutablePeriod((ReadableInstant) null, (ReadableInstant) null, periodType0);
      assertEquals(1, mutablePeriod0.size());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      PeriodType periodType0 = PeriodType.dayTime();
      Period period0 = null;
      try {
        period0 = new Period((ReadablePartial) null, (ReadablePartial) null, periodType0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadablePartial objects must not be null
         //
         verifyException("org.joda.time.base.BasePeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      PeriodType periodType0 = PeriodType.years();
      YearMonthDay yearMonthDay0 = new YearMonthDay(1743L);
      Period period0 = null;
      try {
        period0 = new Period(yearMonthDay0, (ReadablePartial) null, periodType0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadablePartial objects must not be null
         //
         verifyException("org.joda.time.base.BasePeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      PeriodType periodType0 = PeriodType.yearWeekDayTime();
      LocalTime localTime0 = new LocalTime();
      LocalDateTime localDateTime0 = new LocalDateTime();
      Period period0 = null;
      try {
        period0 = new Period(localDateTime0, localTime0, periodType0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadablePartial objects must have the same set of fields
         //
         verifyException("org.joda.time.base.BasePeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      PeriodType periodType0 = PeriodType.seconds();
      MonthDay monthDay0 = new MonthDay((-3811L));
      LocalTime localTime0 = new LocalTime(1, 0, 1, 1);
      Period period0 = null;
      try {
        period0 = new Period(localTime0, monthDay0, periodType0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadablePartial objects must have the same set of fields
         //
         verifyException("org.joda.time.base.BasePeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      PeriodType periodType0 = PeriodType.minutes();
      LocalTime localTime0 = new LocalTime();
      Period period0 = new Period(localTime0, localTime0, periodType0);
      assertEquals(1, period0.size());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      PeriodType periodType0 = PeriodType.months();
      Period period0 = new Period((Object) null, periodType0);
      assertEquals(1, period0.size());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      PeriodType periodType0 = PeriodType.yearDayTime();
      MutablePeriod mutablePeriod0 = new MutablePeriod(periodType0);
      mutablePeriod0.setYears(0);
      assertEquals(6, mutablePeriod0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      PeriodType periodType0 = PeriodType.hours();
      MutablePeriod mutablePeriod0 = new MutablePeriod(periodType0);
      // Undeclared exception!
      try { 
        mutablePeriod0.setWeeks((-6));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Period does not support field 'weeks'
         //
         verifyException("org.joda.time.base.BasePeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      PeriodType periodType0 = PeriodType.hours();
      MutablePeriod mutablePeriod0 = new MutablePeriod(periodType0);
      mutablePeriod0.setYears(0);
      assertEquals(1, mutablePeriod0.size());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      PeriodType periodType0 = PeriodType.yearDayTime();
      MutablePeriod mutablePeriod0 = new MutablePeriod(periodType0);
      mutablePeriod0.addMinutes(0);
      assertEquals(6, mutablePeriod0.size());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      PeriodType periodType0 = PeriodType.hours();
      MutablePeriod mutablePeriod0 = new MutablePeriod(periodType0);
      mutablePeriod0.addMinutes(0);
      assertEquals(1, mutablePeriod0.size());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      PeriodType periodType0 = PeriodType.hours();
      MutablePeriod mutablePeriod0 = new MutablePeriod(periodType0);
      // Undeclared exception!
      try { 
        mutablePeriod0.add((DurationFieldType) null, 0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Period does not support field 'null'
         //
         verifyException("org.joda.time.base.BasePeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      MutablePeriod mutablePeriod0 = new MutablePeriod(1293L, 1293L);
      mutablePeriod0.mergePeriod((ReadablePeriod) null);
      assertEquals(8, mutablePeriod0.size());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      PeriodType periodType0 = PeriodType.yearDayTime();
      MutablePeriod mutablePeriod0 = new MutablePeriod(periodType0);
      mutablePeriod0.add((ReadablePeriod) null);
      assertEquals(6, mutablePeriod0.size());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      PeriodType periodType0 = PeriodType.hours();
      MutablePeriod mutablePeriod0 = new MutablePeriod(periodType0);
      Years years0 = Years.years(80);
      // Undeclared exception!
      try { 
        mutablePeriod0.add((ReadablePeriod) years0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Period does not support field 'years'
         //
         verifyException("org.joda.time.base.BasePeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      MutablePeriod mutablePeriod0 = new MutablePeriod();
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstanceUTC();
      mutablePeriod0.add((long) 1036, (Chronology) buddhistChronology0);
      assertEquals(1, BuddhistChronology.BE);
  }
}
