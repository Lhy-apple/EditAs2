/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 01:00:09 GMT 2023
 */

package org.joda.time;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.joda.time.Chronology;
import org.joda.time.DateTime;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.joda.time.Duration;
import org.joda.time.DurationFieldType;
import org.joda.time.Hours;
import org.joda.time.Instant;
import org.joda.time.Interval;
import org.joda.time.LocalDate;
import org.joda.time.LocalDateTime;
import org.joda.time.LocalTime;
import org.joda.time.Minutes;
import org.joda.time.Months;
import org.joda.time.MutableDateTime;
import org.joda.time.MutablePeriod;
import org.joda.time.Partial;
import org.joda.time.Period;
import org.joda.time.PeriodType;
import org.joda.time.ReadableInstant;
import org.joda.time.ReadablePartial;
import org.joda.time.ReadablePeriod;
import org.joda.time.Seconds;
import org.joda.time.Weeks;
import org.joda.time.Years;
import org.joda.time.chrono.EthiopicChronology;
import org.joda.time.chrono.GJChronology;
import org.joda.time.chrono.GregorianChronology;
import org.joda.time.chrono.LenientChronology;
import org.joda.time.tz.FixedDateTimeZone;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Period_ESTest extends Period_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      LocalTime localTime0 = new LocalTime();
      Period period0 = Period.fieldDifference(localTime0, localTime0);
      // Undeclared exception!
      try { 
        period0.withDays((-1797));
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Field is not supported
         //
         verifyException("org.joda.time.PeriodType", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Period period0 = new Period(0, 0, 0, 0);
      Period period1 = period0.withMinutes(4124);
      assertNotSame(period0, period1);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstanceUTC();
      Period period0 = new Period(634L, (Chronology) ethiopicChronology0);
      Period period1 = period0.minusMillis(1);
      assertNotSame(period1, period0);
      assertFalse(period1.equals((Object)period0));
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Period period0 = Period.millis(1144);
      Period period1 = period0.ZERO.withHours((-1493));
      assertNotSame(period0, period1);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Period period0 = new Period(0L, 0L);
      Period period1 = period0.withWeeks(561);
      assertNotSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      Period period0 = Period.fieldDifference(localDateTime0, localDateTime0);
      // Undeclared exception!
      try { 
        period0.withSeconds(0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Field is not supported
         //
         verifyException("org.joda.time.PeriodType", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Period period0 = new Period(0, 0, 0, 0);
      Period period1 = period0.negated();
      assertNotSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      // Undeclared exception!
      try { 
        Period.parse("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid format: \"\"
         //
         verifyException("org.joda.time.format.PeriodFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      LocalTime localTime0 = new LocalTime();
      Period period0 = new Period(localTime0, localTime0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      GregorianChronology gregorianChronology0 = GregorianChronology.getInstanceUTC();
      LenientChronology lenientChronology0 = LenientChronology.getInstance(gregorianChronology0);
      Period period0 = null;
      try {
        period0 = new Period("([.", lenientChronology0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid format: \"([.\"
         //
         verifyException("org.joda.time.format.PeriodFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Period period0 = new Period();
      Period period1 = period0.minusYears(0);
      assertSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Months months0 = Months.TWELVE;
      PeriodType periodType0 = months0.getPeriodType();
      Period period0 = null;
      try {
        period0 = new Period(1937, 1937, 2, 2, 2, 1937, 2, 1937, periodType0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Period does not support field 'years'
         //
         verifyException("org.joda.time.base.BasePeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PeriodType periodType0 = PeriodType.time();
      Period period0 = new Period(168L, periodType0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Period period0 = Period.minutes(0);
      Period period1 = period0.minusMonths(0);
      assertSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      LocalTime localTime0 = new LocalTime();
      Period period0 = Period.fieldDifference(localTime0, localTime0);
      Period period1 = period0.ZERO.withMillis(438);
      assertNotSame(period0, period1);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      LocalDate localDate0 = new LocalDate((long) (-12), dateTimeZone0);
      Interval interval0 = localDate0.toInterval();
      Period period0 = interval0.toPeriod();
      assertNotNull(period0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Period period0 = Period.weeks(561);
      assertNotNull(period0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      LocalTime localTime0 = LocalTime.MIDNIGHT;
      Period period0 = Period.fieldDifference(localTime0, localTime0);
      Period period1 = period0.toPeriod();
      assertSame(period0, period1);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Period period0 = new Period(0, 0, 0, 0, 0, 0, 0, 0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Period period0 = new Period(981, (-2538), (-180), (-2538));
      Period period1 = period0.minusHours(981);
      assertNotSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      LocalTime localTime0 = new LocalTime();
      Period period0 = new Period();
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetMillis((-347));
      DateTime dateTime0 = localTime0.toDateTimeToday(dateTimeZone0);
      Duration duration0 = period0.toStandardDuration();
      Period period1 = new Period(duration0, dateTime0);
      assertEquals(0L, duration0.getStandardSeconds());
      assertTrue(period1.equals((Object)period0));
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Period period0 = Period.seconds(7);
      Seconds seconds0 = period0.toStandardSeconds();
      assertEquals(7, seconds0.getSeconds());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Period period0 = new Period(7593750, (-2538), (-166), (-2538));
      Period period1 = period0.minusMinutes((-166));
      assertNotSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Period period0 = new Period(2542L, 2542L);
      Weeks weeks0 = period0.ZERO.toStandardWeeks();
      assertEquals(0, weeks0.getWeeks());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Instant instant0 = new Instant(595L);
      Period period0 = new Period(instant0, instant0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      GregorianChronology gregorianChronology0 = GregorianChronology.getInstanceUTC();
      MutablePeriod mutablePeriod0 = new MutablePeriod(393L, (-1454L), gregorianChronology0);
      Period period0 = mutablePeriod0.toPeriod();
      assertNotNull(period0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime((Object) null, (DateTimeZone) null);
      Period period0 = new Period(localDateTime0, localDateTime0, (PeriodType) null);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      DateTime dateTime0 = DateTime.now();
      Period period0 = Period.hours(8);
      Duration duration0 = period0.toDurationFrom(dateTime0);
      Period period1 = new Period(dateTime0, duration0);
      assertEquals(28800000L, duration0.getMillis());
      assertEquals(28800L, duration0.getStandardSeconds());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Object object0 = new Object();
      PeriodType periodType0 = PeriodType.weeks();
      Period period0 = null;
      try {
        period0 = new Period(object0, periodType0, (Chronology) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // No period converter found for type: java.lang.Object
         //
         verifyException("org.joda.time.convert.ConverterManager", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      LocalTime localTime0 = new LocalTime();
      Period period0 = Period.fieldDifference(localTime0, localTime0);
      Minutes minutes0 = period0.toStandardMinutes();
      assertEquals(0, minutes0.getMinutes());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Duration duration0 = Duration.standardSeconds(0);
      MutableDateTime mutableDateTime0 = MutableDateTime.now();
      PeriodType periodType0 = PeriodType.yearDay();
      Period period0 = duration0.toPeriodTo((ReadableInstant) mutableDateTime0, periodType0);
      // Undeclared exception!
      try { 
        period0.minusWeeks(63);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Field is not supported
         //
         verifyException("org.joda.time.PeriodType", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Period period0 = Period.months((-2538));
      assertNotNull(period0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      PeriodType periodType0 = PeriodType.standard();
      Period period0 = new Period(2389L, 30L, periodType0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      PeriodType periodType0 = PeriodType.standard();
      FixedDateTimeZone fixedDateTimeZone0 = (FixedDateTimeZone)DateTimeZone.UTC;
      GJChronology gJChronology0 = GJChronology.getInstance((DateTimeZone) fixedDateTimeZone0);
      Period period0 = new Period(0L, 0L, periodType0, gJChronology0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      DateTime dateTime0 = localDateTime0.toDateTime();
      Duration duration0 = Duration.millis(0L);
      PeriodType periodType0 = PeriodType.years();
      Period period0 = duration0.toPeriodFrom((ReadableInstant) dateTime0, periodType0);
      assertNotNull(period0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Period period0 = new Period(0L, 0L);
      Period period1 = period0.normalizedStandard();
      assertTrue(period1.equals((Object)period0));
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Period period0 = new Period(0L);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Period period0 = Period.ZERO;
      Period period1 = period0.ZERO.minusDays((-1010));
      assertNotSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Period period0 = new Period(2542L, 2542L);
      Period period1 = period0.minusSeconds((-1033));
      assertNotSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Period period0 = Period.seconds(7);
      Period period1 = period0.plusYears(1900);
      Period period2 = period1.normalizedStandard();
      assertTrue(period2.equals((Object)period1));
      assertNotSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Duration duration0 = Duration.standardSeconds(604800L);
      Interval interval0 = new Interval((ReadableInstant) null, duration0);
      DateTime dateTime0 = interval0.getStart();
      PeriodType periodType0 = PeriodType.standard();
      Period period0 = new Period(dateTime0, dateTime0, periodType0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      // Undeclared exception!
      try { 
        Period.fieldDifference((ReadablePartial) null, (ReadablePartial) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadablePartial objects must not be null
         //
         verifyException("org.joda.time.Period", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      LocalTime localTime0 = new LocalTime();
      // Undeclared exception!
      try { 
        Period.fieldDifference(localTime0, (ReadablePartial) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadablePartial objects must not be null
         //
         verifyException("org.joda.time.Period", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      DateTime dateTime0 = localDateTime0.toDateTime();
      LocalDate localDate0 = dateTime0.toLocalDate();
      // Undeclared exception!
      try { 
        Period.fieldDifference(localDateTime0, localDate0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadablePartial objects must have the same set of fields
         //
         verifyException("org.joda.time.Period", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      LocalTime localTime0 = new LocalTime();
      LocalDateTime localDateTime0 = LocalDateTime.now();
      // Undeclared exception!
      try { 
        Period.fieldDifference(localTime0, localDateTime0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadablePartial objects must have the same set of fields
         //
         verifyException("org.joda.time.Period", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance((DateTimeZone) null);
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[8];
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfMinute();
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[1] = dateTimeFieldType0;
      int[] intArray0 = new int[3];
      Partial partial0 = new Partial(gJChronology0, dateTimeFieldTypeArray0, intArray0);
      // Undeclared exception!
      try { 
        Period.fieldDifference(partial0, partial0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadablePartial objects must not have overlapping fields
         //
         verifyException("org.joda.time.Period", e);
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Period period0 = new Period(7593744, 7593744, (-1214), 93);
      PeriodType periodType0 = PeriodType.yearDayTime();
      Period period1 = period0.withPeriodType(periodType0);
      Period period2 = period1.withPeriodType(periodType0);
      assertNotSame(period2, period0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      LocalTime localTime0 = new LocalTime();
      Period period0 = Period.fieldDifference(localTime0, localTime0);
      Period period1 = period0.withFields(period0);
      assertNotSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Period period0 = Period.days((-816));
      Period period1 = period0.withFields((ReadablePeriod) null);
      assertSame(period0, period1);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Period period0 = new Period((-23), 7593750, (-23), (-23));
      DurationFieldType durationFieldType0 = DurationFieldType.HOURS_TYPE;
      Period period1 = period0.withField(durationFieldType0, (-23));
      assertTrue(period1.equals((Object)period0));
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Period period0 = Period.minutes((-279));
      // Undeclared exception!
      try { 
        period0.withField((DurationFieldType) null, (-279));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Field must not be null
         //
         verifyException("org.joda.time.Period", e);
      }
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Period period0 = new Period(0L, 0L);
      Hours hours0 = period0.toStandardHours();
      DurationFieldType durationFieldType0 = hours0.getFieldType();
      period0.withFieldAdded(durationFieldType0, (-403));
      assertEquals(0, hours0.getHours());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Period period0 = new Period(0, 7593750, 0, 0);
      // Undeclared exception!
      try { 
        period0.withFieldAdded((DurationFieldType) null, (-320));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Field must not be null
         //
         verifyException("org.joda.time.Period", e);
      }
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Period period0 = new Period(7593744, 7593744, (-1214), 93);
      Years years0 = Years.MAX_VALUE;
      DurationFieldType durationFieldType0 = years0.getFieldType();
      Period period1 = period0.withFieldAdded(durationFieldType0, 0);
      assertSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Period period0 = new Period(7593744, 7593744, (-1214), 93);
      Period period1 = period0.plus(period0);
      assertNotSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Period period0 = new Period(0L, 0L);
      Period period1 = period0.plus((ReadablePeriod) null);
      assertSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Period period0 = new Period(0, 0, 0, 0);
      Period period1 = period0.plusWeeks(0);
      assertSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Period period0 = new Period(0, 0, 0, 0);
      Period period1 = period0.plusDays(0);
      assertSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Period period0 = new Period(0L, 0L);
      Period period1 = period0.ZERO.plusHours(0);
      assertTrue(period1.equals((Object)period0));
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Period period0 = Period.minutes(7593750);
      Period period1 = period0.minusMinutes(0);
      assertSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Period period0 = new Period(7593750, (-2538), (-166), (-2538));
      Period period1 = period0.plusSeconds(0);
      assertSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      Period period0 = Period.fieldDifference(localDateTime0, localDateTime0);
      Period period1 = period0.ZERO.plusMillis(0);
      assertNotSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Period period0 = Period.seconds(76);
      Weeks weeks0 = Weeks.MIN_VALUE;
      Period period1 = period0.minus(weeks0);
      assertNotSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Period period0 = Period.years((-351));
      Period period1 = period0.minus((ReadablePeriod) null);
      assertSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Period period0 = Period.ZERO;
      Period period1 = period0.multipliedBy(4002);
      assertSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Period period0 = Period.minutes(667);
      Period period1 = period0.multipliedBy(1);
      assertSame(period1, period0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Period period0 = Period.seconds(76);
      Period period1 = period0.minusMonths(320);
      // Undeclared exception!
      try { 
        period1.toStandardDuration();
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Cannot convert to Duration as this period contains months and months vary in length
         //
         verifyException("org.joda.time.Period", e);
      }
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Period period0 = Period.minutes(0);
      Period period1 = period0.minusYears(2012);
      // Undeclared exception!
      try { 
        period1.toStandardDays();
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Cannot convert to Days as this period contains years and years vary in length
         //
         verifyException("org.joda.time.Period", e);
      }
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Period period0 = Period.seconds(76);
      Period period1 = period0.minusMonths(320);
      PeriodType periodType0 = PeriodType.dayTime();
      // Undeclared exception!
      try { 
        period1.normalizedStandard(periodType0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Field is not supported
         //
         verifyException("org.joda.time.PeriodType", e);
      }
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Period period0 = Period.seconds(76);
      Period period1 = period0.plusMonths(7);
      PeriodType periodType0 = PeriodType.dayTime();
      // Undeclared exception!
      try { 
        period1.normalizedStandard(periodType0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Field is not supported
         //
         verifyException("org.joda.time.PeriodType", e);
      }
  }
}
