/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:42:27 GMT 2023
 */

package org.joda.time.chrono;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.joda.time.Chronology;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.joda.time.Instant;
import org.joda.time.LocalDate;
import org.joda.time.MonthDay;
import org.joda.time.MutableDateTime;
import org.joda.time.MutablePeriod;
import org.joda.time.Partial;
import org.joda.time.Period;
import org.joda.time.ReadableInstant;
import org.joda.time.ReadablePartial;
import org.joda.time.ReadablePeriod;
import org.joda.time.Weeks;
import org.joda.time.YearMonth;
import org.joda.time.Years;
import org.joda.time.chrono.BuddhistChronology;
import org.joda.time.chrono.GJChronology;
import org.joda.time.chrono.LenientChronology;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class GJChronology_ESTest extends GJChronology_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      long long0 = gJChronology0.julianToGregorianByWeekyear((-31931865L));
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals((-1241531865L), long0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      MutableDateTime mutableDateTime0 = new MutableDateTime((Chronology) buddhistChronology0);
      // Undeclared exception!
      try { 
        mutableDateTime0.setWeekyear((-236));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The resulting instant is below the supported minimum of 0001-01-01T00:00:00.000-07:52:58 (BuddhistChronology[America/Los_Angeles])
         //
         verifyException("org.joda.time.chrono.LimitChronology", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Instant instant0 = Instant.now();
      DateTimeZone dateTimeZone0 = instant0.getZone();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0);
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.minuteOfHour();
      Partial partial0 = new Partial(dateTimeFieldType0, 2);
      // Undeclared exception!
      try { 
        gJChronology0.set(partial0, (-9223372036854775808L));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 2 for minuteOfHour is not supported
         //
         verifyException("org.joda.time.chrono.GJChronology$CutoverField", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Instant instant0 = Instant.now();
      DateTimeZone dateTimeZone0 = instant0.getZone();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0, (ReadableInstant) instant0, 3);
      MonthDay monthDay0 = new MonthDay(3, 3, gJChronology0);
      assertEquals(3, monthDay0.getDayOfMonth());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Instant instant0 = Instant.now();
      DateTimeZone dateTimeZone0 = instant0.getZone();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0, (ReadableInstant) instant0, 6);
      YearMonth yearMonth0 = new YearMonth(instant0, gJChronology0);
      YearMonth yearMonth1 = yearMonth0.minusYears(6);
      assertEquals(2, yearMonth1.getMonthOfYear());
      assertEquals(2008, yearMonth1.getYear());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      gJChronology0.hashCode();
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      LenientChronology lenientChronology0 = LenientChronology.getInstance(buddhistChronology0);
      MonthDay monthDay0 = new MonthDay((-78071904000000L), (Chronology) lenientChronology0);
      Weeks.weeksBetween((ReadablePartial) monthDay0, (ReadablePartial) monthDay0);
      assertEquals(6, monthDay0.getDayOfMonth());
      assertEquals(1, monthDay0.getMonthOfYear());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      MonthDay monthDay0 = new MonthDay((Chronology) gJChronology0);
      MonthDay monthDay1 = monthDay0.plusDays((-2210));
      assertEquals(27, monthDay1.getDayOfMonth());
      assertEquals(1, monthDay1.getMonthOfYear());
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MutablePeriod mutablePeriod0 = new MutablePeriod((-62135765938990L));
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstanceUTC();
      int[] intArray0 = buddhistChronology0.get((ReadablePeriod) mutablePeriod0, (-62135765938990L), (long) 1);
      assertArrayEquals(new int[] {1968, 11, 4, 2, 22, 58, 58, 991}, intArray0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Years years0 = Years.ONE;
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0, (long) 1, 1);
      long long0 = gJChronology0.add((ReadablePeriod) years0, (-31301516990L), 1);
      assertEquals(234483010L, long0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0, (-12219292800000L), 1);
      assertEquals(1, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      GJChronology gJChronology1 = (GJChronology)gJChronology0.withZone((DateTimeZone) null);
      assertEquals(4, gJChronology1.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      YearMonth yearMonth0 = new YearMonth((Chronology) gJChronology0);
      yearMonth0.toLocalDate(1);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals(2014, yearMonth0.getYear());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      try { 
        gJChronology0.getDateTimeMillis(551, 551, 551, 551);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 551 for monthOfYear must be in the range [1,12]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      long long0 = gJChronology0.getDateTimeMillis(6, 6, 6, 6);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals((-61964524799994L), long0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      try { 
        gJChronology0.getDateTimeMillis(29, 2, 29, 2, 2, 2, 2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 29 for dayOfMonth must be in the range [1,28]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      try { 
        gJChronology0.getDateTimeMillis((-292269054), (-292269054), (-292269054), (-292269054), (-292269054), (-292269054), (-292269054));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -292269054 for hourOfDay must be in the range [0,23]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      try { 
        gJChronology0.getDateTimeMillis(2, 2, (-105), 2, 2, (-105), 2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -105 for secondOfMinute must be in the range [0,59]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstanceUTC();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0);
      long long0 = gJChronology0.getDateTimeMillis(5633, 4, 1, 4, 4, 4, 1);
      assertEquals(115600968244001L, long0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      long long0 = gJChronology0.getDateTimeMillis(6, 6, 6, 6, 6, 6, 6);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals((-61964474455994L), long0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstanceUTC();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0);
      boolean boolean0 = gJChronology0.equals(dateTimeZone0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      boolean boolean0 = gJChronology0.equals(gJChronology0);
      assertTrue(boolean0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Instant instant0 = Instant.now();
      DateTimeZone dateTimeZone0 = instant0.getZone();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0, (ReadableInstant) instant0, 2);
      GJChronology gJChronology1 = GJChronology.getInstance(dateTimeZone0, (ReadableInstant) instant0);
      boolean boolean0 = gJChronology0.equals(gJChronology1);
      assertEquals(4, gJChronology1.getMinimumDaysInFirstWeek());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      DateTimeZone dateTimeZone0 = gJChronology0.getZone();
      MutableDateTime mutableDateTime0 = new MutableDateTime(dateTimeZone0);
      GJChronology gJChronology1 = GJChronology.getInstance(dateTimeZone0, (ReadableInstant) mutableDateTime0);
      boolean boolean0 = gJChronology1.equals(gJChronology0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertFalse(boolean0);
      assertFalse(gJChronology0.equals((Object)gJChronology1));
      assertEquals(4, gJChronology1.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      GJChronology gJChronology1 = GJChronology.getInstance();
      boolean boolean0 = gJChronology0.equals(gJChronology1);
      assertFalse(boolean0);
      assertEquals(4, gJChronology1.getMinimumDaysInFirstWeek());
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      String string0 = gJChronology0.toString();
      assertEquals("GJChronology[America/Los_Angeles]", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0, 0L, 1);
      String string0 = gJChronology0.toString();
      assertEquals("GJChronology[America/Los_Angeles,cutover=1970-01-01,mdfw=1]", string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstanceUTC();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0, (long) 1, 1);
      String string0 = gJChronology0.toString();
      assertEquals("GJChronology[UTC,cutover=1970-01-01T00:00:00.001Z,mdfw=1]", string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      YearMonth yearMonth0 = YearMonth.now((Chronology) gJChronology0);
      Period period0 = Period.years((-1058));
      YearMonth yearMonth1 = yearMonth0.minus(period0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals(2, yearMonth1.getMonthOfYear());
      assertEquals(3072, yearMonth1.getYear());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      YearMonth yearMonth0 = new YearMonth(2, 2, gJChronology0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      MonthDay monthDay0 = new MonthDay(1116L);
      MonthDay monthDay1 = monthDay0.plusMonths(1274);
      // Undeclared exception!
      try { 
        monthDay1.withChronologyRetainFields(gJChronology0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 29 for dayOfMonth must not be larger than 28
         //
         verifyException("org.joda.time.chrono.BaseChronology", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Instant instant0 = Instant.now();
      DateTimeZone dateTimeZone0 = instant0.getZone();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0, (ReadableInstant) instant0, 2);
      LocalDate localDate0 = new LocalDate((Chronology) gJChronology0);
      assertEquals(3, localDate0.size());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      MutablePeriod mutablePeriod0 = new MutablePeriod();
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      int[] intArray0 = buddhistChronology0.get((ReadablePeriod) mutablePeriod0, (long) 1, (-62134761595177L));
      assertArrayEquals(new int[] {(-1968), (-11), (-2), (-5), (-7), (-52), (-53), (-178)}, intArray0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      Years years0 = Years.ONE;
      long long0 = gJChronology0.add((ReadablePeriod) years0, (long) (-1244), (-1244));
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals((-39256531623244L), long0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      Years years0 = Years.ONE;
      long long0 = gJChronology0.add((ReadablePeriod) years0, (-36350553600000L), 60);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals((-34457097600000L), long0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Years years0 = Years.ONE;
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0, (long) 1, 1);
      long long0 = gJChronology0.add((ReadablePeriod) years0, (long) 1, 1);
      assertEquals(30412800001L, long0);
      assertEquals(1, gJChronology0.getMinimumDaysInFirstWeek());
  }
}