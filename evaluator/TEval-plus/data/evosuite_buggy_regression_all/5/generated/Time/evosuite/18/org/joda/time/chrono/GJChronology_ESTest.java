/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:42:58 GMT 2023
 */

package org.joda.time.chrono;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.util.MockGregorianCalendar;
import org.joda.time.Chronology;
import org.joda.time.DateMidnight;
import org.joda.time.DateTime;
import org.joda.time.DateTimeZone;
import org.joda.time.Duration;
import org.joda.time.Interval;
import org.joda.time.LocalDate;
import org.joda.time.LocalDateTime;
import org.joda.time.MonthDay;
import org.joda.time.Months;
import org.joda.time.MutableInterval;
import org.joda.time.MutablePeriod;
import org.joda.time.Partial;
import org.joda.time.Period;
import org.joda.time.ReadableInstant;
import org.joda.time.ReadablePeriod;
import org.joda.time.YearMonth;
import org.joda.time.Years;
import org.joda.time.chrono.AssembledChronology;
import org.joda.time.chrono.BuddhistChronology;
import org.joda.time.chrono.GJChronology;
import org.joda.time.chrono.LenientChronology;
import org.joda.time.tz.FixedDateTimeZone;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class GJChronology_ESTest extends GJChronology_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      MockGregorianCalendar mockGregorianCalendar0 = new MockGregorianCalendar((-3), (-3), (-3));
      LocalDateTime localDateTime0 = LocalDateTime.fromCalendarFields(mockGregorianCalendar0);
      Partial partial0 = new Partial(localDateTime0);
      partial0.withChronologyRetainFields(gJChronology0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      DateMidnight dateMidnight0 = new DateMidnight((-108717465600000L), (Chronology) gJChronology0);
      DateMidnight dateMidnight1 = dateMidnight0.withWeekyear(1582);
      assertEquals((-12214972800000L), dateMidnight1.getMillis());
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0, (ReadableInstant) null);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      FixedDateTimeZone fixedDateTimeZone0 = (FixedDateTimeZone)DateTimeZone.UTC;
      GJChronology gJChronology0 = GJChronology.getInstance((DateTimeZone) fixedDateTimeZone0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      gJChronology0.hashCode();
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      YearMonth yearMonth0 = new YearMonth((-12219292800000L), (Chronology) gJChronology0);
      YearMonth yearMonth1 = yearMonth0.plusYears(1);
      assertEquals(1583, yearMonth1.getYear());
      assertEquals(10, yearMonth1.getMonthOfYear());
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      gJChronology0.equals(gJChronology0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTime dateTime0 = DateTime.now((Chronology) gJChronology0);
      DateTime dateTime1 = dateTime0.withWeekyear((-2331));
      assertEquals((-135692768740680L), dateTime1.getMillis());
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      LenientChronology.getInstance(gJChronology0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      MonthDay monthDay0 = new MonthDay((-109411776000000L), (Chronology) gJChronology0);
      MonthDay monthDay1 = monthDay0.plusDays(92);
      assertEquals(2, monthDay1.getDayOfMonth());
      assertEquals(3, monthDay1.getMonthOfYear());
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      Months months0 = Months.FOUR;
      int[] intArray0 = gJChronology0.get((ReadablePeriod) months0, (-58996684800000L), 699L);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertArrayEquals(new int[] {22434}, intArray0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      MutablePeriod mutablePeriod0 = new MutablePeriod((-2314), (-62032867199996L), gJChronology0);
      DateTime dateTime0 = new DateTime();
      Duration duration0 = mutablePeriod0.toDurationFrom(dateTime0);
      Period period0 = duration0.toPeriodTo((ReadableInstant) dateTime0);
      long long0 = gJChronology0.add((ReadablePeriod) period0, (long) (-2314), 2173);
      assertEquals((-134799594815971300L), long0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetMillis(428);
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0, (-12219292800000L), 1);
      try { 
        gJChronology0.getDateTimeMillis(428, 0, 1, 0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 0 for monthOfYear must be in the range [1,12]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      YearMonth yearMonth0 = new YearMonth((Chronology) gJChronology0);
      Interval interval0 = yearMonth0.toInterval();
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals(1393660800000L, interval0.getEndMillis());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      GJChronology gJChronology1 = (GJChronology)gJChronology0.withZone((DateTimeZone) null);
      assertEquals(4, gJChronology1.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      long long0 = gJChronology0.getDateTimeMillis(4, 4, 4, 4);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals((-62033011621996L), long0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetMillis(442);
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0, (long) 0, 1);
      long long0 = gJChronology0.getDateTimeMillis(73281320, 1, 1, 0, 1, 1, 1);
      assertEquals(1, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals(2312472930508860559L, long0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      long long0 = gJChronology0.getDateTimeMillis(11, 11, 11, 11, 11, 11, 11);
      assertEquals((-61793066928989L), long0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      String string0 = gJChronology0.toString();
      assertEquals("GJChronology[UTC]", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetMillis(442);
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0, (long) 0, 1);
      String string0 = gJChronology0.toString();
      assertEquals("GJChronology[+00:00:00.442,cutover=1970-01-01,mdfw=1]", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance((DateTimeZone) null, (long) 1, 1);
      String string0 = gJChronology0.toString();
      assertEquals("GJChronology[America/Los_Angeles,cutover=1970-01-01T00:00:00.001Z,mdfw=1]", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      AssembledChronology.Fields assembledChronology_Fields0 = new AssembledChronology.Fields();
      gJChronology0.assemble(assembledChronology_Fields0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      YearMonth yearMonth0 = new YearMonth((Chronology) gJChronology0);
      Period period0 = Period.weeks(0);
      YearMonth yearMonth1 = yearMonth0.plus(period0);
      assertEquals(2014, yearMonth1.getYear());
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0, (long) 1, 1);
      LocalDate localDate0 = new LocalDate((-62135769599616L), (Chronology) gJChronology0);
      YearMonth yearMonth0 = new YearMonth((long) 1);
      // Undeclared exception!
      try { 
        localDate0.withFields(yearMonth0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 1969 for year is not supported
         //
         verifyException("org.joda.time.chrono.GJChronology$CutoverField", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      MonthDay monthDay0 = new MonthDay();
      monthDay0.withChronologyRetainFields(gJChronology0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      Period period0 = new Period((-61792894138989L), (-12219292800000L), gJChronology0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      Years years0 = Years.TWO;
      long long0 = gJChronology0.add((ReadablePeriod) years0, (long) 1618, 1618);
      assertEquals(102118320001618L, long0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0, (long) 1, 1);
      MutableInterval mutableInterval0 = new MutableInterval((-62135769600000L), 52L);
      Months months0 = Months.monthsIn(mutableInterval0);
      long long0 = gJChronology0.add((ReadablePeriod) months0, (-62135769600000L), 1);
      assertEquals(1, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals(1123622000L, long0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      Years years0 = Years.TWO;
      long long0 = gJChronology0.add((ReadablePeriod) years0, (-42521587199980L), 2998);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals(146693635200020L, long0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      MutablePeriod mutablePeriod0 = new MutablePeriod((-2314), (-1156L), gJChronology0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }
}