/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:22:58 GMT 2023
 */

package org.joda.time.chrono;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Locale;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.joda.time.Chronology;
import org.joda.time.DateTime;
import org.joda.time.DateTimeField;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.joda.time.DurationField;
import org.joda.time.Instant;
import org.joda.time.LocalDate;
import org.joda.time.LocalDateTime;
import org.joda.time.MonthDay;
import org.joda.time.MutablePeriod;
import org.joda.time.Partial;
import org.joda.time.Period;
import org.joda.time.ReadableInstant;
import org.joda.time.ReadablePartial;
import org.joda.time.ReadablePeriod;
import org.joda.time.YearMonth;
import org.joda.time.Years;
import org.joda.time.chrono.AssembledChronology;
import org.joda.time.chrono.BuddhistChronology;
import org.joda.time.chrono.GJChronology;
import org.joda.time.chrono.LenientChronology;
import org.joda.time.chrono.ZonedChronology;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class GJChronology_ESTest extends GJChronology_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.weekyear();
      Partial partial0 = new Partial(dateTimeFieldType0, 1217, gJChronology0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Instant instant0 = new Instant((-79268918400000L));
      DateTimeZone dateTimeZone0 = instant0.getZone();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      DateTimeField dateTimeField0 = gJChronology0.yearOfEra();
      DateTimeZone dateTimeZone0 = gJChronology0.getZone();
      DurationField durationField0 = gJChronology0.days();
      ZonedChronology.ZonedDateTimeField zonedChronology_ZonedDateTimeField0 = new ZonedChronology.ZonedDateTimeField(dateTimeField0, dateTimeZone0, durationField0, (DurationField) null, (DurationField) null);
      long long0 = zonedChronology_ZonedDateTimeField0.set(3424L, "1582", (Locale) null);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals((-12243225596576L), long0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      long long0 = gJChronology0.julianToGregorianByWeekyear((-352L));
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals((-1209600352L), long0);
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
      LenientChronology lenientChronology0 = LenientChronology.getInstance(gJChronology0);
      DateTime dateTime0 = new DateTime(1210, 1210, 1210, 1210, 1210, 1210, lenientChronology0);
      DateTime dateTime1 = dateTime0.withYearOfCentury(1210);
      assertEquals(17047027810000L, dateTime1.getMillis());
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.era();
      Locale locale0 = Locale.KOREAN;
      // Undeclared exception!
      try { 
        dateTimeField0.getAsShortText(3978, locale0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 3978
         //
         verifyException("org.joda.time.chrono.GJLocaleSymbols", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.era();
      LocalDate localDate0 = new LocalDate((long) 40);
      dateTimeField0.getMinimumValue((ReadablePartial) localDate0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      DateTimeField dateTimeField0 = gJChronology0.yearOfEra();
      DateTimeZone dateTimeZone0 = gJChronology0.getZone();
      DurationField durationField0 = gJChronology0.days();
      ZonedChronology.ZonedDateTimeField zonedChronology_ZonedDateTimeField0 = new ZonedChronology.ZonedDateTimeField(dateTimeField0, dateTimeZone0, durationField0, (DurationField) null, (DurationField) null);
      Locale locale0 = Locale.ROOT;
      zonedChronology_ZonedDateTimeField0.getMaximumShortTextLength(locale0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.era();
      // Undeclared exception!
      try { 
        dateTimeField0.getMaximumValue((ReadablePartial) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.joda.time.chrono.BaseChronology", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.era();
      // Undeclared exception!
      try { 
        dateTimeField0.getDifference(1L, 1L);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // eras field is unsupported
         //
         verifyException("org.joda.time.field.UnsupportedDurationField", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.era();
      dateTimeField0.getMaximumTextLength((Locale) null);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      MonthDay monthDay0 = MonthDay.now((Chronology) gJChronology0);
      MonthDay monthDay1 = monthDay0.plusDays(1);
      assertEquals(2, monthDay1.getMonthOfYear());
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals(15, monthDay1.getDayOfMonth());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.era();
      Locale locale0 = Locale.CHINESE;
      // Undeclared exception!
      try { 
        dateTimeField0.getAsText(40, locale0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 40
         //
         verifyException("org.joda.time.chrono.GJLocaleSymbols", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.era();
      // Undeclared exception!
      try { 
        dateTimeField0.getDifferenceAsLong(449L, 12);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // eras field is unsupported
         //
         verifyException("org.joda.time.field.UnsupportedDurationField", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      MutablePeriod mutablePeriod0 = new MutablePeriod();
      int[] intArray0 = gJChronology0.get((ReadablePeriod) mutablePeriod0, 2434L, (-62041046400039L));
      assertArrayEquals(new int[] {(-1965), (-11), (-4), (-1), (-16), 0, (-2), (-473)}, intArray0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Years years0 = Years.years(1);
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      long long0 = gJChronology0.add((ReadablePeriod) years0, (long) 1, 1);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals(31536000001L, long0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0, (-12219292800000L), 1);
      AssembledChronology.Fields assembledChronology_Fields0 = new AssembledChronology.Fields();
      gJChronology0.assemble(assembledChronology_Fields0);
      assertEquals(1, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      GJChronology gJChronology1 = (GJChronology)gJChronology0.withZone((DateTimeZone) null);
      assertEquals(4, gJChronology1.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      long long0 = gJChronology0.getDateTimeMillis(1644, 2, 2, 2);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals((-10284768421998L), long0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      long long0 = gJChronology0.getDateTimeMillis(3, 3, 3, 3);
      assertEquals((-62067427199997L), long0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      long long0 = gJChronology0.getDateTimeMillis((-3849), 2, 29, 2, 29, 2, 2);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals((-183595930257998L), long0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      LocalDateTime localDateTime0 = new LocalDateTime(1209600000L, (Chronology) gJChronology0);
      DateTime dateTime0 = localDateTime0.toDateTime();
      assertEquals(1238400000L, dateTime0.getMillis());
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      DateTime dateTime0 = null;
      try {
        dateTime0 = new DateTime(37, 37, 37, 37, 37, 37, gJChronology0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 37 for hourOfDay must be in the range [0,23]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      try { 
        gJChronology0.getDateTimeMillis(97, 2, 97, 97, 97, 2, 97);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 97 for hourOfDay must be in the range [0,23]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      boolean boolean0 = gJChronology0.equals("AD");
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      boolean boolean0 = gJChronology0.equals(gJChronology0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      DateTime dateTime0 = new DateTime((Object) null, gJChronology0);
      GJChronology gJChronology1 = GJChronology.getInstance((DateTimeZone) null, (ReadableInstant) dateTime0, 4);
      boolean boolean0 = gJChronology0.equals(gJChronology1);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      GJChronology gJChronology1 = GJChronology.getInstance();
      boolean boolean0 = gJChronology1.equals(gJChronology0);
      assertEquals(4, gJChronology1.getMinimumDaysInFirstWeek());
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      String string0 = gJChronology0.toString();
      assertEquals("GJChronology[America/Los_Angeles]", string0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0, (-149805504000000L), 1);
      String string0 = gJChronology0.toString();
      assertEquals("GJChronology[America/Los_Angeles,cutover=-2778-11-08,mdfw=1]", string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      Instant instant0 = new Instant();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0, (ReadableInstant) instant0);
      String string0 = gJChronology0.toString();
      assertEquals("GJChronology[America/Los_Angeles,cutover=2014-02-14T20:21:21.320Z]", string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.era();
      dateTimeField0.getAsText((-19023062400000L));
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.dayOfMonth();
      String string0 = dateTimeField0.getAsText(2841L);
      assertEquals("31", string0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.era();
      dateTimeField0.getAsShortText((-12219292800000L));
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      DateTimeField dateTimeField0 = gJChronology0.yearOfEra();
      DateTimeZone dateTimeZone0 = gJChronology0.getZone();
      DurationField durationField0 = gJChronology0.days();
      ZonedChronology.ZonedDateTimeField zonedChronology_ZonedDateTimeField0 = new ZonedChronology.ZonedDateTimeField(dateTimeField0, dateTimeZone0, durationField0, (DurationField) null, (DurationField) null);
      String string0 = zonedChronology_ZonedDateTimeField0.getAsShortText((-12219292800000L));
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals("1582", string0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      MonthDay monthDay0 = MonthDay.now((Chronology) gJChronology0);
      Period period0 = new Period((long) 0, (Chronology) gJChronology0);
      MonthDay monthDay1 = monthDay0.plus(period0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals(14, monthDay1.getDayOfMonth());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.era();
      long long0 = dateTimeField0.addWrapField((-12219292800000L), 40);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals((-12219292800000L), long0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      YearMonth yearMonth0 = YearMonth.now();
      long long0 = gJChronology0.set(yearMonth0, (-5647336506460800000L));
      assertEquals(1391212800000L, long0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forID((String) null);
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0, (-1L), 3);
      LocalDateTime localDateTime0 = new LocalDateTime(778L);
      LocalDate localDate0 = localDateTime0.toLocalDate();
      DateTime dateTime0 = new DateTime((Chronology) gJChronology0);
      // Undeclared exception!
      try { 
        localDate0.toDateTime(dateTime0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 31 for dayOfMonth is not supported
         //
         verifyException("org.joda.time.chrono.GJChronology$CutoverField", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.era();
      Locale locale0 = Locale.UK;
      long long0 = dateTimeField0.set(9223372036854775789L, "AD", locale0);
      assertEquals(9223372036854775789L, long0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.era();
      Locale locale0 = Locale.ENGLISH;
      long long0 = dateTimeField0.set((-19023062400000L), "AD", locale0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals((-19023062400000L), long0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.era();
      dateTimeField0.isLeap((-79268918400000L));
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.era();
      dateTimeField0.isLeap(512L);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.dayOfMonth();
      dateTimeField0.getLeapAmount((-62101295999998L));
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.era();
      dateTimeField0.getLeapAmount(2841L);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.era();
      int int0 = dateTimeField0.getMinimumValue(9223372036829575795L);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.era();
      int int0 = dateTimeField0.getMinimumValue((-28800000L));
      assertEquals(1, int0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      MonthDay monthDay0 = new MonthDay((-1L), (Chronology) gJChronology0);
      MonthDay monthDay1 = monthDay0.withMonthOfYear(2);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals(28, monthDay1.getDayOfMonth());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.dayOfMonth();
      long long0 = dateTimeField0.roundHalfEven(1);
      assertEquals(28800000L, long0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.era();
      long long0 = dateTimeField0.roundHalfEven((-255L));
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals((-9223372036825975809L), long0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      DateTimeField dateTimeField0 = gJChronology0.era();
      long long0 = dateTimeField0.roundHalfEven((-23036601599995L));
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals((-62105356800000L), long0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.era();
      long long0 = dateTimeField0.roundHalfEven((-79271564340000L));
      assertEquals((-62105328422000L), long0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      MutablePeriod mutablePeriod0 = new MutablePeriod();
      int[] intArray0 = gJChronology0.get((ReadablePeriod) mutablePeriod0, (-62041046400039L), 2747L);
      assertArrayEquals(new int[] {1965, 11, 4, 1, 16, 0, 2, 786}, intArray0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.year();
      long long0 = dateTimeField0.getDifferenceAsLong((-62134732800005L), (-62134732800005L));
      assertEquals(0L, long0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.year();
      long long0 = dateTimeField0.getDifferenceAsLong(31083663600000L, (-62134732800000L));
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
      assertEquals(2953L, long0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.year();
      long long0 = dateTimeField0.getDifferenceAsLong(12, 12);
      assertEquals(0L, long0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField0 = gJChronology0.year();
      long long0 = dateTimeField0.addWrapField((long) 40, 40);
      assertEquals(1262304000040L, long0);
      assertEquals(4, gJChronology0.getMinimumDaysInFirstWeek());
  }
}