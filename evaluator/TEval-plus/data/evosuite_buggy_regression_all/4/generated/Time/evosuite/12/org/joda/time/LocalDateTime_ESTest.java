/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:22:35 GMT 2023
 */

package org.joda.time;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Calendar;
import java.util.Date;
import java.util.Locale;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.util.MockGregorianCalendar;
import org.joda.time.Chronology;
import org.joda.time.DateTimeField;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.joda.time.Duration;
import org.joda.time.DurationFieldType;
import org.joda.time.Hours;
import org.joda.time.LocalDateTime;
import org.joda.time.Minutes;
import org.joda.time.ReadableDuration;
import org.joda.time.ReadablePartial;
import org.joda.time.ReadablePeriod;
import org.joda.time.Weeks;
import org.joda.time.chrono.BuddhistChronology;
import org.joda.time.chrono.CopticChronology;
import org.joda.time.chrono.EthiopicChronology;
import org.joda.time.chrono.GJChronology;
import org.joda.time.chrono.ISOChronology;
import org.joda.time.chrono.LenientChronology;
import org.joda.time.chrono.ZonedChronology;
import org.joda.time.tz.FixedDateTimeZone;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class LocalDateTime_ESTest extends LocalDateTime_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test000()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocalDateTime.parse("$BpT*v]qXUJ");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid format: \"$BpT*v]qXUJ\"
         //
         verifyException("org.joda.time.format.DateTimeFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test001()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.hourOfDay();
      assertNotNull(localDateTime_Property0);
  }

  @Test(timeout = 4000)
  public void test002()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      Date date0 = localDateTime0.toDate();
      assertEquals("Fri Feb 14 12:21:21 GMT 2014", date0.toString());
  }

  @Test(timeout = 4000)
  public void test003()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime((-9223372016640000050L), (Chronology) null);
      localDateTime0.withMillisOfDay(2688);
  }

  @Test(timeout = 4000)
  public void test004()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.withYear((-153));
  }

  @Test(timeout = 4000)
  public void test005()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      LocalDateTime localDateTime0 = new LocalDateTime((long) 1, dateTimeZone0);
      localDateTime0.withHourOfDay(1);
  }

  @Test(timeout = 4000)
  public void test006()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.withDayOfMonth(8);
  }

  @Test(timeout = 4000)
  public void test007()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      // Undeclared exception!
      try { 
        localDateTime0.withMillisOfSecond((-436));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -436 for millisOfSecond must be in the range [0,999]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test008()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.toLocalTime();
  }

  @Test(timeout = 4000)
  public void test009()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.secondOfMinute();
  }

  @Test(timeout = 4000)
  public void test010()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      Hours hours0 = Hours.ONE;
      localDateTime0.minus((ReadablePeriod) hours0);
  }

  @Test(timeout = 4000)
  public void test011()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.getWeekOfWeekyear();
  }

  @Test(timeout = 4000)
  public void test012()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.withYearOfCentury(1);
  }

  @Test(timeout = 4000)
  public void test013()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.withDayOfYear(5);
  }

  @Test(timeout = 4000)
  public void test014()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.getEra();
  }

  @Test(timeout = 4000)
  public void test015()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.getDayOfYear();
  }

  @Test(timeout = 4000)
  public void test016()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.millisOfSecond();
  }

  @Test(timeout = 4000)
  public void test017()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      // Undeclared exception!
      try { 
        localDateTime0.withMonthOfYear((-2411));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -2411 for monthOfYear must be in the range [1,12]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test018()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.millisOfDay();
  }

  @Test(timeout = 4000)
  public void test019()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime((DateTimeZone) null);
      localDateTime0.plusDays(0);
  }

  @Test(timeout = 4000)
  public void test020()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      LocalDateTime localDateTime0 = new LocalDateTime((Object) null, dateTimeZone0);
  }

  @Test(timeout = 4000)
  public void test021()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minuteOfHour();
  }

  @Test(timeout = 4000)
  public void test022()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.getYearOfCentury();
  }

  @Test(timeout = 4000)
  public void test023()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime((Object) null);
  }

  @Test(timeout = 4000)
  public void test024()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.monthOfYear();
      localDateTime_Property0.roundHalfEvenCopy();
  }

  @Test(timeout = 4000)
  public void test025()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1);
  }

  @Test(timeout = 4000)
  public void test026()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.withWeekOfWeekyear(1);
  }

  @Test(timeout = 4000)
  public void test027()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.weekOfWeekyear();
  }

  @Test(timeout = 4000)
  public void test028()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      // Undeclared exception!
      try { 
        localDateTime0.withWeekyear((-2147483637));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 2147483637 for weekyear must be in the range [-292275054,292278993]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test029()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      // Undeclared exception!
      try { 
        localDateTime0.withDayOfWeek((-2147483637));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -2147483637 for dayOfWeek must be in the range [1,7]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test030()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.toDateTime();
  }

  @Test(timeout = 4000)
  public void test031()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minus((ReadableDuration) null);
  }

  @Test(timeout = 4000)
  public void test032()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.dayOfWeek();
  }

  @Test(timeout = 4000)
  public void test033()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.withYearOfEra(1984);
  }

  @Test(timeout = 4000)
  public void test034()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      Weeks weeks0 = Weeks.ONE;
      Duration duration0 = weeks0.toStandardDuration();
      localDateTime0.plus((ReadableDuration) duration0);
  }

  @Test(timeout = 4000)
  public void test035()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(21260793600000L);
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstanceUTC();
      DateTimeField dateTimeField0 = buddhistChronology0.yearOfCentury();
      LocalDateTime.Property localDateTime_Property0 = new LocalDateTime.Property(localDateTime0, dateTimeField0);
      localDateTime_Property0.roundFloorCopy();
  }

  @Test(timeout = 4000)
  public void test036()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.getYearOfEra();
  }

  @Test(timeout = 4000)
  public void test037()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.yearOfCentury();
  }

  @Test(timeout = 4000)
  public void test038()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1, 1);
  }

  @Test(timeout = 4000)
  public void test039()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.getMillisOfDay();
  }

  @Test(timeout = 4000)
  public void test040()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.withCenturyOfEra(1687);
  }

  @Test(timeout = 4000)
  public void test041()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.withDate(1, 1, 1);
  }

  @Test(timeout = 4000)
  public void test042()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      // Undeclared exception!
      try { 
        localDateTime0.withEra(2147245065);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 2147245065 for era must be in the range [0,1]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test043()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      // Undeclared exception!
      try { 
        localDateTime0.withSecondOfMinute(25734375);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 25734375 for secondOfMinute must be in the range [0,59]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test044()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.getCenturyOfEra();
  }

  @Test(timeout = 4000)
  public void test045()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.weekyear();
  }

  @Test(timeout = 4000)
  public void test046()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.era();
  }

  @Test(timeout = 4000)
  public void test047()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.getWeekyear();
  }

  @Test(timeout = 4000)
  public void test048()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      // Undeclared exception!
      try { 
        localDateTime0.withMinuteOfHour(2147330267);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 2147330267 for minuteOfHour must be in the range [0,59]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test049()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(60000L, (DateTimeZone) null);
      // Undeclared exception!
      try { 
        localDateTime0.withTime(1281, 3, 3, 1281);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 1281 for hourOfDay must be in the range [0,23]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test050()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      Minutes minutes0 = Minutes.ZERO;
      localDateTime0.plus((ReadablePeriod) minutes0);
  }

  @Test(timeout = 4000)
  public void test051()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.getDayOfWeek();
  }

  @Test(timeout = 4000)
  public void test052()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.toLocalDate();
  }

  @Test(timeout = 4000)
  public void test053()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetHoursMinutes(50, 50);
      LocalDateTime localDateTime0 = LocalDateTime.now(dateTimeZone0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.dayOfMonth();
      // Undeclared exception!
      try { 
        localDateTime_Property0.setCopy("gcMa1[.Yfs}fNg<${J");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value \"gcMa1[.Yfs}fNg<${J\" for dayOfMonth is not supported
         //
         verifyException("org.joda.time.field.BaseDateTimeField", e);
      }
  }

  @Test(timeout = 4000)
  public void test054()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.yearOfEra();
      localDateTime_Property0.withMinimumValue();
  }

  @Test(timeout = 4000)
  public void test055()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(765L);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.dayOfYear();
      localDateTime_Property0.roundHalfCeilingCopy();
  }

  @Test(timeout = 4000)
  public void test056()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetHoursMinutes(50, 50);
      LocalDateTime localDateTime0 = LocalDateTime.now(dateTimeZone0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.dayOfMonth();
      localDateTime_Property0.addToCopy((long) 50);
  }

  @Test(timeout = 4000)
  public void test057()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetHoursMinutes(50, 50);
      LocalDateTime localDateTime0 = LocalDateTime.now(dateTimeZone0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.centuryOfEra();
      localDateTime_Property0.addWrapFieldToCopy(1519);
  }

  @Test(timeout = 4000)
  public void test058()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0);
      LocalDateTime localDateTime0 = new LocalDateTime(1L, (Chronology) gJChronology0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.dayOfYear();
      localDateTime_Property0.roundCeilingCopy();
  }

  @Test(timeout = 4000)
  public void test059()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetHoursMinutes(50, 50);
      LocalDateTime localDateTime0 = LocalDateTime.now(dateTimeZone0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.centuryOfEra();
      localDateTime_Property0.getLocalDateTime();
  }

  @Test(timeout = 4000)
  public void test060()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetHoursMinutes(50, 50);
      LocalDateTime localDateTime0 = LocalDateTime.now(dateTimeZone0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.dayOfMonth();
      localDateTime_Property0.addToCopy(47);
  }

  @Test(timeout = 4000)
  public void test061()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.weekOfWeekyear();
      LocalDateTime localDateTime0 = new LocalDateTime((long) 1, dateTimeZone0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.property(dateTimeFieldType0);
      localDateTime_Property0.roundHalfFloorCopy();
  }

  @Test(timeout = 4000)
  public void test062()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      LocalDateTime localDateTime0 = new LocalDateTime(64L, (Chronology) buddhistChronology0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.year();
      localDateTime_Property0.getChronology();
  }

  @Test(timeout = 4000)
  public void test063()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      LocalDateTime localDateTime0 = new LocalDateTime(64L, (Chronology) buddhistChronology0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.year();
      LocalDateTime localDateTime1 = localDateTime_Property0.withMaximumValue();
      localDateTime1.toDate();
  }

  @Test(timeout = 4000)
  public void test064()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocalDateTime.now((DateTimeZone) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // Zone must not be null
         //
         verifyException("org.joda.time.LocalDateTime", e);
      }
  }

  @Test(timeout = 4000)
  public void test065()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocalDateTime.now((Chronology) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // Chronology must not be null
         //
         verifyException("org.joda.time.LocalDateTime", e);
      }
  }

  @Test(timeout = 4000)
  public void test066()  throws Throwable  {
      MockGregorianCalendar mockGregorianCalendar0 = new MockGregorianCalendar(2147482664, 2147482664, 2147482664, 2147482664, 2147482664);
      LocalDateTime.fromCalendarFields(mockGregorianCalendar0);
  }

  @Test(timeout = 4000)
  public void test067()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocalDateTime.fromCalendarFields((Calendar) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The calendar must not be null
         //
         verifyException("org.joda.time.LocalDateTime", e);
      }
  }

  @Test(timeout = 4000)
  public void test068()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocalDateTime.fromDateFields((Date) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The date must not be null
         //
         verifyException("org.joda.time.LocalDateTime", e);
      }
  }

  @Test(timeout = 4000)
  public void test069()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      // Undeclared exception!
      try { 
        localDateTime0.getFieldType(99);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Invalid index: 99
         //
         verifyException("org.joda.time.LocalDateTime", e);
      }
  }

  @Test(timeout = 4000)
  public void test070()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      // Undeclared exception!
      try { 
        localDateTime0.getValue(1880);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Invalid index: 1880
         //
         verifyException("org.joda.time.LocalDateTime", e);
      }
  }

  @Test(timeout = 4000)
  public void test071()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      // Undeclared exception!
      try { 
        localDateTime0.get((DateTimeFieldType) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The DateTimeFieldType must not be null
         //
         verifyException("org.joda.time.LocalDateTime", e);
      }
  }

  @Test(timeout = 4000)
  public void test072()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetHoursMinutes(50, 50);
      LocalDateTime localDateTime0 = LocalDateTime.now(dateTimeZone0);
      localDateTime0.isSupported((DateTimeFieldType) null);
  }

  @Test(timeout = 4000)
  public void test073()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0);
      LocalDateTime localDateTime0 = new LocalDateTime((Chronology) gJChronology0);
      DurationFieldType durationFieldType0 = DurationFieldType.centuries();
      localDateTime0.isSupported(durationFieldType0);
  }

  @Test(timeout = 4000)
  public void test074()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.isSupported((DurationFieldType) null);
  }

  @Test(timeout = 4000)
  public void test075()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.equals(localDateTime0);
  }

  @Test(timeout = 4000)
  public void test076()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstanceUTC();
      localDateTime0.equals(ethiopicChronology0);
  }

  @Test(timeout = 4000)
  public void test077()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      LocalDateTime localDateTime0 = new LocalDateTime(54L, (Chronology) buddhistChronology0);
      localDateTime0.toDate();
  }

  @Test(timeout = 4000)
  public void test078()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime((-9223372016640000050L), (Chronology) null);
      localDateTime0.toDate();
  }

  @Test(timeout = 4000)
  public void test079()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.compareTo((ReadablePartial) localDateTime0);
  }

  @Test(timeout = 4000)
  public void test080()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      // Undeclared exception!
      try { 
        localDateTime0.compareTo((ReadablePartial) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.joda.time.base.AbstractPartial", e);
      }
  }

  @Test(timeout = 4000)
  public void test081()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      LocalDateTime localDateTime1 = localDateTime0.plusDays(326565);
      localDateTime0.isBefore(localDateTime1);
  }

  @Test(timeout = 4000)
  public void test082()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.withFields(localDateTime0);
  }

  @Test(timeout = 4000)
  public void test083()  throws Throwable  {
      ISOChronology iSOChronology0 = ISOChronology.getInstance();
      LenientChronology lenientChronology0 = LenientChronology.getInstance(iSOChronology0);
      FixedDateTimeZone fixedDateTimeZone0 = (FixedDateTimeZone)DateTimeZone.UTC;
      ZonedChronology zonedChronology0 = ZonedChronology.getInstance(lenientChronology0, fixedDateTimeZone0);
      LocalDateTime localDateTime0 = new LocalDateTime(3, 3, (-2147483029), (-668), 3, (-668), (-668), zonedChronology0);
      localDateTime0.withFields((ReadablePartial) null);
  }

  @Test(timeout = 4000)
  public void test084()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.clockhourOfDay();
      // Undeclared exception!
      try { 
        localDateTime0.withField(dateTimeFieldType0, 2143477148);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 2143477148 for clockhourOfDay must be in the range [1,24]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test085()  throws Throwable  {
      CopticChronology copticChronology0 = CopticChronology.getInstance();
      LocalDateTime localDateTime0 = LocalDateTime.now((Chronology) copticChronology0);
      // Undeclared exception!
      try { 
        localDateTime0.withField((DateTimeFieldType) null, 1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Field must not be null
         //
         verifyException("org.joda.time.LocalDateTime", e);
      }
  }

  @Test(timeout = 4000)
  public void test086()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      LocalDateTime localDateTime0 = LocalDateTime.now(dateTimeZone0);
      DurationFieldType durationFieldType0 = DurationFieldType.MINUTES_TYPE;
      localDateTime0.withFieldAdded(durationFieldType0, 0);
  }

  @Test(timeout = 4000)
  public void test087()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      // Undeclared exception!
      try { 
        localDateTime0.withFieldAdded((DurationFieldType) null, (-19));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Field must not be null
         //
         verifyException("org.joda.time.LocalDateTime", e);
      }
  }

  @Test(timeout = 4000)
  public void test088()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      LocalDateTime localDateTime0 = LocalDateTime.now(dateTimeZone0);
      DurationFieldType durationFieldType0 = DurationFieldType.MINUTES_TYPE;
      localDateTime0.withFieldAdded(durationFieldType0, 1);
  }

  @Test(timeout = 4000)
  public void test089()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      Duration duration0 = Duration.standardDays(0L);
      localDateTime0.withDurationAdded(duration0, 0);
  }

  @Test(timeout = 4000)
  public void test090()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.withPeriodAdded((ReadablePeriod) null, 1488);
  }

  @Test(timeout = 4000)
  public void test091()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      Weeks weeks0 = Weeks.ONE;
      localDateTime0.withPeriodAdded(weeks0, 0);
  }

  @Test(timeout = 4000)
  public void test092()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.plusYears(788);
  }

  @Test(timeout = 4000)
  public void test093()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.plusYears(0);
  }

  @Test(timeout = 4000)
  public void test094()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.plusMonths(13);
  }

  @Test(timeout = 4000)
  public void test095()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.plusMonths(0);
  }

  @Test(timeout = 4000)
  public void test096()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.plusWeeks(4599);
  }

  @Test(timeout = 4000)
  public void test097()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetHoursMinutes(50, 50);
      LocalDateTime localDateTime0 = LocalDateTime.now(dateTimeZone0);
      localDateTime0.plusWeeks(0);
  }

  @Test(timeout = 4000)
  public void test098()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.plusHours(1);
  }

  @Test(timeout = 4000)
  public void test099()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.plusHours(0);
  }

  @Test(timeout = 4000)
  public void test100()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.plusMinutes(2);
  }

  @Test(timeout = 4000)
  public void test101()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.plusMinutes(0);
  }

  @Test(timeout = 4000)
  public void test102()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      LocalDateTime localDateTime0 = LocalDateTime.now(dateTimeZone0);
      localDateTime0.plusSeconds((-2079));
  }

  @Test(timeout = 4000)
  public void test103()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.plusSeconds(0);
  }

  @Test(timeout = 4000)
  public void test104()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.plusMillis((-1978));
  }

  @Test(timeout = 4000)
  public void test105()  throws Throwable  {
      ISOChronology iSOChronology0 = ISOChronology.getInstanceUTC();
      LocalDateTime localDateTime0 = LocalDateTime.now((Chronology) iSOChronology0);
      localDateTime0.plusMillis(0);
  }

  @Test(timeout = 4000)
  public void test106()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusYears(2);
  }

  @Test(timeout = 4000)
  public void test107()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.minusYears(0);
  }

  @Test(timeout = 4000)
  public void test108()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusMonths((-1));
  }

  @Test(timeout = 4000)
  public void test109()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusMonths(0);
  }

  @Test(timeout = 4000)
  public void test110()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.minusWeeks(2447);
  }

  @Test(timeout = 4000)
  public void test111()  throws Throwable  {
      ISOChronology iSOChronology0 = ISOChronology.getInstanceUTC();
      LocalDateTime localDateTime0 = LocalDateTime.now((Chronology) iSOChronology0);
      localDateTime0.minusWeeks(0);
  }

  @Test(timeout = 4000)
  public void test112()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      LocalDateTime localDateTime0 = new LocalDateTime((long) 1, dateTimeZone0);
      localDateTime0.minusDays(3419);
  }

  @Test(timeout = 4000)
  public void test113()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusDays(0);
  }

  @Test(timeout = 4000)
  public void test114()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetHoursMinutes(50, 50);
      LocalDateTime localDateTime0 = LocalDateTime.now(dateTimeZone0);
      localDateTime0.minusHours(565);
  }

  @Test(timeout = 4000)
  public void test115()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusHours(0);
  }

  @Test(timeout = 4000)
  public void test116()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusMinutes(2147461043);
  }

  @Test(timeout = 4000)
  public void test117()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusMinutes(0);
  }

  @Test(timeout = 4000)
  public void test118()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusSeconds(45);
  }

  @Test(timeout = 4000)
  public void test119()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusSeconds(0);
  }

  @Test(timeout = 4000)
  public void test120()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusMillis((-2142569496));
  }

  @Test(timeout = 4000)
  public void test121()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusMillis(0);
  }

  @Test(timeout = 4000)
  public void test122()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      // Undeclared exception!
      try { 
        localDateTime0.property((DateTimeFieldType) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The DateTimeFieldType must not be null
         //
         verifyException("org.joda.time.LocalDateTime", e);
      }
  }

  @Test(timeout = 4000)
  public void test123()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      // Undeclared exception!
      try { 
        localDateTime0.toString("org.joda.time.format.PeriodFormatterBuilder$Composite");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Illegal pattern component: o
         //
         verifyException("org.joda.time.format.DateTimeFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test124()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.toString((String) null);
  }

  @Test(timeout = 4000)
  public void test125()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      try { 
        localDateTime0.toString("", locale0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid pattern specification
         //
         verifyException("org.joda.time.format.DateTimeFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test126()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0);
      LocalDateTime localDateTime0 = new LocalDateTime(1L, (Chronology) gJChronology0);
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      localDateTime0.toString((String) null, locale0);
  }
}
