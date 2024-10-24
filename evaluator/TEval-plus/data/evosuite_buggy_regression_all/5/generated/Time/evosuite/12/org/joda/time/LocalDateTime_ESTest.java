/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:44:07 GMT 2023
 */

package org.joda.time;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Calendar;
import java.util.Date;
import java.util.Locale;
import java.util.TimeZone;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.util.MockGregorianCalendar;
import org.joda.time.Chronology;
import org.joda.time.DateMidnight;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.joda.time.Duration;
import org.joda.time.DurationFieldType;
import org.joda.time.LocalDate;
import org.joda.time.LocalDateTime;
import org.joda.time.Months;
import org.joda.time.ReadableDuration;
import org.joda.time.ReadablePartial;
import org.joda.time.ReadablePeriod;
import org.joda.time.chrono.EthiopicChronology;
import org.joda.time.chrono.IslamicChronology;
import org.joda.time.tz.FixedDateTimeZone;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class LocalDateTime_ESTest extends LocalDateTime_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test000()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocalDateTime.parse("D^xn5o]#Yq");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid format: \"D^xn5o]#Yq\"
         //
         verifyException("org.joda.time.format.DateTimeFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test001()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      Date date0 = localDateTime0.toDate();
      assertEquals("Fri Feb 14 12:21:21 GMT 2014", date0.toString());
  }

  @Test(timeout = 4000)
  public void test002()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.withMillisOfDay(534);
  }

  @Test(timeout = 4000)
  public void test003()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.withYear(0);
  }

  @Test(timeout = 4000)
  public void test004()  throws Throwable  {
      FixedDateTimeZone fixedDateTimeZone0 = (FixedDateTimeZone)DateTimeZone.UTC;
      LocalDateTime localDateTime0 = new LocalDateTime(1907L, (DateTimeZone) fixedDateTimeZone0);
  }

  @Test(timeout = 4000)
  public void test005()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1);
      localDateTime0.withDayOfMonth(1);
  }

  @Test(timeout = 4000)
  public void test006()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      localDateTime0.toString((String) null, locale0);
  }

  @Test(timeout = 4000)
  public void test007()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      // Undeclared exception!
      try { 
        localDateTime0.withMillisOfSecond((-1638));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -1638 for millisOfSecond must be in the range [0,999]
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
      Months months0 = Months.MIN_VALUE;
      localDateTime0.minus((ReadablePeriod) months0);
  }

  @Test(timeout = 4000)
  public void test010()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.getWeekOfWeekyear();
  }

  @Test(timeout = 4000)
  public void test011()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.withYearOfCentury(1);
  }

  @Test(timeout = 4000)
  public void test012()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1);
      localDateTime0.withDayOfYear(1);
  }

  @Test(timeout = 4000)
  public void test013()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.getEra();
  }

  @Test(timeout = 4000)
  public void test014()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.year();
  }

  @Test(timeout = 4000)
  public void test015()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.getDayOfYear();
  }

  @Test(timeout = 4000)
  public void test016()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.millisOfSecond();
  }

  @Test(timeout = 4000)
  public void test017()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1, 1, 1);
      localDateTime0.withMonthOfYear(1);
  }

  @Test(timeout = 4000)
  public void test018()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.millisOfDay();
      localDateTime_Property0.roundCeilingCopy();
  }

  @Test(timeout = 4000)
  public void test019()  throws Throwable  {
      FixedDateTimeZone fixedDateTimeZone0 = (FixedDateTimeZone)DateTimeZone.UTC;
      LocalDateTime.now((DateTimeZone) fixedDateTimeZone0);
  }

  @Test(timeout = 4000)
  public void test020()  throws Throwable  {
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstanceUTC();
      LocalDateTime localDateTime0 = LocalDateTime.now((Chronology) ethiopicChronology0);
      localDateTime0.toDate();
  }

  @Test(timeout = 4000)
  public void test021()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1);
      localDateTime0.withHourOfDay(1);
  }

  @Test(timeout = 4000)
  public void test022()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.dayOfYear();
  }

  @Test(timeout = 4000)
  public void test023()  throws Throwable  {
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstance();
      FixedDateTimeZone fixedDateTimeZone0 = (FixedDateTimeZone)DateTimeZone.UTC;
      LocalDateTime localDateTime0 = null;
      try {
        localDateTime0 = new LocalDateTime(ethiopicChronology0, fixedDateTimeZone0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // No partial converter found for type: org.joda.time.chrono.EthiopicChronology
         //
         verifyException("org.joda.time.convert.ConverterManager", e);
      }
  }

  @Test(timeout = 4000)
  public void test024()  throws Throwable  {
      DateMidnight dateMidnight0 = new DateMidnight((Object) null);
      Date date0 = dateMidnight0.toDate();
      LocalDateTime localDateTime0 = LocalDateTime.fromDateFields(date0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.yearOfEra();
      localDateTime_Property0.addWrapFieldToCopy((-609));
  }

  @Test(timeout = 4000)
  public void test025()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.minuteOfHour();
  }

  @Test(timeout = 4000)
  public void test026()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.getYearOfCentury();
  }

  @Test(timeout = 4000)
  public void test027()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime((Object) null);
  }

  @Test(timeout = 4000)
  public void test028()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.monthOfYear();
  }

  @Test(timeout = 4000)
  public void test029()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      // Undeclared exception!
      try { 
        localDateTime0.withWeekOfWeekyear(554);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 554 for weekOfWeekyear must be in the range [1,52]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test030()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.weekOfWeekyear();
  }

  @Test(timeout = 4000)
  public void test031()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1);
      localDateTime0.withWeekyear(1);
  }

  @Test(timeout = 4000)
  public void test032()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      // Undeclared exception!
      try { 
        localDateTime0.withDayOfWeek((-2807));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -2807 for dayOfWeek must be in the range [1,7]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test033()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.toDateTime();
  }

  @Test(timeout = 4000)
  public void test034()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.dayOfWeek();
  }

  @Test(timeout = 4000)
  public void test035()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1, 1, 1);
      localDateTime0.withYearOfEra(1);
  }

  @Test(timeout = 4000)
  public void test036()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      Duration duration0 = Duration.ZERO;
      localDateTime0.plus((ReadableDuration) duration0);
  }

  @Test(timeout = 4000)
  public void test037()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1439641281320L);
  }

  @Test(timeout = 4000)
  public void test038()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.getYearOfEra();
  }

  @Test(timeout = 4000)
  public void test039()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.yearOfCentury();
  }

  @Test(timeout = 4000)
  public void test040()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1, 1);
      localDateTime0.minusMonths(1);
  }

  @Test(timeout = 4000)
  public void test041()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.getMillisOfDay();
  }

  @Test(timeout = 4000)
  public void test042()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.withCenturyOfEra(0);
  }

  @Test(timeout = 4000)
  public void test043()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1);
      localDateTime0.withDate(1, 1, 1);
  }

  @Test(timeout = 4000)
  public void test044()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      // Undeclared exception!
      try { 
        localDateTime0.withEra((-4924));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -4924 for era must be in the range [0,1]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test045()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      // Undeclared exception!
      try { 
        localDateTime0.withSecondOfMinute((-1479));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -1479 for secondOfMinute must be in the range [0,59]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test046()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.getCenturyOfEra();
  }

  @Test(timeout = 4000)
  public void test047()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.weekyear();
  }

  @Test(timeout = 4000)
  public void test048()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.era();
  }

  @Test(timeout = 4000)
  public void test049()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.getWeekyear();
  }

  @Test(timeout = 4000)
  public void test050()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      // Undeclared exception!
      try { 
        localDateTime0.withMinuteOfHour(2769);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 2769 for minuteOfHour must be in the range [0,59]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test051()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1);
      localDateTime0.withTime(1, 1, 1, 1);
  }

  @Test(timeout = 4000)
  public void test052()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.plus((ReadablePeriod) null);
  }

  @Test(timeout = 4000)
  public void test053()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.getDayOfWeek();
  }

  @Test(timeout = 4000)
  public void test054()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.toLocalDate();
  }

  @Test(timeout = 4000)
  public void test055()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.secondOfMinute();
      // Undeclared exception!
      try { 
        localDateTime_Property0.setCopy("?IK,r\"/gNW0W)");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value \"?IK,r\"/gNW0W)\" for secondOfMinute is not supported
         //
         verifyException("org.joda.time.field.BaseDateTimeField", e);
      }
  }

  @Test(timeout = 4000)
  public void test056()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forTimeZone((TimeZone) null);
      IslamicChronology islamicChronology0 = IslamicChronology.getInstance(dateTimeZone0);
      LocalDateTime localDateTime0 = LocalDateTime.now((Chronology) islamicChronology0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.hourOfDay();
      localDateTime_Property0.withMinimumValue();
  }

  @Test(timeout = 4000)
  public void test057()  throws Throwable  {
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstance();
      LocalDateTime localDateTime0 = new LocalDateTime((Chronology) ethiopicChronology0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.centuryOfEra();
      localDateTime_Property0.roundHalfCeilingCopy();
  }

  @Test(timeout = 4000)
  public void test058()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forTimeZone((TimeZone) null);
      IslamicChronology islamicChronology0 = IslamicChronology.getInstance(dateTimeZone0);
      LocalDateTime localDateTime0 = LocalDateTime.now((Chronology) islamicChronology0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.hourOfDay();
      localDateTime_Property0.addToCopy((long) 97);
  }

  @Test(timeout = 4000)
  public void test059()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.secondOfMinute();
      localDateTime_Property0.roundHalfEvenCopy();
  }

  @Test(timeout = 4000)
  public void test060()  throws Throwable  {
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstance();
      LocalDateTime localDateTime0 = new LocalDateTime((Chronology) ethiopicChronology0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.dayOfMonth();
      localDateTime_Property0.getLocalDateTime();
  }

  @Test(timeout = 4000)
  public void test061()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.secondOfMinute();
      localDateTime_Property0.addToCopy(2270);
  }

  @Test(timeout = 4000)
  public void test062()  throws Throwable  {
      DateMidnight dateMidnight0 = new DateMidnight((Object) null);
      Date date0 = dateMidnight0.toDate();
      LocalDateTime localDateTime0 = LocalDateTime.fromDateFields(date0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.yearOfEra();
      localDateTime_Property0.roundHalfFloorCopy();
  }

  @Test(timeout = 4000)
  public void test063()  throws Throwable  {
      DateMidnight dateMidnight0 = new DateMidnight((Object) null);
      Date date0 = dateMidnight0.toDate();
      LocalDateTime localDateTime0 = LocalDateTime.fromDateFields(date0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.yearOfEra();
      localDateTime_Property0.roundFloorCopy();
  }

  @Test(timeout = 4000)
  public void test064()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.hourOfDay();
      localDateTime_Property0.getChronology();
  }

  @Test(timeout = 4000)
  public void test065()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.hourOfDay();
      localDateTime_Property0.withMaximumValue();
  }

  @Test(timeout = 4000)
  public void test066()  throws Throwable  {
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
  public void test067()  throws Throwable  {
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
  public void test068()  throws Throwable  {
      MockGregorianCalendar mockGregorianCalendar0 = new MockGregorianCalendar();
      LocalDateTime localDateTime0 = LocalDateTime.fromCalendarFields(mockGregorianCalendar0);
      localDateTime0.minusHours(0);
  }

  @Test(timeout = 4000)
  public void test069()  throws Throwable  {
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
  public void test070()  throws Throwable  {
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
  public void test071()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      // Undeclared exception!
      try { 
        localDateTime0.getFieldType(2442);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Invalid index: 2442
         //
         verifyException("org.joda.time.LocalDateTime", e);
      }
  }

  @Test(timeout = 4000)
  public void test072()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      // Undeclared exception!
      try { 
        localDateTime0.getValue(4);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Invalid index: 4
         //
         verifyException("org.joda.time.LocalDateTime", e);
      }
  }

  @Test(timeout = 4000)
  public void test073()  throws Throwable  {
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
  public void test074()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.isSupported((DateTimeFieldType) null);
  }

  @Test(timeout = 4000)
  public void test075()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      DurationFieldType durationFieldType0 = DurationFieldType.MILLIS_TYPE;
      localDateTime0.isSupported(durationFieldType0);
  }

  @Test(timeout = 4000)
  public void test076()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.isSupported((DurationFieldType) null);
  }

  @Test(timeout = 4000)
  public void test077()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.equals(localDateTime0);
  }

  @Test(timeout = 4000)
  public void test078()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      Duration duration0 = new Duration((-916L), 2269L);
      localDateTime0.equals(duration0);
  }

  @Test(timeout = 4000)
  public void test079()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      DurationFieldType durationFieldType0 = DurationFieldType.years();
      LocalDateTime localDateTime1 = localDateTime0.withFieldAdded(durationFieldType0, (-3056));
      localDateTime1.toDate();
  }

  @Test(timeout = 4000)
  public void test080()  throws Throwable  {
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstance();
      LocalDateTime localDateTime0 = new LocalDateTime((Chronology) ethiopicChronology0);
      localDateTime0.compareTo((ReadablePartial) localDateTime0);
  }

  @Test(timeout = 4000)
  public void test081()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      LocalDate localDate0 = new LocalDate();
      // Undeclared exception!
      try { 
        localDateTime0.isAfter(localDate0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // ReadablePartial objects must have matching field types
         //
         verifyException("org.joda.time.base.AbstractPartial", e);
      }
  }

  @Test(timeout = 4000)
  public void test082()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      LocalDateTime localDateTime1 = localDateTime0.plusMonths(2477);
      localDateTime0.isEqual(localDateTime1);
  }

  @Test(timeout = 4000)
  public void test083()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.withFields(localDateTime0);
  }

  @Test(timeout = 4000)
  public void test084()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.withFields((ReadablePartial) null);
  }

  @Test(timeout = 4000)
  public void test085()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.millisOfDay();
      // Undeclared exception!
      try { 
        localDateTime0.withField(dateTimeFieldType0, (-2322));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -2322 for millisOfDay must be in the range [0,86399999]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test086()  throws Throwable  {
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstance();
      LocalDateTime localDateTime0 = new LocalDateTime((Chronology) ethiopicChronology0);
      // Undeclared exception!
      try { 
        localDateTime0.withField((DateTimeFieldType) null, 478);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Field must not be null
         //
         verifyException("org.joda.time.LocalDateTime", e);
      }
  }

  @Test(timeout = 4000)
  public void test087()  throws Throwable  {
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstance();
      LocalDateTime localDateTime0 = new LocalDateTime((-62104057140000L), (Chronology) ethiopicChronology0);
      // Undeclared exception!
      try { 
        localDateTime0.withFieldAdded((DurationFieldType) null, 470);
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
      LocalDateTime localDateTime0 = LocalDateTime.now();
      DurationFieldType durationFieldType0 = DurationFieldType.HOURS_TYPE;
      localDateTime0.withFieldAdded(durationFieldType0, 0);
  }

  @Test(timeout = 4000)
  public void test089()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.minus((ReadableDuration) null);
  }

  @Test(timeout = 4000)
  public void test090()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      Duration duration0 = Duration.standardHours(594L);
      localDateTime0.withDurationAdded(duration0, 0);
  }

  @Test(timeout = 4000)
  public void test091()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1);
      Months months0 = Months.NINE;
      localDateTime0.withPeriodAdded(months0, 0);
  }

  @Test(timeout = 4000)
  public void test092()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.plusYears((-4266));
  }

  @Test(timeout = 4000)
  public void test093()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.plusYears(0);
  }

  @Test(timeout = 4000)
  public void test094()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.plusMonths(0);
  }

  @Test(timeout = 4000)
  public void test095()  throws Throwable  {
      DateMidnight dateMidnight0 = new DateMidnight((Object) null);
      Date date0 = dateMidnight0.toDate();
      LocalDateTime localDateTime0 = LocalDateTime.fromDateFields(date0);
      localDateTime0.plusWeeks((-413));
  }

  @Test(timeout = 4000)
  public void test096()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.plusWeeks(0);
  }

  @Test(timeout = 4000)
  public void test097()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1);
      localDateTime0.plusDays(1);
  }

  @Test(timeout = 4000)
  public void test098()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.plusDays(0);
  }

  @Test(timeout = 4000)
  public void test099()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.plusHours(1);
  }

  @Test(timeout = 4000)
  public void test100()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.plusHours(0);
  }

  @Test(timeout = 4000)
  public void test101()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1, 1, 1);
      localDateTime0.plusMinutes(1);
  }

  @Test(timeout = 4000)
  public void test102()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.plusMinutes(0);
  }

  @Test(timeout = 4000)
  public void test103()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.plusSeconds((-1676));
  }

  @Test(timeout = 4000)
  public void test104()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.plusSeconds(0);
  }

  @Test(timeout = 4000)
  public void test105()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forTimeZone((TimeZone) null);
      IslamicChronology islamicChronology0 = IslamicChronology.getInstance(dateTimeZone0);
      LocalDateTime localDateTime0 = LocalDateTime.now((Chronology) islamicChronology0);
      localDateTime0.plusMillis(3);
  }

  @Test(timeout = 4000)
  public void test106()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.plusMillis(0);
  }

  @Test(timeout = 4000)
  public void test107()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusYears(729);
  }

  @Test(timeout = 4000)
  public void test108()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusYears(0);
  }

  @Test(timeout = 4000)
  public void test109()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.minusMonths(0);
  }

  @Test(timeout = 4000)
  public void test110()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusWeeks(1);
  }

  @Test(timeout = 4000)
  public void test111()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.minusWeeks(0);
  }

  @Test(timeout = 4000)
  public void test112()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1);
      localDateTime0.minusDays(1);
  }

  @Test(timeout = 4000)
  public void test113()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusDays(0);
  }

  @Test(timeout = 4000)
  public void test114()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.minusHours(1718);
  }

  @Test(timeout = 4000)
  public void test115()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusMinutes((-1095));
  }

  @Test(timeout = 4000)
  public void test116()  throws Throwable  {
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstance();
      LocalDateTime localDateTime0 = new LocalDateTime((Chronology) ethiopicChronology0);
      localDateTime0.minusMinutes(0);
  }

  @Test(timeout = 4000)
  public void test117()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusSeconds(1443);
  }

  @Test(timeout = 4000)
  public void test118()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.minusSeconds(0);
  }

  @Test(timeout = 4000)
  public void test119()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1, 1);
      localDateTime0.minusMillis(1);
  }

  @Test(timeout = 4000)
  public void test120()  throws Throwable  {
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstance();
      LocalDateTime localDateTime0 = new LocalDateTime((Chronology) ethiopicChronology0);
      localDateTime0.minusMillis(0);
  }

  @Test(timeout = 4000)
  public void test121()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.yearOfCentury();
      localDateTime0.property(dateTimeFieldType0);
  }

  @Test(timeout = 4000)
  public void test122()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
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
        localDateTime0.toString("Field must not be null");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Illegal pattern component: F
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
      Locale locale0 = Locale.GERMAN;
      try { 
        localDateTime0.toString("`}cO", locale0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Illegal pattern component: c
         //
         verifyException("org.joda.time.format.DateTimeFormat", e);
      }
  }
}
