/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 18:19:58 GMT 2023
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
import org.evosuite.runtime.mock.java.util.MockDate;
import org.evosuite.runtime.mock.java.util.MockGregorianCalendar;
import org.joda.time.Chronology;
import org.joda.time.DateTime;
import org.joda.time.DateTimeField;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.joda.time.Duration;
import org.joda.time.DurationFieldType;
import org.joda.time.Interval;
import org.joda.time.LocalDate;
import org.joda.time.LocalDateTime;
import org.joda.time.Months;
import org.joda.time.Period;
import org.joda.time.ReadableDuration;
import org.joda.time.ReadablePartial;
import org.joda.time.ReadablePeriod;
import org.joda.time.Years;
import org.joda.time.chrono.BuddhistChronology;
import org.joda.time.chrono.CopticChronology;
import org.joda.time.chrono.GJChronology;
import org.joda.time.tz.FixedDateTimeZone;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class LocalDateTime_ESTest extends LocalDateTime_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test000()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocalDateTime.parse("America/New_York");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid format: \"America/New_York\"
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
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1, 1, 1);
      localDateTime0.withMillisOfDay(1);
  }

  @Test(timeout = 4000)
  public void test004()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.withYear(1);
  }

  @Test(timeout = 4000)
  public void test005()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      // Undeclared exception!
      try { 
        localDateTime0.withDayOfMonth(0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 0 for dayOfMonth must be in the range [1,28]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test006()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.withMillisOfSecond(1);
  }

  @Test(timeout = 4000)
  public void test007()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.toLocalTime();
  }

  @Test(timeout = 4000)
  public void test008()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minus((ReadablePeriod) null);
  }

  @Test(timeout = 4000)
  public void test009()  throws Throwable  {
      MockGregorianCalendar mockGregorianCalendar0 = new MockGregorianCalendar();
      LocalDateTime localDateTime0 = LocalDateTime.fromCalendarFields(mockGregorianCalendar0);
      localDateTime0.getWeekOfWeekyear();
  }

  @Test(timeout = 4000)
  public void test010()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.withYearOfCentury(1);
  }

  @Test(timeout = 4000)
  public void test011()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1);
      localDateTime0.withDayOfYear(1);
  }

  @Test(timeout = 4000)
  public void test012()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.getEra();
  }

  @Test(timeout = 4000)
  public void test013()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.dayOfMonth();
  }

  @Test(timeout = 4000)
  public void test014()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.getDayOfYear();
  }

  @Test(timeout = 4000)
  public void test015()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.millisOfSecond();
  }

  @Test(timeout = 4000)
  public void test016()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.withMonthOfYear(1);
  }

  @Test(timeout = 4000)
  public void test017()  throws Throwable  {
      CopticChronology copticChronology0 = CopticChronology.getInstanceUTC();
      LocalDateTime localDateTime0 = new LocalDateTime((Chronology) copticChronology0);
      localDateTime0.toDate();
  }

  @Test(timeout = 4000)
  public void test018()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.withHourOfDay(5);
  }

  @Test(timeout = 4000)
  public void test019()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.dayOfYear();
  }

  @Test(timeout = 4000)
  public void test020()  throws Throwable  {
      Object object0 = new Object();
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      LocalDateTime localDateTime0 = null;
      try {
        localDateTime0 = new LocalDateTime(object0, dateTimeZone0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // No partial converter found for type: java.lang.Object
         //
         verifyException("org.joda.time.convert.ConverterManager", e);
      }
  }

  @Test(timeout = 4000)
  public void test021()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.yearOfEra();
  }

  @Test(timeout = 4000)
  public void test022()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.getYearOfCentury();
  }

  @Test(timeout = 4000)
  public void test023()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime((Object) null);
  }

  @Test(timeout = 4000)
  public void test024()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.monthOfYear();
  }

  @Test(timeout = 4000)
  public void test025()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      // Undeclared exception!
      try { 
        localDateTime0.withWeekOfWeekyear(1777);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 1777 for weekOfWeekyear must be in the range [1,52]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test026()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.weekOfWeekyear();
  }

  @Test(timeout = 4000)
  public void test027()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.withWeekyear((-631));
  }

  @Test(timeout = 4000)
  public void test028()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      LocalDateTime localDateTime0 = new LocalDateTime(dateTimeZone0);
      localDateTime0.withDayOfWeek(1);
  }

  @Test(timeout = 4000)
  public void test029()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.toDateTime();
  }

  @Test(timeout = 4000)
  public void test030()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.withYearOfEra(1);
  }

  @Test(timeout = 4000)
  public void test031()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.plus((ReadableDuration) null);
  }

  @Test(timeout = 4000)
  public void test032()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.getYearOfEra();
  }

  @Test(timeout = 4000)
  public void test033()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.yearOfCentury();
      localDateTime_Property0.getChronology();
  }

  @Test(timeout = 4000)
  public void test034()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.centuryOfEra();
  }

  @Test(timeout = 4000)
  public void test035()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1, 1);
      localDateTime0.minusYears(1);
  }

  @Test(timeout = 4000)
  public void test036()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.getMillisOfDay();
  }

  @Test(timeout = 4000)
  public void test037()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      // Undeclared exception!
      try { 
        localDateTime0.withCenturyOfEra((-31));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -31 for centuryOfEra must be in the range [0,2922789]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test038()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      // Undeclared exception!
      try { 
        localDateTime0.withDate(3687, 3687, 3687);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 3687 for monthOfYear must be in the range [1,12]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test039()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      // Undeclared exception!
      try { 
        localDateTime0.withEra((-2004318070));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -2004318070 for era must be in the range [0,1]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test040()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      // Undeclared exception!
      try { 
        localDateTime0.withSecondOfMinute((-998));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -998 for secondOfMinute must be in the range [0,59]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test041()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.getCenturyOfEra();
  }

  @Test(timeout = 4000)
  public void test042()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.weekyear();
  }

  @Test(timeout = 4000)
  public void test043()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.era();
  }

  @Test(timeout = 4000)
  public void test044()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.getWeekyear();
  }

  @Test(timeout = 4000)
  public void test045()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      // Undeclared exception!
      try { 
        localDateTime0.withMinuteOfHour(988);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 988 for minuteOfHour must be in the range [0,59]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test046()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      // Undeclared exception!
      try { 
        localDateTime0.withTime((-968), (-968), (-968), (-968));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -968 for hourOfDay must be in the range [0,23]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test047()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      Months months0 = Months.TWO;
      localDateTime0.plus((ReadablePeriod) months0);
  }

  @Test(timeout = 4000)
  public void test048()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.getDayOfWeek();
  }

  @Test(timeout = 4000)
  public void test049()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.toLocalDate();
  }

  @Test(timeout = 4000)
  public void test050()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(0L);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.year();
      // Undeclared exception!
      try { 
        localDateTime_Property0.setCopy((String) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value null for year is not supported
         //
         verifyException("org.joda.time.field.BaseDateTimeField", e);
      }
  }

  @Test(timeout = 4000)
  public void test051()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      LocalDateTime localDateTime0 = new LocalDateTime((-2122675199988L), dateTimeZone0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.minuteOfHour();
      localDateTime_Property0.withMinimumValue();
  }

  @Test(timeout = 4000)
  public void test052()  throws Throwable  {
      CopticChronology copticChronology0 = CopticChronology.getInstanceUTC();
      LocalDateTime localDateTime0 = LocalDateTime.now((Chronology) copticChronology0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.secondOfMinute();
      localDateTime_Property0.roundHalfCeilingCopy();
  }

  @Test(timeout = 4000)
  public void test053()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      LocalDateTime localDateTime0 = new LocalDateTime(dateTimeZone0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.millisOfDay();
      localDateTime_Property0.addToCopy(0L);
  }

  @Test(timeout = 4000)
  public void test054()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime((-927L));
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.dayOfWeek();
      localDateTime_Property0.addWrapFieldToCopy(4);
  }

  @Test(timeout = 4000)
  public void test055()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      FixedDateTimeZone fixedDateTimeZone0 = (FixedDateTimeZone)DateTimeZone.UTC;
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance((DateTimeZone) fixedDateTimeZone0);
      DateTimeField dateTimeField0 = buddhistChronology0.minuteOfHour();
      LocalDateTime.Property localDateTime_Property0 = new LocalDateTime.Property(localDateTime0, dateTimeField0);
      localDateTime_Property0.roundHalfEvenCopy();
  }

  @Test(timeout = 4000)
  public void test056()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime((long) 1964);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.secondOfMinute();
      localDateTime_Property0.roundCeilingCopy();
  }

  @Test(timeout = 4000)
  public void test057()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime((-927L));
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.dayOfWeek();
      localDateTime_Property0.getLocalDateTime();
  }

  @Test(timeout = 4000)
  public void test058()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime((-927L));
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.dayOfWeek();
      localDateTime_Property0.addToCopy(292271022);
  }

  @Test(timeout = 4000)
  public void test059()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime((-927L));
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.dayOfWeek();
      localDateTime_Property0.roundHalfFloorCopy();
  }

  @Test(timeout = 4000)
  public void test060()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime((-927L));
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.dayOfWeek();
      localDateTime_Property0.roundFloorCopy();
  }

  @Test(timeout = 4000)
  public void test061()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstanceUTC();
      LocalDateTime localDateTime0 = new LocalDateTime(0L, (Chronology) buddhistChronology0);
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.yearOfCentury();
      localDateTime_Property0.withMaximumValue();
  }

  @Test(timeout = 4000)
  public void test062()  throws Throwable  {
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
  public void test063()  throws Throwable  {
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
  public void test064()  throws Throwable  {
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
  public void test065()  throws Throwable  {
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
  public void test066()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      // Undeclared exception!
      try { 
        localDateTime0.getField((-4859));
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Invalid index: -4859
         //
         verifyException("org.joda.time.LocalDateTime", e);
      }
  }

  @Test(timeout = 4000)
  public void test067()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      // Undeclared exception!
      try { 
        localDateTime0.getValue(93);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Invalid index: 93
         //
         verifyException("org.joda.time.LocalDateTime", e);
      }
  }

  @Test(timeout = 4000)
  public void test068()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
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
  public void test069()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.isSupported((DateTimeFieldType) null);
  }

  @Test(timeout = 4000)
  public void test070()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      DurationFieldType durationFieldType0 = DurationFieldType.minutes();
      localDateTime0.isSupported(durationFieldType0);
  }

  @Test(timeout = 4000)
  public void test071()  throws Throwable  {
      CopticChronology copticChronology0 = CopticChronology.getInstanceUTC();
      LocalDateTime localDateTime0 = LocalDateTime.now((Chronology) copticChronology0);
      localDateTime0.isSupported((DurationFieldType) null);
  }

  @Test(timeout = 4000)
  public void test072()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1);
      localDateTime0.equals(localDateTime0);
  }

  @Test(timeout = 4000)
  public void test073()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1, 1, 1, 1, 1);
      GJChronology gJChronology0 = GJChronology.getInstance();
      localDateTime0.equals(gJChronology0);
  }

  @Test(timeout = 4000)
  public void test074()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.centuryOfEra();
      LocalDateTime.Property localDateTime_Property0 = localDateTime0.property(dateTimeFieldType0);
      Interval interval0 = localDateTime_Property0.toInterval();
      Period period0 = interval0.toPeriod();
      LocalDateTime localDateTime1 = localDateTime0.withPeriodAdded(period0, (-1696));
      localDateTime1.toDate();
  }

  @Test(timeout = 4000)
  public void test075()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.isAfter(localDateTime0);
  }

  @Test(timeout = 4000)
  public void test076()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      LocalDate localDate0 = new LocalDate();
      // Undeclared exception!
      try { 
        localDateTime0.compareTo((ReadablePartial) localDate0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // ReadablePartial objects must have matching field types
         //
         verifyException("org.joda.time.base.AbstractPartial", e);
      }
  }

  @Test(timeout = 4000)
  public void test077()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      LocalDateTime localDateTime1 = localDateTime0.plusDays(2);
      localDateTime0.compareTo((ReadablePartial) localDateTime1);
  }

  @Test(timeout = 4000)
  public void test078()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      LocalDateTime localDateTime0 = new LocalDateTime(dateTimeZone0);
      localDateTime0.withFields(localDateTime0);
  }

  @Test(timeout = 4000)
  public void test079()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.withFields((ReadablePartial) null);
  }

  @Test(timeout = 4000)
  public void test080()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfMinute();
      // Undeclared exception!
      try { 
        localDateTime0.withField(dateTimeFieldType0, (-1039));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -1039 for secondOfMinute must be in the range [0,59]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test081()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      // Undeclared exception!
      try { 
        localDateTime0.withField((DateTimeFieldType) null, 11);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Field must not be null
         //
         verifyException("org.joda.time.LocalDateTime", e);
      }
  }

  @Test(timeout = 4000)
  public void test082()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      LocalDateTime localDateTime0 = new LocalDateTime((-2122675199988L), dateTimeZone0);
      DurationFieldType durationFieldType0 = DurationFieldType.DAYS_TYPE;
      localDateTime0.withFieldAdded(durationFieldType0, (-1610612735));
  }

  @Test(timeout = 4000)
  public void test083()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      // Undeclared exception!
      try { 
        localDateTime0.withFieldAdded((DurationFieldType) null, 2344);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Field must not be null
         //
         verifyException("org.joda.time.LocalDateTime", e);
      }
  }

  @Test(timeout = 4000)
  public void test084()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      DurationFieldType durationFieldType0 = DurationFieldType.DAYS_TYPE;
      localDateTime0.withFieldAdded(durationFieldType0, 0);
  }

  @Test(timeout = 4000)
  public void test085()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstanceUTC();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      LocalDateTime localDateTime0 = LocalDateTime.now(dateTimeZone0);
      Duration duration0 = Duration.standardHours(1000L);
      localDateTime0.minus((ReadableDuration) duration0);
  }

  @Test(timeout = 4000)
  public void test086()  throws Throwable  {
      Duration duration0 = Duration.ZERO;
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.withDurationAdded(duration0, 0);
  }

  @Test(timeout = 4000)
  public void test087()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      Years years0 = Years.TWO;
      localDateTime0.withPeriodAdded(years0, 0);
  }

  @Test(timeout = 4000)
  public void test088()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.plusYears(3805);
  }

  @Test(timeout = 4000)
  public void test089()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstanceUTC();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      LocalDateTime localDateTime0 = LocalDateTime.now(dateTimeZone0);
      localDateTime0.plusYears(0);
  }

  @Test(timeout = 4000)
  public void test090()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      LocalDateTime localDateTime0 = new LocalDateTime(dateTimeZone0);
      localDateTime0.plusMonths((-2327));
  }

  @Test(timeout = 4000)
  public void test091()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.plusMonths(0);
  }

  @Test(timeout = 4000)
  public void test092()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.plusWeeks(1);
  }

  @Test(timeout = 4000)
  public void test093()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.plusWeeks(0);
  }

  @Test(timeout = 4000)
  public void test094()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(0L);
      localDateTime0.plusDays(0);
  }

  @Test(timeout = 4000)
  public void test095()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      LocalDateTime localDateTime0 = new LocalDateTime((-2122675199988L), dateTimeZone0);
      localDateTime0.plusHours((-1610612735));
  }

  @Test(timeout = 4000)
  public void test096()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.plusHours(0);
  }

  @Test(timeout = 4000)
  public void test097()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      LocalDateTime localDateTime0 = new LocalDateTime(dateTimeZone0);
      localDateTime0.plusMinutes(1);
  }

  @Test(timeout = 4000)
  public void test098()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.plusMinutes(0);
  }

  @Test(timeout = 4000)
  public void test099()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      LocalDateTime localDateTime0 = new LocalDateTime((-2122675199988L), dateTimeZone0);
      localDateTime0.plusSeconds((-1610612735));
  }

  @Test(timeout = 4000)
  public void test100()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.plusSeconds(0);
  }

  @Test(timeout = 4000)
  public void test101()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime((long) 1964);
      localDateTime0.plusMillis(128);
  }

  @Test(timeout = 4000)
  public void test102()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(1578L);
      localDateTime0.plusMillis(0);
  }

  @Test(timeout = 4000)
  public void test103()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.minusYears(0);
  }

  @Test(timeout = 4000)
  public void test104()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusMonths(2147483634);
  }

  @Test(timeout = 4000)
  public void test105()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusMonths(0);
  }

  @Test(timeout = 4000)
  public void test106()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusWeeks(1);
  }

  @Test(timeout = 4000)
  public void test107()  throws Throwable  {
      MockDate mockDate0 = new MockDate((-25), (-25), 1000);
      LocalDateTime localDateTime0 = LocalDateTime.fromDateFields(mockDate0);
      localDateTime0.minusWeeks(0);
  }

  @Test(timeout = 4000)
  public void test108()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.minusDays((-839));
  }

  @Test(timeout = 4000)
  public void test109()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusDays(0);
  }

  @Test(timeout = 4000)
  public void test110()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      LocalDateTime localDateTime0 = new LocalDateTime(dateTimeZone0);
      localDateTime0.minusHours(1);
  }

  @Test(timeout = 4000)
  public void test111()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(0L);
      localDateTime0.minusHours(0);
  }

  @Test(timeout = 4000)
  public void test112()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusMinutes(2);
  }

  @Test(timeout = 4000)
  public void test113()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusMinutes(0);
  }

  @Test(timeout = 4000)
  public void test114()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime(0L);
      localDateTime0.minusSeconds(1467);
  }

  @Test(timeout = 4000)
  public void test115()  throws Throwable  {
      DateTime dateTime0 = DateTime.now();
      LocalDateTime localDateTime0 = dateTime0.toLocalDateTime();
      localDateTime0.minusSeconds(0);
  }

  @Test(timeout = 4000)
  public void test116()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      localDateTime0.minusMillis((-2143222766));
  }

  @Test(timeout = 4000)
  public void test117()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.minusMillis(0);
  }

  @Test(timeout = 4000)
  public void test118()  throws Throwable  {
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
  public void test119()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime((long) 1964);
      // Undeclared exception!
      try { 
        localDateTime0.toString("Df=wJ");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Illegal pattern component: f
         //
         verifyException("org.joda.time.format.DateTimeFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test120()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      localDateTime0.toString((String) null);
  }

  @Test(timeout = 4000)
  public void test121()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      Locale locale0 = Locale.GERMANY;
      try { 
        localDateTime0.toString("kcMcBLlI4xSY", locale0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Illegal pattern component: c
         //
         verifyException("org.joda.time.format.DateTimeFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test122()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      Locale locale0 = Locale.GERMANY;
      localDateTime0.toString((String) null, locale0);
  }
}