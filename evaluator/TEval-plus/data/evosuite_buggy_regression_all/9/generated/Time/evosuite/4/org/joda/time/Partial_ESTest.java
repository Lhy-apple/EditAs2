/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:58:44 GMT 2023
 */

package org.joda.time;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.util.Locale;
import java.util.TimeZone;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.joda.time.Chronology;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.joda.time.Duration;
import org.joda.time.DurationFieldType;
import org.joda.time.Instant;
import org.joda.time.Minutes;
import org.joda.time.MonthDay;
import org.joda.time.MutablePeriod;
import org.joda.time.Partial;
import org.joda.time.Period;
import org.joda.time.PeriodType;
import org.joda.time.ReadableInstant;
import org.joda.time.ReadablePartial;
import org.joda.time.ReadablePeriod;
import org.joda.time.YearMonth;
import org.joda.time.chrono.BuddhistChronology;
import org.joda.time.chrono.CopticChronology;
import org.joda.time.chrono.GJChronology;
import org.joda.time.chrono.GregorianChronology;
import org.joda.time.chrono.ISOChronology;
import org.joda.time.chrono.IslamicChronology;
import org.joda.time.chrono.StrictChronology;
import org.joda.time.format.DateTimeFormatter;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Partial_ESTest extends Partial_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[1];
      int[] intArray0 = new int[1];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must not contain null: index 0
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Partial partial0 = new Partial();
      assertEquals(0, partial0.size());
      
      CopticChronology copticChronology0 = CopticChronology.getInstanceUTC();
      Partial partial1 = partial0.withChronologyRetainFields(copticChronology0);
      assertNotSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      YearMonth yearMonth0 = YearMonth.now();
      Partial partial0 = new Partial(yearMonth0);
      Locale locale0 = Locale.JAPAN;
      String string0 = partial0.toString((String) null, locale0);
      assertEquals("2014-02", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      YearMonth yearMonth0 = YearMonth.now();
      Partial partial0 = new Partial(yearMonth0);
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.monthOfYear();
      Partial partial1 = partial0.with(dateTimeFieldType0, 1);
      assertEquals(2, partial1.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Partial partial0 = new Partial();
      Minutes minutes0 = Minutes.MAX_VALUE;
      Partial partial1 = partial0.plus(minutes0);
      assertTrue(partial1.equals((Object)partial0));
      assertNotSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Partial partial0 = new Partial();
      MutablePeriod mutablePeriod0 = new MutablePeriod();
      Partial partial1 = partial0.minus(mutablePeriod0);
      assertNotSame(partial1, partial0);
      assertTrue(partial1.equals((Object)partial0));
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.centuryOfEra();
      Partial partial0 = new Partial(dateTimeFieldType0, 0);
      DateTimeFieldType[] dateTimeFieldTypeArray0 = partial0.getFieldTypes();
      assertEquals(1, dateTimeFieldTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[6];
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.era();
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstanceUTC();
      Partial partial0 = new Partial(dateTimeFieldType0, 1, buddhistChronology0);
      Partial.Property partial_Property0 = partial0.property(dateTimeFieldTypeArray0[0]);
      // Undeclared exception!
      try { 
        partial_Property0.setCopy("[yearOfCentury=1, yearOfEra=-1909]");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value \"[yearOfCentury=1, yearOfEra=-1909]\" for era is not supported
         //
         verifyException("org.joda.time.field.BaseDateTimeField", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.halfdayOfDay();
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      Partial partial0 = new Partial(dateTimeFieldType0, 0, gJChronology0);
      Partial.Property partial_Property0 = partial0.property(dateTimeFieldType0);
      int int0 = partial_Property0.get();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ISOChronology iSOChronology0 = ISOChronology.getInstance();
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfDay();
      Partial partial0 = new Partial(dateTimeFieldType0, 1040, iSOChronology0);
      Partial.Property partial_Property0 = new Partial.Property(partial0, 1040);
      // Undeclared exception!
      try { 
        partial_Property0.addWrapFieldToCopy(3);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 1040
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      YearMonth yearMonth0 = new YearMonth();
      Partial partial0 = new Partial(yearMonth0);
      Partial.Property partial_Property0 = new Partial.Property(partial0, 1);
      Partial partial1 = partial_Property0.withMaximumValue();
      assertEquals(2, partial1.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.halfdayOfDay();
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      Partial partial0 = new Partial(dateTimeFieldType0, 0, gJChronology0);
      Partial.Property partial_Property0 = partial0.property(dateTimeFieldType0);
      Partial partial1 = partial_Property0.getPartial();
      assertSame(partial0, partial1);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Partial partial0 = new Partial();
      Partial.Property partial_Property0 = new Partial.Property(partial0, 1);
      // Undeclared exception!
      try { 
        partial_Property0.addToCopy(0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 1
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ISOChronology iSOChronology0 = ISOChronology.getInstance();
      YearMonth yearMonth0 = new YearMonth((Chronology) iSOChronology0);
      Partial partial0 = new Partial(yearMonth0);
      Partial.Property partial_Property0 = new Partial.Property(partial0, 1598);
      // Undeclared exception!
      try { 
        partial_Property0.withMinimumValue();
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 1598
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[4];
      IslamicChronology islamicChronology0 = IslamicChronology.getInstanceUTC();
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0[3], (-1849), islamicChronology0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The field type must not be null
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      int[] intArray0 = new int[0];
      Partial partial0 = null;
      try {
        partial0 = new Partial((DateTimeFieldType[]) null, intArray0, (Chronology) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must not be null
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[1];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, (int[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Values array must not be null
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[4];
      int[] intArray0 = new int[1];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Values array must be the same length as the types array
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[0];
      int[] intArray0 = new int[0];
      Partial partial0 = new Partial(dateTimeFieldTypeArray0, intArray0, (Chronology) null);
      assertEquals(0, partial0.size());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfMinute();
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.yearOfCentury();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[4];
      dateTimeFieldTypeArray0[0] = dateTimeFieldType1;
      dateTimeFieldTypeArray0[1] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[2] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[3] = dateTimeFieldType0;
      int[] intArray0 = new int[4];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must not contain duplicate: secondOfMinute
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[6];
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.weekyearOfCentury();
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.halfdayOfDay();
      dateTimeFieldTypeArray0[1] = dateTimeFieldType1;
      DateTimeFieldType dateTimeFieldType2 = DateTimeFieldType.era();
      dateTimeFieldTypeArray0[2] = dateTimeFieldType2;
      dateTimeFieldTypeArray0[3] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[4] = dateTimeFieldTypeArray0[2];
      dateTimeFieldTypeArray0[5] = dateTimeFieldTypeArray0[0];
      int[] intArray0 = new int[6];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must be in order largest-smallest: halfdayOfDay < era
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.millisOfSecond();
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.weekyearOfCentury();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[6];
      DateTimeFieldType dateTimeFieldType2 = DateTimeFieldType.weekyear();
      dateTimeFieldTypeArray0[0] = dateTimeFieldType2;
      dateTimeFieldTypeArray0[1] = dateTimeFieldType1;
      dateTimeFieldTypeArray0[2] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[3] = dateTimeFieldType1;
      dateTimeFieldTypeArray0[4] = dateTimeFieldTypeArray0[2];
      dateTimeFieldTypeArray0[5] = dateTimeFieldType0;
      int[] intArray0 = new int[6];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must be in order largest-smallest: millisOfSecond < weekyearOfCentury
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.year();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[4];
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[1] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[2] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[3] = dateTimeFieldType0;
      int[] intArray0 = new int[4];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must not contain duplicate: year
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.year();
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.yearOfCentury();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[4];
      dateTimeFieldTypeArray0[0] = dateTimeFieldType1;
      dateTimeFieldTypeArray0[1] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[2] = dateTimeFieldType1;
      dateTimeFieldTypeArray0[3] = dateTimeFieldType0;
      int[] intArray0 = new int[4];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must be in order largest-smallest: yearOfCentury < year
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[6];
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.weekyear();
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.dayOfYear();
      dateTimeFieldTypeArray0[1] = dateTimeFieldType1;
      DateTimeFieldType dateTimeFieldType2 = DateTimeFieldType.dayOfWeek();
      dateTimeFieldTypeArray0[2] = dateTimeFieldType2;
      dateTimeFieldTypeArray0[3] = dateTimeFieldTypeArray0[1];
      dateTimeFieldTypeArray0[4] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[5] = dateTimeFieldTypeArray0[4];
      int[] intArray0 = new int[6];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0, (Chronology) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must be in order largest-smallest: dayOfWeek < dayOfYear
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Partial partial0 = null;
      try {
        partial0 = new Partial((ReadablePartial) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The partial must not be null
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      YearMonth yearMonth0 = YearMonth.now();
      Partial partial0 = new Partial(yearMonth0);
      ISOChronology iSOChronology0 = ISOChronology.getInstance();
      Partial partial1 = partial0.withChronologyRetainFields(iSOChronology0);
      assertSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      YearMonth yearMonth0 = new YearMonth();
      Partial partial0 = new Partial(yearMonth0);
      // Undeclared exception!
      try { 
        partial0.with((DateTimeFieldType) null, 653);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The field type must not be null
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      YearMonth yearMonth0 = YearMonth.now();
      Partial partial0 = new Partial(yearMonth0);
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.era();
      Partial partial1 = partial0.with(dateTimeFieldType0, 1);
      assertEquals(3, partial1.size());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.minuteOfHour();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[1];
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.era();
      dateTimeFieldTypeArray0[0] = dateTimeFieldType1;
      int[] intArray0 = new int[1];
      Partial partial0 = new Partial(dateTimeFieldTypeArray0, intArray0);
      // Undeclared exception!
      try { 
        partial0.with(dateTimeFieldType0, 1730);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 1730 for minuteOfHour must not be larger than 59
         //
         verifyException("org.joda.time.chrono.BaseChronology", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      YearMonth yearMonth0 = YearMonth.now();
      Partial partial0 = new Partial(yearMonth0);
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.centuryOfEra();
      Partial partial1 = partial0.with(dateTimeFieldType0, 1);
      assertEquals(3, partial1.size());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ISOChronology iSOChronology0 = ISOChronology.getInstance();
      MonthDay monthDay0 = new MonthDay((Chronology) iSOChronology0);
      Partial partial0 = new Partial(monthDay0);
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.dayOfYear();
      Partial partial1 = partial0.with(dateTimeFieldType0, 1);
      assertEquals(3, partial1.size());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      ISOChronology iSOChronology0 = ISOChronology.getInstance();
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.yearOfCentury();
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.yearOfEra();
      Partial partial0 = new Partial(dateTimeFieldType0, 1, iSOChronology0);
      // Undeclared exception!
      try { 
        partial0.with(dateTimeFieldType1, 0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 0 for yearOfEra must not be smaller than 1
         //
         verifyException("org.joda.time.chrono.BaseChronology", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfMinute();
      Partial partial0 = new Partial(dateTimeFieldType0, 0);
      Partial partial1 = partial0.with(dateTimeFieldType0, 0);
      assertEquals(1, partial1.size());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Partial partial0 = new Partial();
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.dayOfYear();
      Partial partial1 = partial0.without(dateTimeFieldType0);
      assertSame(partial0, partial1);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfMinute();
      Partial partial0 = new Partial(dateTimeFieldType0, 0);
      Partial partial1 = partial0.without(dateTimeFieldType0);
      assertEquals(0, partial1.size());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfMinute();
      Partial partial0 = new Partial(dateTimeFieldType0, 0);
      Partial partial1 = partial0.withField(dateTimeFieldType0, 22);
      assertNotSame(partial1, partial0);
      assertFalse(partial1.equals((Object)partial0));
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfMinute();
      GregorianChronology gregorianChronology0 = GregorianChronology.getInstance();
      StrictChronology strictChronology0 = StrictChronology.getInstance(gregorianChronology0);
      Partial partial0 = new Partial(dateTimeFieldType0, 0, strictChronology0);
      Partial partial1 = partial0.withField(dateTimeFieldType0, 0);
      assertSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfMinute();
      DurationFieldType durationFieldType0 = dateTimeFieldType0.getDurationType();
      Partial partial0 = new Partial(dateTimeFieldType0, 7);
      Partial partial1 = partial0.withFieldAdded(durationFieldType0, 7);
      assertNotSame(partial1, partial0);
      assertFalse(partial1.equals((Object)partial0));
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfMinute();
      DurationFieldType durationFieldType0 = dateTimeFieldType0.getDurationType();
      Partial partial0 = new Partial(dateTimeFieldType0, 0);
      Partial partial1 = partial0.withFieldAdded(durationFieldType0, 0);
      assertSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.year();
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      Partial partial0 = new Partial(dateTimeFieldType0, (-180), buddhistChronology0);
      DurationFieldType durationFieldType0 = dateTimeFieldType0.getDurationType();
      Partial partial1 = partial0.withFieldAddWrapped(durationFieldType0, (-180));
      assertNotSame(partial1, partial0);
      assertFalse(partial1.equals((Object)partial0));
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.millisOfSecond();
      DurationFieldType durationFieldType0 = dateTimeFieldType0.getDurationType();
      GregorianChronology gregorianChronology0 = GregorianChronology.getInstance();
      StrictChronology strictChronology0 = StrictChronology.getInstance(gregorianChronology0);
      Partial partial0 = new Partial(dateTimeFieldType0, 0, strictChronology0);
      Partial partial1 = partial0.withFieldAddWrapped(durationFieldType0, 0);
      assertSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      YearMonth yearMonth0 = new YearMonth();
      Partial partial0 = new Partial(yearMonth0);
      Partial partial1 = partial0.withPeriodAdded((ReadablePeriod) null, 1);
      assertSame(partial0, partial1);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      ISOChronology iSOChronology0 = ISOChronology.getInstance();
      YearMonth yearMonth0 = new YearMonth((Chronology) iSOChronology0);
      Partial partial0 = new Partial(yearMonth0);
      Duration duration0 = Duration.standardSeconds(416L);
      PeriodType periodType0 = PeriodType.days();
      Period period0 = duration0.toPeriod(periodType0);
      Partial partial1 = partial0.withPeriodAdded(period0, 0);
      assertSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetMillis(892);
      GJChronology gJChronology0 = GJChronology.getInstance(dateTimeZone0);
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[1];
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.clockhourOfDay();
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      int[] intArray0 = new int[8];
      Partial partial0 = new Partial(gJChronology0, dateTimeFieldTypeArray0, intArray0);
      Period period0 = Period.weeks(892);
      Partial partial1 = partial0.withPeriodAdded(period0, 892);
      assertTrue(partial1.equals((Object)partial0));
      assertNotSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      YearMonth yearMonth0 = YearMonth.now();
      Partial partial0 = new Partial(yearMonth0);
      boolean boolean0 = partial0.isMatch((ReadableInstant) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Partial partial0 = new Partial();
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.millisOfDay();
      Partial partial1 = partial0.with(dateTimeFieldType0, 93);
      Instant instant0 = Instant.now();
      boolean boolean0 = partial1.isMatch((ReadableInstant) instant0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      YearMonth yearMonth0 = YearMonth.now();
      Partial partial0 = new Partial(yearMonth0);
      // Undeclared exception!
      try { 
        partial0.isMatch((ReadablePartial) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The partial must not be null
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfMinute();
      Partial partial0 = new Partial(dateTimeFieldType0, 0);
      boolean boolean0 = partial0.isMatch((ReadablePartial) partial0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      IslamicChronology islamicChronology0 = IslamicChronology.getInstance((DateTimeZone) null);
      YearMonth yearMonth0 = new YearMonth((Chronology) islamicChronology0);
      Partial partial0 = new Partial(yearMonth0);
      Partial.Property partial_Property0 = new Partial.Property(partial0, 0);
      Partial partial1 = partial_Property0.setCopy(1);
      boolean boolean0 = partial0.isMatch((ReadablePartial) partial1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.year();
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      Partial partial0 = new Partial(dateTimeFieldType0, (-180), buddhistChronology0);
      partial0.getFormatter();
      DateTimeFormatter dateTimeFormatter0 = partial0.getFormatter();
      assertEquals(2000, dateTimeFormatter0.getDefaultYear());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Partial partial0 = new Partial();
      String string0 = partial0.toString((String) null);
      assertEquals("[]", string0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfMinute();
      ZoneOffset zoneOffset0 = ZoneOffset.UTC;
      TimeZone timeZone0 = TimeZone.getTimeZone((ZoneId) zoneOffset0);
      DateTimeZone dateTimeZone0 = DateTimeZone.forTimeZone(timeZone0);
      IslamicChronology.LeapYearPatternType islamicChronology_LeapYearPatternType0 = IslamicChronology.LEAP_YEAR_15_BASED;
      IslamicChronology islamicChronology0 = IslamicChronology.getInstance(dateTimeZone0, islamicChronology_LeapYearPatternType0);
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.yearOfCentury();
      Partial partial0 = new Partial(dateTimeFieldType1, 32, islamicChronology0);
      Partial partial1 = partial0.with(dateTimeFieldType0, 1);
      DateTimeFormatter dateTimeFormatter0 = partial1.getFormatter();
      assertFalse(dateTimeFormatter0.isOffsetParsed());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.millisOfSecond();
      GregorianChronology gregorianChronology0 = GregorianChronology.getInstance();
      StrictChronology strictChronology0 = StrictChronology.getInstance(gregorianChronology0);
      Partial partial0 = new Partial(dateTimeFieldType0, 0, strictChronology0);
      partial0.getFormatter();
      String string0 = partial0.toString();
      assertEquals("---.000", string0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.millisOfDay();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[1];
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.weekyearOfCentury();
      dateTimeFieldTypeArray0[0] = dateTimeFieldType1;
      int[] intArray0 = new int[1];
      Partial partial0 = new Partial(dateTimeFieldTypeArray0, intArray0);
      Partial partial1 = partial0.with(dateTimeFieldType0, 2);
      String string0 = partial1.toString();
      assertEquals("[weekyearOfCentury=0, millisOfDay=2]", string0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Partial partial0 = new Partial();
      String string0 = partial0.toString("xd<$");
      assertEquals("\uFFFD\uFFFD<$", string0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      YearMonth yearMonth0 = YearMonth.now();
      Partial partial0 = new Partial(yearMonth0);
      Locale locale0 = Locale.KOREA;
      // Undeclared exception!
      try { 
        partial0.toString("", locale0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid pattern specification
         //
         verifyException("org.joda.time.format.DateTimeFormat", e);
      }
  }
}