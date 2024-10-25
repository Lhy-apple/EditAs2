/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:14:11 GMT 2023
 */

package org.joda.time;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Locale;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.joda.time.Chronology;
import org.joda.time.DateTime;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.joda.time.Days;
import org.joda.time.DurationFieldType;
import org.joda.time.LocalTime;
import org.joda.time.Partial;
import org.joda.time.Period;
import org.joda.time.ReadableInstant;
import org.joda.time.ReadablePartial;
import org.joda.time.ReadablePeriod;
import org.joda.time.Seconds;
import org.joda.time.chrono.CopticChronology;
import org.joda.time.chrono.EthiopicChronology;
import org.joda.time.chrono.GJChronology;
import org.joda.time.chrono.ISOChronology;
import org.joda.time.chrono.IslamicChronology;
import org.joda.time.chrono.JulianChronology;
import org.joda.time.chrono.LenientChronology;
import org.joda.time.format.DateTimeFormatter;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Partial_ESTest extends Partial_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[3];
      int[] intArray0 = new int[3];
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
      LocalTime localTime0 = LocalTime.now();
      Partial partial0 = new Partial(localTime0);
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.millisOfDay();
      Partial partial1 = partial0.with(dateTimeFieldType0, 17887500);
      assertEquals(5, partial1.size());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Partial partial0 = new Partial();
      assertEquals(0, partial0.size());
      
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.year();
      Partial partial1 = partial0.with(dateTimeFieldType0, 27);
      String string0 = partial1.toString();
      assertEquals("0027", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      LocalTime localTime0 = LocalTime.now();
      Partial partial0 = new Partial(localTime0);
      Days days0 = Days.daysBetween((ReadablePartial) partial0, (ReadablePartial) localTime0);
      Partial partial1 = partial0.plus(days0);
      assertTrue(partial1.equals((Object)partial0));
      assertNotSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfDay();
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstanceUTC();
      Partial partial0 = new Partial(dateTimeFieldType0, 500, ethiopicChronology0);
      Seconds seconds0 = Seconds.MAX_VALUE;
      Days days0 = seconds0.toStandardDays();
      Partial partial1 = partial0.minus(days0);
      assertNotSame(partial1, partial0);
      assertTrue(partial1.equals((Object)partial0));
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.yearOfEra();
      Partial partial0 = new Partial(dateTimeFieldType0, 3616);
      DateTimeFieldType[] dateTimeFieldTypeArray0 = partial0.getFieldTypes();
      assertEquals(1, dateTimeFieldTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JulianChronology julianChronology0 = JulianChronology.getInstance();
      Partial partial0 = new Partial(julianChronology0);
      Partial.Property partial_Property0 = new Partial.Property(partial0, 2615625);
      // Undeclared exception!
      try { 
        partial_Property0.setCopy("v*EtA^PzA");
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.halfdayOfDay();
      Partial partial0 = new Partial(dateTimeFieldType0, 0);
      Partial.Property partial_Property0 = partial0.property(dateTimeFieldType0);
      int int0 = partial_Property0.get();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.halfdayOfDay();
      Partial partial0 = new Partial(dateTimeFieldType0, 0);
      Partial.Property partial_Property0 = partial0.property(dateTimeFieldType0);
      Partial partial1 = partial_Property0.addWrapFieldToCopy(0);
      assertTrue(partial1.equals((Object)partial0));
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfDay();
      Partial partial0 = new Partial(dateTimeFieldType0, 0);
      Partial.Property partial_Property0 = partial0.property(dateTimeFieldType0);
      Partial partial1 = partial_Property0.withMaximumValue();
      assertNotSame(partial0, partial1);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      Partial partial0 = new Partial(gJChronology0);
      Partial.Property partial_Property0 = new Partial.Property(partial0, 44);
      Partial partial1 = partial_Property0.getPartial();
      assertSame(partial0, partial1);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.hourOfDay();
      ISOChronology iSOChronology0 = ISOChronology.getInstance();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[9];
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      int[] intArray0 = new int[9];
      Partial partial0 = new Partial(iSOChronology0, dateTimeFieldTypeArray0, intArray0);
      Partial.Property partial_Property0 = partial0.property(dateTimeFieldType0);
      // Undeclared exception!
      try { 
        partial_Property0.addToCopy((-347));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Maximum value exceeded for add
         //
         verifyException("org.joda.time.field.BaseDateTimeField", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Partial partial0 = new Partial();
      Partial.Property partial_Property0 = new Partial.Property(partial0, (-463));
      // Undeclared exception!
      try { 
        partial_Property0.withMinimumValue();
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Partial partial0 = null;
      try {
        partial0 = new Partial((DateTimeFieldType) null, 32);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The field type must not be null
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Partial partial0 = null;
      try {
        partial0 = new Partial((DateTimeFieldType[]) null, (int[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must not be null
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[3];
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
  public void test16()  throws Throwable  {
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[3];
      int[] intArray0 = new int[0];
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
  public void test17()  throws Throwable  {
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[0];
      int[] intArray0 = new int[0];
      GJChronology gJChronology0 = GJChronology.getInstance();
      Partial partial0 = new Partial(dateTimeFieldTypeArray0, intArray0, gJChronology0);
      String string0 = partial0.toString((String) null, (Locale) null);
      assertEquals("[]", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.dayOfMonth();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[1];
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      int[] intArray0 = new int[1];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 0 for dayOfMonth must not be smaller than 1
         //
         verifyException("org.joda.time.chrono.BaseChronology", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.millisOfDay();
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.dayOfMonth();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[3];
      dateTimeFieldTypeArray0[0] = dateTimeFieldType1;
      dateTimeFieldTypeArray0[1] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[2] = dateTimeFieldType1;
      int[] intArray0 = new int[3];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must be in order largest-smallest: millisOfDay < dayOfMonth
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.millisOfDay();
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.dayOfMonth();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[3];
      dateTimeFieldTypeArray0[0] = dateTimeFieldType1;
      dateTimeFieldTypeArray0[1] = dateTimeFieldType0;
      DateTimeFieldType dateTimeFieldType2 = DateTimeFieldType.era();
      dateTimeFieldTypeArray0[2] = dateTimeFieldType2;
      int[] intArray0 = new int[3];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must be in order largest-smallest: millisOfDay < era
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[3];
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.era();
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[1] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[2] = dateTimeFieldTypeArray0[0];
      int[] intArray0 = new int[3];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must not contain duplicate: era
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[3];
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.era();
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.dayOfMonth();
      dateTimeFieldTypeArray0[1] = dateTimeFieldType1;
      dateTimeFieldTypeArray0[2] = dateTimeFieldType1;
      int[] intArray0 = new int[3];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must not contain duplicate: dayOfMonth
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      LocalTime localTime0 = LocalTime.now();
      Partial partial0 = new Partial(localTime0);
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.millisOfDay();
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.year();
      CopticChronology copticChronology0 = CopticChronology.getInstanceUTC();
      LenientChronology lenientChronology0 = LenientChronology.getInstance(copticChronology0);
      int[] intArray0 = lenientChronology0.get((ReadablePartial) partial0, (long) 1);
      DateTimeFieldType dateTimeFieldType2 = DateTimeFieldType.yearOfEra();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[4];
      dateTimeFieldTypeArray0[0] = dateTimeFieldType2;
      dateTimeFieldTypeArray0[1] = dateTimeFieldType1;
      dateTimeFieldTypeArray0[2] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[3] = dateTimeFieldType0;
      Partial partial1 = null;
      try {
        partial1 = new Partial(dateTimeFieldTypeArray0, intArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must be in order largest-smallest: yearOfEra < year
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[3];
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.clockhourOfHalfday();
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.hourOfDay();
      dateTimeFieldTypeArray0[1] = dateTimeFieldType1;
      dateTimeFieldTypeArray0[2] = dateTimeFieldType1;
      int[] intArray0 = new int[3];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must be in order largest-smallest: clockhourOfHalfday < hourOfDay
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[3];
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.dayOfMonth();
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.dayOfWeek();
      dateTimeFieldTypeArray0[1] = dateTimeFieldType1;
      dateTimeFieldTypeArray0[2] = dateTimeFieldType1;
      int[] intArray0 = new int[3];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must not contain duplicate: dayOfWeek
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
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
  public void test27()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfDay();
      Partial partial0 = new Partial(dateTimeFieldType0, 22);
      CopticChronology copticChronology0 = CopticChronology.getInstanceUTC();
      Partial partial1 = partial0.withChronologyRetainFields(copticChronology0);
      assertNotSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      ISOChronology iSOChronology0 = ISOChronology.getInstance((DateTimeZone) null);
      Partial partial0 = new Partial(iSOChronology0);
      Partial partial1 = partial0.withChronologyRetainFields(iSOChronology0);
      assertSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Partial partial0 = new Partial();
      // Undeclared exception!
      try { 
        partial0.with((DateTimeFieldType) null, 1135);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The field type must not be null
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.minuteOfHour();
      JulianChronology julianChronology0 = JulianChronology.getInstance();
      Partial partial0 = new Partial(dateTimeFieldType0, 18, julianChronology0);
      // Undeclared exception!
      try { 
        partial0.with(dateTimeFieldType0, 89);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 89 for minuteOfHour must be in the range [0,59]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Partial partial0 = new Partial();
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.era();
      // Undeclared exception!
      try { 
        partial0.with(dateTimeFieldType0, (-347));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -347 for era must not be smaller than 0
         //
         verifyException("org.joda.time.chrono.BaseChronology", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.millisOfDay();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[8];
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[1] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[2] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[3] = dateTimeFieldType0;
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.era();
      dateTimeFieldTypeArray0[4] = dateTimeFieldType1;
      int[] intArray0 = new int[1];
      Partial partial0 = new Partial((Chronology) null, dateTimeFieldTypeArray0, intArray0);
      DateTimeFieldType dateTimeFieldType2 = DateTimeFieldType.millisOfSecond();
      // Undeclared exception!
      try { 
        partial0.with(dateTimeFieldType2, 4567);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Partial partial0 = new Partial();
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.year();
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.dayOfMonth();
      Partial partial1 = partial0.with(dateTimeFieldType1, 27);
      Partial partial2 = partial1.with(dateTimeFieldType0, 27);
      assertNotSame(partial0, partial2);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      IslamicChronology islamicChronology0 = IslamicChronology.getInstanceUTC();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[2];
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.halfdayOfDay();
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      int[] intArray0 = new int[7];
      intArray0[0] = 1;
      Partial partial0 = new Partial(islamicChronology0, dateTimeFieldTypeArray0, intArray0);
      Partial partial1 = partial0.with(dateTimeFieldType0, 1);
      assertSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      LocalTime localTime0 = LocalTime.now();
      Partial partial0 = new Partial(localTime0);
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.millisOfDay();
      Partial partial1 = partial0.without(dateTimeFieldType0);
      assertEquals(4, partial1.size());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfDay();
      Partial partial0 = new Partial(dateTimeFieldType0, 22);
      Partial partial1 = partial0.without(dateTimeFieldType0);
      assertEquals(0, partial1.size());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.year();
      Partial partial0 = new Partial(dateTimeFieldType0, (-255));
      Partial partial1 = partial0.withField(dateTimeFieldType0, 22);
      assertNotSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfDay();
      Partial partial0 = new Partial(dateTimeFieldType0, 22);
      Partial partial1 = partial0.withField(dateTimeFieldType0, 22);
      assertSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfDay();
      DurationFieldType durationFieldType0 = dateTimeFieldType0.getDurationType();
      Partial partial0 = new Partial(dateTimeFieldType0, 0);
      Partial partial1 = partial0.withFieldAdded(durationFieldType0, 20);
      assertNotSame(partial1, partial0);
      assertFalse(partial1.equals((Object)partial0));
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfDay();
      DurationFieldType durationFieldType0 = dateTimeFieldType0.getDurationType();
      Partial partial0 = new Partial(dateTimeFieldType0, 0);
      Partial partial1 = partial0.withFieldAdded(durationFieldType0, 0);
      assertSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfDay();
      DurationFieldType durationFieldType0 = dateTimeFieldType0.getDurationType();
      Partial partial0 = new Partial(dateTimeFieldType0, 7);
      Partial partial1 = partial0.withFieldAddWrapped(durationFieldType0, 7);
      assertNotSame(partial1, partial0);
      assertFalse(partial1.equals((Object)partial0));
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfDay();
      DurationFieldType durationFieldType0 = dateTimeFieldType0.getDurationType();
      Partial partial0 = new Partial(dateTimeFieldType0, 7);
      Partial partial1 = partial0.withFieldAddWrapped(durationFieldType0, 0);
      assertSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Partial partial0 = new Partial();
      Partial partial1 = partial0.withPeriodAdded((ReadablePeriod) null, (-1589));
      assertEquals(0, partial1.size());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Partial partial0 = new Partial();
      Period period0 = new Period(0L, (-1L));
      Partial partial1 = partial0.withPeriodAdded(period0, 0);
      assertSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfDay();
      Partial partial0 = new Partial(dateTimeFieldType0, 22);
      Period period0 = Period.hours(22);
      Partial partial1 = partial0.withPeriodAdded(period0, 20);
      assertNotSame(partial1, partial0);
      assertTrue(partial1.equals((Object)partial0));
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Partial partial0 = new Partial();
      boolean boolean0 = partial0.isMatch((ReadableInstant) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      IslamicChronology islamicChronology0 = IslamicChronology.getInstanceUTC();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[2];
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.halfdayOfDay();
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      int[] intArray0 = new int[7];
      intArray0[0] = 1;
      Partial partial0 = new Partial(islamicChronology0, dateTimeFieldTypeArray0, intArray0);
      DateTime dateTime0 = new DateTime((Chronology) islamicChronology0);
      // Undeclared exception!
      try { 
        partial0.isMatch((ReadableInstant) dateTime0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      IslamicChronology islamicChronology0 = IslamicChronology.getInstanceUTC();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[2];
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.halfdayOfDay();
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      int[] intArray0 = new int[7];
      Partial partial0 = new Partial(islamicChronology0, dateTimeFieldTypeArray0, intArray0);
      DateTime dateTime0 = new DateTime((Chronology) islamicChronology0);
      boolean boolean0 = partial0.isMatch((ReadableInstant) dateTime0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Partial partial0 = new Partial();
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
  public void test50()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfDay();
      Partial partial0 = new Partial(dateTimeFieldType0, 22);
      boolean boolean0 = partial0.isMatch((ReadablePartial) partial0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      IslamicChronology islamicChronology0 = IslamicChronology.getInstanceUTC();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[2];
      int[] intArray0 = new int[7];
      Partial partial0 = new Partial(islamicChronology0, dateTimeFieldTypeArray0, intArray0);
      partial0.getFormatter();
      DateTimeFormatter dateTimeFormatter0 = partial0.getFormatter();
      assertNull(dateTimeFormatter0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      IslamicChronology islamicChronology0 = IslamicChronology.getInstanceUTC();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[2];
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.hourOfDay();
      dateTimeFieldTypeArray0[1] = dateTimeFieldType0;
      int[] intArray0 = new int[7];
      Partial partial0 = new Partial(islamicChronology0, dateTimeFieldTypeArray0, intArray0);
      DateTimeFormatter dateTimeFormatter0 = partial0.getFormatter();
      assertEquals(2000, dateTimeFormatter0.getDefaultYear());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      IslamicChronology islamicChronology0 = IslamicChronology.getInstanceUTC();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[2];
      int[] intArray0 = new int[7];
      Partial partial0 = new Partial(islamicChronology0, dateTimeFieldTypeArray0, intArray0);
      partial0.getFormatter();
      // Undeclared exception!
      try { 
        partial0.toString();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      IslamicChronology islamicChronology0 = IslamicChronology.getInstanceUTC();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[2];
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.halfdayOfDay();
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      int[] intArray0 = new int[7];
      Partial partial0 = new Partial(islamicChronology0, dateTimeFieldTypeArray0, intArray0);
      // Undeclared exception!
      try { 
        partial0.toString();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Partial partial0 = new Partial();
      // Undeclared exception!
      try { 
        partial0.toString("[secondOfDay=7]");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Illegal pattern component: c
         //
         verifyException("org.joda.time.format.DateTimeFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Partial partial0 = new Partial();
      String string0 = partial0.toString((String) null);
      assertEquals("[]", string0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      LocalTime localTime0 = LocalTime.now();
      Partial partial0 = new Partial(localTime0);
      // Undeclared exception!
      try { 
        partial0.toString("NoHours", (Locale) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Illegal pattern component: N
         //
         verifyException("org.joda.time.format.DateTimeFormat", e);
      }
  }
}
