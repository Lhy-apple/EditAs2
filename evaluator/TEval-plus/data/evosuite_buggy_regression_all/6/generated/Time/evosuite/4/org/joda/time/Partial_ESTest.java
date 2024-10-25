/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 01:00:41 GMT 2023
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
import org.joda.time.DurationFieldType;
import org.joda.time.LocalDate;
import org.joda.time.LocalDateTime;
import org.joda.time.LocalTime;
import org.joda.time.Minutes;
import org.joda.time.Partial;
import org.joda.time.Period;
import org.joda.time.ReadableInstant;
import org.joda.time.ReadablePartial;
import org.joda.time.ReadablePeriod;
import org.joda.time.Seconds;
import org.joda.time.chrono.BuddhistChronology;
import org.joda.time.chrono.CopticChronology;
import org.joda.time.chrono.EthiopicChronology;
import org.joda.time.chrono.GJChronology;
import org.joda.time.chrono.IslamicChronology;
import org.joda.time.format.DateTimeFormatter;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Partial_ESTest extends Partial_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      LocalDate localDate0 = new LocalDate();
      LocalTime localTime0 = new LocalTime();
      LocalDateTime localDateTime0 = localDate0.toLocalDateTime(localTime0);
      Partial partial0 = new Partial(localDateTime0);
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.secondOfMinute();
      // Undeclared exception!
      try { 
        partial0.property(dateTimeFieldType0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Field 'secondOfMinute' is not supported
         //
         verifyException("org.joda.time.base.AbstractPartial", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[0];
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
  public void test02()  throws Throwable  {
      LocalDate localDate0 = LocalDate.now();
      LocalTime localTime0 = new LocalTime();
      LocalDateTime localDateTime0 = localDate0.toLocalDateTime(localTime0);
      Partial partial0 = new Partial(localDateTime0);
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.millisOfSecond();
      Partial partial1 = partial0.with(dateTimeFieldType0, 32);
      Partial partial2 = partial1.with(dateTimeFieldType0, 32);
      assertSame(partial2, partial1);
      assertEquals(5, partial2.size());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.minuteOfDay();
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      Partial partial0 = new Partial(dateTimeFieldType0, 0, gJChronology0);
      DurationFieldType durationFieldType0 = dateTimeFieldType0.getDurationType();
      Partial partial1 = partial0.withFieldAddWrapped(durationFieldType0, 1);
      assertNotSame(partial1, partial0);
      assertFalse(partial1.equals((Object)partial0));
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Partial partial0 = new Partial((Chronology) null);
      assertEquals(0, partial0.size());
      
      Seconds seconds0 = Seconds.secondsBetween((ReadablePartial) partial0, (ReadablePartial) partial0);
      Partial partial1 = partial0.plus(seconds0);
      assertTrue(partial1.equals((Object)partial0));
      assertNotSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Partial partial0 = new Partial((Chronology) null);
      DateTimeFieldType[] dateTimeFieldTypeArray0 = partial0.getFieldTypes();
      assertEquals(0, dateTimeFieldTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      Partial partial0 = new Partial(localDateTime0);
      Partial.Property partial_Property0 = new Partial.Property(partial0, 9);
      // Undeclared exception!
      try { 
        partial_Property0.setCopy("Types array must not be null");
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 9
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      Partial partial0 = new Partial(localDateTime0);
      Partial.Property partial_Property0 = new Partial.Property(partial0, 1);
      String string0 = partial_Property0.getAsText();
      assertEquals("February", string0);
      assertEquals(2, partial_Property0.get());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Partial partial0 = new Partial((Chronology) null);
      Partial.Property partial_Property0 = new Partial.Property(partial0, 0);
      // Undeclared exception!
      try { 
        partial_Property0.addWrapFieldToCopy(0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 0
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Partial partial0 = new Partial();
      Partial.Property partial_Property0 = new Partial.Property(partial0, 401);
      // Undeclared exception!
      try { 
        partial_Property0.setCopy(401);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 401
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Partial partial0 = new Partial();
      Partial.Property partial_Property0 = new Partial.Property(partial0, (-2205));
      Partial partial1 = partial_Property0.getPartial();
      assertEquals(0, partial1.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Partial.Property partial_Property0 = new Partial.Property((Partial) null, (-4743));
      // Undeclared exception!
      try { 
        partial_Property0.addToCopy(1715);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.joda.time.Partial$Property", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Partial partial0 = new Partial((Chronology) null);
      Partial.Property partial_Property0 = new Partial.Property(partial0, 32);
      // Undeclared exception!
      try { 
        partial_Property0.withMaximumValue();
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 32
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CopticChronology copticChronology0 = CopticChronology.getInstance();
      Partial partial0 = new Partial(copticChronology0);
      Partial.Property partial_Property0 = new Partial.Property(partial0, 1);
      // Undeclared exception!
      try { 
        partial_Property0.withMinimumValue();
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 1
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[2];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0[1], 1);
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
      IslamicChronology islamicChronology0 = IslamicChronology.getInstanceUTC();
      Partial partial0 = null;
      try {
        partial0 = new Partial((DateTimeFieldType[]) null, intArray0, islamicChronology0);
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
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[1];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, (int[]) null, buddhistChronology0);
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
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstanceUTC();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[2];
      int[] intArray0 = new int[2];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0, buddhistChronology0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must not contain null: index 0
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[0];
      int[] intArray0 = new int[0];
      Partial partial0 = new Partial(dateTimeFieldTypeArray0, intArray0);
      DateTimeFormatter dateTimeFormatter0 = partial0.getFormatter();
      assertNull(dateTimeFormatter0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.minuteOfDay();
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstanceUTC();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[2];
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[1] = dateTimeFieldType0;
      int[] intArray0 = new int[2];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0, buddhistChronology0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must not contain duplicate: minuteOfDay
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.era();
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.minuteOfDay();
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[2];
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[1] = dateTimeFieldType1;
      int[] intArray0 = new int[2];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0, buddhistChronology0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 0 for era must not be smaller than 1
         //
         verifyException("org.joda.time.chrono.BaseChronology", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.minuteOfDay();
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.secondOfMinute();
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[2];
      dateTimeFieldTypeArray0[0] = dateTimeFieldType1;
      dateTimeFieldTypeArray0[1] = dateTimeFieldType0;
      int[] intArray0 = new int[2];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0, buddhistChronology0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must be in order largest-smallest: secondOfMinute < minuteOfDay
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.minuteOfDay();
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.hourOfHalfday();
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[2];
      dateTimeFieldTypeArray0[0] = dateTimeFieldType1;
      dateTimeFieldTypeArray0[1] = dateTimeFieldType0;
      int[] intArray0 = new int[2];
      Partial partial0 = new Partial(dateTimeFieldTypeArray0, intArray0, buddhistChronology0);
      assertEquals(2, partial0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.minuteOfDay();
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.era();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[2];
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[1] = dateTimeFieldType1;
      int[] intArray0 = new int[2];
      IslamicChronology islamicChronology0 = IslamicChronology.getInstance();
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0, islamicChronology0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must be in order largest-smallest: minuteOfDay < era
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.year();
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[2];
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[1] = dateTimeFieldType0;
      int[] intArray0 = new int[2];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0, buddhistChronology0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must not contain duplicate: year
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.yearOfCentury();
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[2];
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.year();
      dateTimeFieldTypeArray0[1] = dateTimeFieldType1;
      int[] intArray0 = new int[2];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0, buddhistChronology0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must be in order largest-smallest: yearOfCentury < year
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.hourOfHalfday();
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.clockhourOfDay();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[3];
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      dateTimeFieldTypeArray0[1] = dateTimeFieldType1;
      dateTimeFieldTypeArray0[2] = dateTimeFieldType1;
      int[] intArray0 = new int[3];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0, (Chronology) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Types array must be in order largest-smallest: hourOfHalfday < clockhourOfDay
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.yearOfCentury();
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstanceUTC();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[2];
      dateTimeFieldTypeArray0[0] = dateTimeFieldType0;
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.yearOfEra();
      dateTimeFieldTypeArray0[1] = dateTimeFieldType1;
      int[] intArray0 = new int[2];
      Partial partial0 = null;
      try {
        partial0 = new Partial(dateTimeFieldTypeArray0, intArray0, buddhistChronology0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 0 for yearOfCentury must not be smaller than 1
         //
         verifyException("org.joda.time.chrono.BaseChronology", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
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
  public void test29()  throws Throwable  {
      Partial partial0 = new Partial();
      CopticChronology copticChronology0 = CopticChronology.getInstance();
      Partial partial1 = partial0.withChronologyRetainFields(copticChronology0);
      assertNotSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Partial partial0 = new Partial((Chronology) null);
      Partial partial1 = partial0.withChronologyRetainFields((Chronology) null);
      assertSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.minuteOfDay();
      DateTimeFieldType[] dateTimeFieldTypeArray0 = new DateTimeFieldType[2];
      Partial partial0 = new Partial(dateTimeFieldType0, 1);
      // Undeclared exception!
      try { 
        partial0.with(dateTimeFieldTypeArray0[0], 1372);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The field type must not be null
         //
         verifyException("org.joda.time.Partial", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      LocalDate localDate0 = LocalDate.now();
      LocalTime localTime0 = new LocalTime();
      LocalDateTime localDateTime0 = localDate0.toLocalDateTime(localTime0);
      Partial partial0 = new Partial(localDateTime0);
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.millisOfSecond();
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.era();
      Partial partial1 = partial0.with(dateTimeFieldType1, 1);
      Partial partial2 = partial1.with(dateTimeFieldType0, 1);
      assertEquals(6, partial2.size());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.yearOfCentury();
      Partial partial0 = new Partial(dateTimeFieldType0, 1);
      DateTimeFieldType dateTimeFieldType1 = DateTimeFieldType.centuryOfEra();
      Partial partial1 = partial0.with(dateTimeFieldType1, 1);
      assertEquals(2, partial1.size());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      LocalTime localTime0 = new LocalTime();
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.minuteOfDay();
      Partial partial0 = new Partial(localTime0);
      // Undeclared exception!
      try { 
        partial0.with(dateTimeFieldType0, (-2371));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -2371 for minuteOfDay must not be smaller than 0
         //
         verifyException("org.joda.time.chrono.BaseChronology", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.yearOfEra();
      Partial partial0 = new Partial(dateTimeFieldType0, 8);
      // Undeclared exception!
      try { 
        partial0.with(dateTimeFieldType0, (-418));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -418 for yearOfEra must be in the range [1,292278993]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Partial partial0 = new Partial((Chronology) null);
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.halfdayOfDay();
      Partial partial1 = partial0.without(dateTimeFieldType0);
      assertEquals(0, partial1.size());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Partial partial0 = new Partial((Chronology) null);
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.halfdayOfDay();
      Partial partial1 = partial0.with(dateTimeFieldType0, 0);
      Partial partial2 = partial1.without(dateTimeFieldType0);
      assertEquals(0, partial2.size());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.weekyearOfCentury();
      Partial partial0 = new Partial(dateTimeFieldType0, 1, (Chronology) null);
      Partial partial1 = partial0.withField(dateTimeFieldType0, 8);
      boolean boolean0 = partial1.isMatch((ReadablePartial) partial0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      LocalDate localDate0 = new LocalDate();
      Partial partial0 = new Partial(localDate0);
      int[] intArray0 = new int[8];
      intArray0[2] = 8;
      Partial partial1 = new Partial(partial0, intArray0);
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.dayOfMonth();
      Partial partial2 = partial1.withField(dateTimeFieldType0, 8);
      assertNotSame(partial2, partial0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      LocalDate localDate0 = LocalDate.now();
      LocalTime localTime0 = new LocalTime();
      LocalDateTime localDateTime0 = localDate0.toLocalDateTime(localTime0);
      Partial partial0 = new Partial(localDateTime0);
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.millisOfSecond();
      DurationFieldType durationFieldType0 = dateTimeFieldType0.getDurationType();
      Partial partial1 = partial0.withFieldAdded(durationFieldType0, (-812));
      assertNotSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Partial partial0 = new Partial((Chronology) null);
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.halfdayOfDay();
      Partial partial1 = partial0.with(dateTimeFieldType0, 0);
      DurationFieldType durationFieldType0 = dateTimeFieldType0.getDurationType();
      Partial partial2 = partial1.withFieldAdded(durationFieldType0, 0);
      assertSame(partial2, partial1);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.minuteOfDay();
      GJChronology gJChronology0 = GJChronology.getInstanceUTC();
      Partial partial0 = new Partial(dateTimeFieldType0, 0, gJChronology0);
      DurationFieldType durationFieldType0 = dateTimeFieldType0.getDurationType();
      Partial partial1 = partial0.withFieldAddWrapped(durationFieldType0, 0);
      assertSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      LocalTime localTime0 = new LocalTime();
      Partial partial0 = new Partial(localTime0);
      Partial partial1 = partial0.minus((ReadablePeriod) null);
      assertSame(partial0, partial1);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Partial partial0 = new Partial((Chronology) null);
      Minutes minutes0 = Minutes.ZERO;
      Partial partial1 = partial0.withPeriodAdded(minutes0, 0);
      assertSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      LocalDate localDate0 = new LocalDate();
      LocalTime localTime0 = new LocalTime();
      LocalDateTime localDateTime0 = localDate0.toLocalDateTime(localTime0);
      Partial partial0 = new Partial(localDateTime0);
      Period period0 = Period.weeks((-2458));
      Partial partial1 = partial0.minus(period0);
      assertTrue(partial1.equals((Object)partial0));
      assertNotSame(partial1, partial0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      LocalTime localTime0 = new LocalTime();
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstance();
      DateTime dateTime0 = DateTime.now((Chronology) ethiopicChronology0);
      Partial partial0 = new Partial(localTime0);
      boolean boolean0 = partial0.isMatch((ReadableInstant) dateTime0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Partial partial0 = new Partial((Chronology) null);
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.halfdayOfDay();
      Partial partial1 = partial0.with(dateTimeFieldType0, 0);
      DateTime dateTime0 = DateTime.now();
      boolean boolean0 = partial1.isMatch((ReadableInstant) dateTime0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      Partial partial0 = new Partial(localDateTime0);
      boolean boolean0 = partial0.isMatch((ReadablePartial) localDateTime0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstanceUTC();
      Partial partial0 = new Partial(buddhistChronology0);
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
      LocalDateTime localDateTime0 = new LocalDateTime();
      Partial partial0 = new Partial(localDateTime0);
      partial0.toString();
      DateTimeFormatter dateTimeFormatter0 = partial0.getFormatter();
      assertTrue(dateTimeFormatter0.isParser());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      Partial partial0 = new Partial(localDateTime0);
      partial0.toString();
      Locale locale0 = Locale.GERMAN;
      String string0 = partial0.toString((String) null, locale0);
      assertEquals("[year=2014, monthOfYear=2, dayOfMonth=14, millisOfDay=44481320]", string0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Partial partial0 = new Partial();
      String string0 = partial0.toString();
      assertEquals("[]", string0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      LocalTime localTime0 = new LocalTime();
      Partial partial0 = new Partial(localTime0);
      String string0 = partial0.toString();
      assertEquals("12:21:21.320", string0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      LocalDate localDate0 = new LocalDate();
      LocalTime localTime0 = new LocalTime();
      LocalDateTime localDateTime0 = localDate0.toLocalDateTime(localTime0);
      Partial partial0 = new Partial(localDateTime0);
      // Undeclared exception!
      try { 
        partial0.toString("d\"|GK/j=qt?:k?+F\"1");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Illegal pattern component: j
         //
         verifyException("org.joda.time.format.DateTimeFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      Partial partial0 = new Partial(localDateTime0);
      String string0 = partial0.toString((String) null);
      assertEquals("[year=2014, monthOfYear=2, dayOfMonth=14, millisOfDay=44481320]", string0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Partial partial0 = new Partial();
      Locale locale0 = Locale.CANADA;
      String string0 = partial0.toString("[]", locale0);
      assertEquals("[]", string0);
  }
}
