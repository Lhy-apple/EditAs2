/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 20:14:53 GMT 2023
 */

package org.joda.time.base;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Date;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.joda.time.Chronology;
import org.joda.time.DateTime;
import org.joda.time.Days;
import org.joda.time.DurationFieldType;
import org.joda.time.Hours;
import org.joda.time.Instant;
import org.joda.time.LocalDate;
import org.joda.time.LocalDateTime;
import org.joda.time.LocalTime;
import org.joda.time.Minutes;
import org.joda.time.Months;
import org.joda.time.MutablePeriod;
import org.joda.time.ReadableInstant;
import org.joda.time.ReadablePartial;
import org.joda.time.ReadablePeriod;
import org.joda.time.Weeks;
import org.joda.time.Years;
import org.joda.time.base.BaseSingleFieldPeriod;
import org.joda.time.chrono.IslamicChronology;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BaseSingleFieldPeriod_ESTest extends BaseSingleFieldPeriod_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      Months months0 = Months.monthsBetween((ReadablePartial) localDateTime0, (ReadablePartial) localDateTime0);
      assertEquals(0, months0.getMonths());
      
      Hours hours0 = Hours.standardHoursIn(months0);
      assertEquals(0, hours0.getHours());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Months months0 = Months.EIGHT;
      months0.hashCode();
      assertEquals(8, months0.getMonths());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Months months0 = Months.EIGHT;
      months0.toPeriod();
      assertEquals(8, months0.getMonths());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Years years0 = Years.THREE;
      years0.toMutablePeriod();
      assertEquals(3, years0.getYears());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Weeks weeks0 = Weeks.THREE;
      Weeks weeks1 = weeks0.negated();
      assertEquals(3, weeks0.getWeeks());
      assertEquals((-3), weeks1.getWeeks());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      // Undeclared exception!
      try { 
        Hours.hoursBetween((ReadableInstant) null, (ReadableInstant) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadableInstant objects must not be null
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      IslamicChronology islamicChronology0 = IslamicChronology.getInstanceUTC();
      DateTime dateTime0 = DateTime.now((Chronology) islamicChronology0);
      // Undeclared exception!
      try { 
        Years.yearsBetween((ReadableInstant) dateTime0, (ReadableInstant) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadableInstant objects must not be null
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Instant instant0 = new Instant();
      Days days0 = Days.daysBetween((ReadableInstant) instant0, (ReadableInstant) instant0);
      assertEquals(0, days0.getDays());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      // Undeclared exception!
      try { 
        Months.monthsBetween((ReadablePartial) null, (ReadablePartial) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadablePartial objects must not be null
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      // Undeclared exception!
      try { 
        Hours.hoursBetween((ReadablePartial) localDateTime0, (ReadablePartial) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadablePartial objects must not be null
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      Date date0 = localDateTime0.toDate();
      LocalDate localDate0 = LocalDate.fromDateFields(date0);
      // Undeclared exception!
      try { 
        Months.monthsBetween((ReadablePartial) localDateTime0, (ReadablePartial) localDate0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadablePartial objects must have the same set of fields
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      LocalDateTime localDateTime0 = LocalDateTime.now();
      Months months0 = Months.monthsBetween((ReadablePartial) localDateTime0, (ReadablePartial) localDateTime0);
      LocalTime localTime0 = new LocalTime();
      // Undeclared exception!
      try { 
        BaseSingleFieldPeriod.between(localDateTime0, localTime0, months0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ReadablePartial objects must have the same set of fields
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Minutes minutes0 = Minutes.standardMinutesIn((ReadablePeriod) null);
      assertEquals(0, minutes0.getMinutes());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Years years0 = Years.MIN_VALUE;
      // Undeclared exception!
      try { 
        Hours.standardHoursIn(years0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Cannot convert period to duration as years is not precise in the period P-2147483648Y
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Days days0 = Days.ONE;
      Days days1 = Days.standardDaysIn(days0);
      assertSame(days1, days0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Months months0 = Months.EIGHT;
      // Undeclared exception!
      try { 
        months0.getFieldType(1857);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // 1857
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Months months0 = Months.EIGHT;
      // Undeclared exception!
      try { 
        months0.getValue(168);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // 168
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Years years0 = Years.THREE;
      DurationFieldType durationFieldType0 = DurationFieldType.centuries();
      int int0 = years0.get(durationFieldType0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Years years0 = Years.THREE;
      DurationFieldType durationFieldType0 = years0.getFieldType();
      int int0 = years0.get(durationFieldType0);
      assertEquals(3, int0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Years years0 = Years.MAX_VALUE;
      boolean boolean0 = years0.isSupported((DurationFieldType) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Years years0 = Years.ONE;
      DurationFieldType durationFieldType0 = years0.getFieldType();
      boolean boolean0 = years0.isSupported(durationFieldType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Years years0 = Years.ONE;
      boolean boolean0 = years0.equals(years0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Years years0 = Years.TWO;
      MutablePeriod mutablePeriod0 = new MutablePeriod(years0);
      boolean boolean0 = years0.equals(mutablePeriod0);
      assertEquals(2, years0.getYears());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      LocalDateTime localDateTime0 = new LocalDateTime();
      Years years0 = Years.TWO;
      boolean boolean0 = years0.equals(localDateTime0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Years years0 = Years.THREE;
      Months months0 = Months.ONE;
      boolean boolean0 = years0.equals(months0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Years years0 = Years.THREE;
      Years years1 = Years.TWO;
      boolean boolean0 = years1.equals(years0);
      assertEquals(2, years1.getYears());
      assertFalse(boolean0);
      assertFalse(years0.equals((Object)years1));
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Years years0 = Years.ONE;
      int int0 = years0.compareTo((BaseSingleFieldPeriod) years0);
      assertEquals(0, int0);
      assertEquals(1, years0.getYears());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Months months0 = Months.TWELVE;
      Days days0 = Days.FIVE;
      // Undeclared exception!
      try { 
        months0.compareTo((BaseSingleFieldPeriod) days0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // class org.joda.time.Months cannot be compared to class org.joda.time.Days
         //
         verifyException("org.joda.time.base.BaseSingleFieldPeriod", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Months months0 = Months.MAX_VALUE;
      Months months1 = Months.MIN_VALUE;
      int int0 = months0.compareTo((BaseSingleFieldPeriod) months1);
      assertEquals(Integer.MAX_VALUE, months0.getMonths());
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Years years0 = Years.THREE;
      Years years1 = Years.TWO;
      int int0 = years1.compareTo((BaseSingleFieldPeriod) years0);
      assertEquals((-1), int0);
  }
}
