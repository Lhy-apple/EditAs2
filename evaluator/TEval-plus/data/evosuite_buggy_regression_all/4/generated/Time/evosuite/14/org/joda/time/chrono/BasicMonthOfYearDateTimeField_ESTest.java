/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:20:56 GMT 2023
 */

package org.joda.time.chrono;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.joda.time.Chronology;
import org.joda.time.DateTimeZone;
import org.joda.time.MonthDay;
import org.joda.time.Period;
import org.joda.time.ReadablePartial;
import org.joda.time.ReadablePeriod;
import org.joda.time.YearMonth;
import org.joda.time.chrono.BasicMonthOfYearDateTimeField;
import org.joda.time.chrono.CopticChronology;
import org.joda.time.chrono.EthiopicChronology;
import org.joda.time.chrono.GJMonthOfYearDateTimeField;
import org.joda.time.chrono.GregorianChronology;
import org.joda.time.chrono.IslamicChronology;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BasicMonthOfYearDateTimeField_ESTest extends BasicMonthOfYearDateTimeField_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      IslamicChronology islamicChronology0 = IslamicChronology.getInstanceUTC();
      GJMonthOfYearDateTimeField gJMonthOfYearDateTimeField0 = new GJMonthOfYearDateTimeField(islamicChronology0);
      long long0 = gJMonthOfYearDateTimeField0.remainder(1);
      assertEquals(1814400001L, long0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      GregorianChronology gregorianChronology0 = GregorianChronology.getInstanceUTC();
      BasicMonthOfYearDateTimeField basicMonthOfYearDateTimeField0 = new BasicMonthOfYearDateTimeField(gregorianChronology0, (-3727));
      boolean boolean0 = basicMonthOfYearDateTimeField0.isLenient();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      IslamicChronology islamicChronology0 = IslamicChronology.getInstance();
      BasicMonthOfYearDateTimeField basicMonthOfYearDateTimeField0 = new BasicMonthOfYearDateTimeField(islamicChronology0, 1);
      long long0 = basicMonthOfYearDateTimeField0.addWrapField((long) 1, 1);
      assertEquals(2505600001L, long0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstanceUTC();
      GJMonthOfYearDateTimeField gJMonthOfYearDateTimeField0 = new GJMonthOfYearDateTimeField(ethiopicChronology0);
      YearMonth yearMonth0 = new YearMonth(1, 1);
      int[] intArray0 = new int[3];
      intArray0[0] = 1;
      intArray0[1] = 1;
      int[] intArray1 = gJMonthOfYearDateTimeField0.add((ReadablePartial) yearMonth0, (-20587489), intArray0, 2144069524);
      int[] intArray2 = gJMonthOfYearDateTimeField0.add((ReadablePartial) yearMonth0, 1, intArray1, 1);
      assertArrayEquals(new int[] {164928426, 1}, intArray2);
      assertArrayEquals(new int[] {164928425, 13}, intArray1);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CopticChronology copticChronology0 = CopticChronology.getInstanceUTC();
      Period period0 = Period.ZERO;
      int[] intArray0 = copticChronology0.get((ReadablePeriod) period0, (long) 1, 3266L);
      assertArrayEquals(new int[] {0, 0, 0, 0, 0, 0, 3, 265}, intArray0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstance();
      GJMonthOfYearDateTimeField gJMonthOfYearDateTimeField0 = new GJMonthOfYearDateTimeField(ethiopicChronology0);
      long long0 = gJMonthOfYearDateTimeField0.add(5037L, (-1056L));
      assertEquals((-2563919994963L), long0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstanceUTC();
      BasicMonthOfYearDateTimeField basicMonthOfYearDateTimeField0 = new BasicMonthOfYearDateTimeField(ethiopicChronology0, 1);
      long long0 = basicMonthOfYearDateTimeField0.add((long) 1, (-2147483645));
      assertEquals((-5213033066476799999L), long0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstance();
      GJMonthOfYearDateTimeField gJMonthOfYearDateTimeField0 = new GJMonthOfYearDateTimeField(ethiopicChronology0);
      long long0 = gJMonthOfYearDateTimeField0.add((long) 2543, 2629800000L);
      assertEquals(6383859729177602543L, long0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetHours(1);
      GregorianChronology gregorianChronology0 = GregorianChronology.getInstance(dateTimeZone0);
      GJMonthOfYearDateTimeField gJMonthOfYearDateTimeField0 = new GJMonthOfYearDateTimeField(gregorianChronology0);
      // Undeclared exception!
      try { 
        gJMonthOfYearDateTimeField0.add((-2147L), (-9223372036854775799L));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Magnitude of add amount is too large: -9223372036854775799
         //
         verifyException("org.joda.time.chrono.BasicMonthOfYearDateTimeField", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      IslamicChronology islamicChronology0 = IslamicChronology.getInstanceUTC();
      GJMonthOfYearDateTimeField gJMonthOfYearDateTimeField0 = new GJMonthOfYearDateTimeField(islamicChronology0);
      // Undeclared exception!
      try { 
        gJMonthOfYearDateTimeField0.add((long) 1593, (-9223372036854775808L));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Magnitude of add amount is too large: -9223372036854775808
         //
         verifyException("org.joda.time.chrono.BasicMonthOfYearDateTimeField", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CopticChronology copticChronology0 = CopticChronology.getInstance((DateTimeZone) null);
      GJMonthOfYearDateTimeField gJMonthOfYearDateTimeField0 = new GJMonthOfYearDateTimeField(copticChronology0);
      // Undeclared exception!
      try { 
        gJMonthOfYearDateTimeField0.add(5097600000L, 5097600000L);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Magnitude of add amount is too large: 5097600000
         //
         verifyException("org.joda.time.chrono.BasicMonthOfYearDateTimeField", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstanceUTC();
      GJMonthOfYearDateTimeField gJMonthOfYearDateTimeField0 = new GJMonthOfYearDateTimeField(ethiopicChronology0);
      long long0 = gJMonthOfYearDateTimeField0.add(45964800001L, 2592000000L);
      assertEquals(6292099983110400001L, long0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CopticChronology copticChronology0 = CopticChronology.getInstance();
      GJMonthOfYearDateTimeField gJMonthOfYearDateTimeField0 = new GJMonthOfYearDateTimeField(copticChronology0);
      MonthDay monthDay0 = new MonthDay((Chronology) copticChronology0);
      gJMonthOfYearDateTimeField0.add((ReadablePartial) monthDay0, 0, (int[]) null, 0);
      assertEquals(6, monthDay0.getMonthOfYear());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstance();
      GJMonthOfYearDateTimeField gJMonthOfYearDateTimeField0 = new GJMonthOfYearDateTimeField(ethiopicChronology0);
      long long0 = gJMonthOfYearDateTimeField0.getDifferenceAsLong(259200000L, 305856004309L);
      assertEquals((-125L), long0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstance();
      BasicMonthOfYearDateTimeField basicMonthOfYearDateTimeField0 = new BasicMonthOfYearDateTimeField(ethiopicChronology0, 1);
      long long0 = basicMonthOfYearDateTimeField0.getDifferenceAsLong(604800000L, 1);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      EthiopicChronology ethiopicChronology0 = EthiopicChronology.getInstance();
      GJMonthOfYearDateTimeField gJMonthOfYearDateTimeField0 = new GJMonthOfYearDateTimeField(ethiopicChronology0);
      int int0 = gJMonthOfYearDateTimeField0.getLeapAmount(355L);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      IslamicChronology islamicChronology0 = IslamicChronology.getInstanceUTC();
      GJMonthOfYearDateTimeField gJMonthOfYearDateTimeField0 = new GJMonthOfYearDateTimeField(islamicChronology0);
      int int0 = gJMonthOfYearDateTimeField0.getLeapAmount(10000000011L);
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      IslamicChronology islamicChronology0 = IslamicChronology.getInstanceUTC();
      GJMonthOfYearDateTimeField gJMonthOfYearDateTimeField0 = new GJMonthOfYearDateTimeField(islamicChronology0);
      boolean boolean0 = gJMonthOfYearDateTimeField0.isLeap(31083663600000L);
      assertFalse(boolean0);
  }
}