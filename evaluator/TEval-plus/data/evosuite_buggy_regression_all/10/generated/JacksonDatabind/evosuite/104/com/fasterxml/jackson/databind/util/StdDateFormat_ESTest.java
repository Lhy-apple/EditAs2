/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:43:16 GMT 2023
 */

package com.fasterxml.jackson.databind.util;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.util.StdDateFormat;
import java.text.ParseException;
import java.text.ParsePosition;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.util.Calendar;
import java.util.Date;
import java.util.Locale;
import java.util.SimpleTimeZone;
import java.util.TimeZone;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.text.MockSimpleDateFormat;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.junit.runner.RunWith;
import sun.util.calendar.ZoneInfo;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StdDateFormat_ESTest extends StdDateFormat_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      StdDateFormat stdDateFormat1 = stdDateFormat0.clone();
      assertFalse(stdDateFormat1.isColonIncludedInTimeZone());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      try { 
        stdDateFormat0.instance.parseObject("yyyy-MM-dd'T'HH:mm:ss.SSSZ");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Format.parseObject(String) failed
         //
         verifyException("java.text.Format", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      Date date0 = stdDateFormat0.parse("0000-00-00");
      assertEquals("Fri Feb 14 20:21:21 GMT 2014", date0.toString());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      String string0 = stdDateFormat0.instance.toString();
      assertEquals("DateFormat com.fasterxml.jackson.databind.util.StdDateFormat: (timezone: null, locale: en_US, lenient: null)", string0);
      assertFalse(stdDateFormat0.isColonIncludedInTimeZone());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      TimeZone timeZone0 = stdDateFormat0.getTimeZone();
      assertNull(timeZone0);
      assertFalse(stdDateFormat0.isColonIncludedInTimeZone());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      Date date0 = stdDateFormat0.parse("0");
      assertEquals("Thu Jan 01 00:00:00 GMT 1970", date0.toString());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ZoneInfo zoneInfo0 = (ZoneInfo)StdDateFormat.DEFAULT_TIMEZONE;
      Locale locale0 = Locale.GERMANY;
      MockSimpleDateFormat mockSimpleDateFormat0 = (MockSimpleDateFormat)StdDateFormat.getRFC1123Format(zoneInfo0, locale0);
      assertEquals("EEE, tt MMM uuuu HH:mm:ss zzz", mockSimpleDateFormat0.toLocalizedPattern());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TimeZone timeZone0 = StdDateFormat.getDefaultTimeZone();
      // Undeclared exception!
      try { 
        StdDateFormat.getISO8601Format(timeZone0, (Locale) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ZoneInfo zoneInfo0 = (ZoneInfo)StdDateFormat.DEFAULT_TIMEZONE;
      Locale locale0 = new Locale("W:8CfIh", "W:8CfIh", "W:8CfIh");
      Boolean boolean0 = Boolean.TRUE;
      StdDateFormat stdDateFormat0 = new StdDateFormat(zoneInfo0, locale0, boolean0);
      boolean boolean1 = stdDateFormat0.isColonIncludedInTimeZone();
      assertFalse(boolean1);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      TimeZone timeZone0 = StdDateFormat.getDefaultTimeZone();
      stdDateFormat0.instance.setTimeZone(timeZone0);
      assertEquals("UTC", timeZone0.getID());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      stdDateFormat0.hashCode();
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      ZoneOffset zoneOffset0 = ZoneOffset.MIN;
      TimeZone timeZone0 = TimeZone.getTimeZone((ZoneId) zoneOffset0);
      StdDateFormat stdDateFormat1 = stdDateFormat0.withTimeZone(timeZone0);
      stdDateFormat1.setTimeZone(timeZone0);
      assertFalse(stdDateFormat1.isColonIncludedInTimeZone());
      assertNotSame(stdDateFormat1, stdDateFormat0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      StdDateFormat stdDateFormat1 = stdDateFormat0.instance.withTimeZone((TimeZone) null);
      assertFalse(stdDateFormat1.isColonIncludedInTimeZone());
      assertNotSame(stdDateFormat1, stdDateFormat0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ZoneInfo zoneInfo0 = (ZoneInfo)StdDateFormat.DEFAULT_TIMEZONE;
      Locale locale0 = Locale.KOREAN;
      StdDateFormat stdDateFormat0 = new StdDateFormat(zoneInfo0, locale0);
      StdDateFormat stdDateFormat1 = stdDateFormat0.withTimeZone(zoneInfo0);
      assertFalse(stdDateFormat1.isColonIncludedInTimeZone());
      assertSame(stdDateFormat1, stdDateFormat0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.CANADA;
      Boolean boolean0 = new Boolean("");
      StdDateFormat stdDateFormat0 = new StdDateFormat(timeZone0, locale0, boolean0);
      TimeZone timeZone1 = TimeZone.getTimeZone("yyyy-MM-dd'T'HH:mm:ss.SSSZ");
      StdDateFormat stdDateFormat1 = stdDateFormat0.withTimeZone(timeZone1);
      assertSame(stdDateFormat1, stdDateFormat0);
      assertFalse(stdDateFormat1.isLenient());
      assertFalse(stdDateFormat1.isColonIncludedInTimeZone());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      Locale locale0 = Locale.KOREAN;
      StdDateFormat stdDateFormat1 = stdDateFormat0.withLocale(locale0);
      try { 
        stdDateFormat1.parse("yyyy-MM-dd'T'HH:mm:ss.SSSZ");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Cannot parse date \"yyyy-MM-dd'T'HH:mm:ss.SSSZ\": not compatible with any of standard forms (\"yyyy-MM-dd'T'HH:mm:ss.SSSZ\", \"yyyy-MM-dd'T'HH:mm:ss.SSS\", \"EEE, dd MMM yyyy HH:mm:ss zzz\", \"yyyy-MM-dd\")
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMAN;
      StdDateFormat stdDateFormat0 = new StdDateFormat(timeZone0, locale0);
      StdDateFormat stdDateFormat1 = stdDateFormat0.withLocale(locale0);
      assertFalse(stdDateFormat1.isColonIncludedInTimeZone());
      assertSame(stdDateFormat1, stdDateFormat0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      Boolean boolean0 = new Boolean("yyyy-MM-dd'T'HH:mm:ss.SSSZ");
      StdDateFormat stdDateFormat1 = stdDateFormat0.instance.withLenient(boolean0);
      assertFalse(stdDateFormat1.isLenient());
      assertFalse(stdDateFormat1.isColonIncludedInTimeZone());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      stdDateFormat0.setLenient(false);
      Boolean boolean0 = new Boolean("yyyy-MM-dd'T'HH:mm:ss.SSSZ");
      StdDateFormat stdDateFormat1 = stdDateFormat0.instance.withLenient(boolean0);
      assertFalse(stdDateFormat0.isLenient());
      assertSame(stdDateFormat0, stdDateFormat1);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Locale locale0 = Locale.KOREAN;
      Boolean boolean0 = Boolean.TRUE;
      StdDateFormat stdDateFormat0 = new StdDateFormat((TimeZone) null, locale0, boolean0, true);
      StdDateFormat stdDateFormat1 = stdDateFormat0.withColonInTimeZone(false);
      assertFalse(stdDateFormat1.isColonIncludedInTimeZone());
      assertNotSame(stdDateFormat1, stdDateFormat0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      StdDateFormat stdDateFormat1 = stdDateFormat0.withColonInTimeZone(false);
      assertSame(stdDateFormat1, stdDateFormat0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ZoneInfo zoneInfo0 = (ZoneInfo)StdDateFormat.DEFAULT_TIMEZONE;
      Locale locale0 = Locale.ENGLISH;
      Boolean boolean0 = Boolean.TRUE;
      StdDateFormat stdDateFormat0 = new StdDateFormat(zoneInfo0, locale0, boolean0);
      stdDateFormat0.setLenient(true);
      assertFalse(stdDateFormat0.isColonIncludedInTimeZone());
      assertTrue(stdDateFormat0.isLenient());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      // Undeclared exception!
      try { 
        stdDateFormat0.parse("0000-00-00T00:00");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No match available
         //
         verifyException("java.util.regex.Matcher", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      SimpleTimeZone simpleTimeZone0 = new SimpleTimeZone(870, "");
      Locale locale0 = Locale.PRC;
      Boolean boolean0 = Boolean.valueOf("!p&");
      StdDateFormat stdDateFormat0 = new StdDateFormat(simpleTimeZone0, locale0, boolean0, true);
      ParsePosition parsePosition0 = new ParsePosition(4);
      stdDateFormat0._parseDate("!p&", parsePosition0);
      assertEquals("java.text.ParsePosition[index=4,errorIndex=4]", parsePosition0.toString());
      assertEquals(4, parsePosition0.getErrorIndex());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      try { 
        stdDateFormat0.parse("k");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Cannot parse date \"k\": not compatible with any of standard forms (\"yyyy-MM-dd'T'HH:mm:ss.SSSZ\", \"yyyy-MM-dd'T'HH:mm:ss.SSS\", \"EEE, dd MMM yyyy HH:mm:ss zzz\", \"yyyy-MM-dd\")
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      try { 
        stdDateFormat0.parse("-");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Timestamp value - out of 64-bit value range
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      MockDate mockDate0 = new MockDate((-2130), 870, (-2130), 4, 870);
      SimpleTimeZone simpleTimeZone0 = new SimpleTimeZone(870, "");
      Locale locale0 = Locale.PRC;
      Boolean boolean0 = Boolean.valueOf("!p&");
      StdDateFormat stdDateFormat0 = new StdDateFormat(simpleTimeZone0, locale0, boolean0, true);
      String string0 = stdDateFormat0.format((Date) mockDate0);
      assertEquals("0165-08-30T18:30:00.870+00:00", string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      MockDate mockDate0 = new MockDate((-2130), (-2130), (-2130), (-2130), (-2130));
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      ZoneOffset zoneOffset0 = ZoneOffset.MIN;
      TimeZone timeZone0 = TimeZone.getTimeZone((ZoneId) zoneOffset0);
      StdDateFormat stdDateFormat1 = stdDateFormat0.withTimeZone(timeZone0);
      String string0 = stdDateFormat1.format((Date) mockDate0);
      assertEquals("0415-06-01T00:30:00.000-1800", string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      MockDate mockDate0 = new MockDate((-112), (-112), 2024, (-112), 60000);
      Locale locale0 = Locale.KOREAN;
      Boolean boolean0 = Boolean.TRUE;
      StdDateFormat stdDateFormat0 = new StdDateFormat((TimeZone) null, locale0, boolean0, true);
      String string0 = stdDateFormat0.format((Date) mockDate0);
      assertEquals("1784-04-22T00:00:00.000+00:00", string0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      MockDate mockDate0 = new MockDate((-1828), (-1828), (-1828));
      String string0 = stdDateFormat0.instance.format((Date) mockDate0);
      assertEquals("0087-08-29T00:00:00.000+0000", string0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      String string0 = stdDateFormat0.instance.toPattern();
      assertFalse(stdDateFormat0.isColonIncludedInTimeZone());
      assertEquals("[one of: 'yyyy-MM-dd'T'HH:mm:ss.SSSZ', 'EEE, dd MMM yyyy HH:mm:ss zzz' (lenient)]", string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      Calendar calendar0 = stdDateFormat0._getCalendar((TimeZone) null);
      boolean boolean0 = stdDateFormat0.equals(calendar0);
      assertEquals("org.evosuite.runtime.mock.java.util.MockGregorianCalendar[time=-64860566400000,areFieldsSet=false,areAllFieldsSet=false,lenient=true,zone=null,firstDayOfWeek=1,minimalDaysInFirstWeek=1,ERA=0,YEAR=87,MONTH=7,WEEK_OF_YEAR=35,WEEK_OF_MONTH=5,DAY_OF_MONTH=29,DAY_OF_YEAR=241,DAY_OF_WEEK=5,DAY_OF_WEEK_IN_MONTH=5,AM_PM=0,HOUR=0,HOUR_OF_DAY=0,MINUTE=0,SECOND=0,MILLISECOND=0,ZONE_OFFSET=0,DST_OFFSET=0]", calendar0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      boolean boolean0 = stdDateFormat0.equals(stdDateFormat0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      try { 
        stdDateFormat0.parse("5s1~]|cu r@w2");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Cannot parse date \"5s1~]|cu r@w2\": not compatible with any of standard forms (\"yyyy-MM-dd'T'HH:mm:ss.SSSZ\", \"yyyy-MM-dd'T'HH:mm:ss.SSS\", \"EEE, dd MMM yyyy HH:mm:ss zzz\", \"yyyy-MM-dd\")
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      try { 
        stdDateFormat0.parse("60000-00-00T00:00");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Cannot parse date \"60000-00-00T00:00\": not compatible with any of standard forms (\"yyyy-MM-dd'T'HH:mm:ss.SSSZ\", \"yyyy-MM-dd'T'HH:mm:ss.SSS\", \"EEE, dd MMM yyyy HH:mm:ss zzz\", \"yyyy-MM-dd\")
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      try { 
        stdDateFormat0.parse("0000--00T00:00");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Cannot parse date \"0000--00T00:00\": not compatible with any of standard forms (\"yyyy-MM-dd'T'HH:mm:ss.SSSZ\", \"yyyy-MM-dd'T'HH:mm:ss.SSS\", \"EEE, dd MMM yyyy HH:mm:ss zzz\", \"yyyy-MM-dd\")
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      ParsePosition parsePosition0 = new ParsePosition(117);
      try { 
        stdDateFormat0._parseAsISO8601("yyyy-MM-dd'T'HH:mm:ss.SSSZ", parsePosition0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Cannot parse date \"yyyy-MM-dd'T'HH:mm:ss.SSSZ\": while it seems to fit format 'yyyy-MM-dd'T'HH:mm:ss.SSSZ', parsing fails (leniency? null)
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      try { 
        stdDateFormat0.parse("0;00-00-00");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Cannot parse date \"0;00-00-00\": while it seems to fit format 'yyyy-MM-dd', parsing fails (leniency? null)
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      ZoneOffset zoneOffset0 = ZoneOffset.MIN;
      TimeZone timeZone0 = TimeZone.getTimeZone((ZoneId) zoneOffset0);
      StdDateFormat stdDateFormat1 = stdDateFormat0.withTimeZone(timeZone0);
      try { 
        stdDateFormat1.parse("Z#v<}IYZQ");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Cannot parse date \"Z#v<}IYZQ\": not compatible with any of standard forms (\"yyyy-MM-dd'T'HH:mm:ss.SSSZ\", \"yyyy-MM-dd'T'HH:mm:ss.SSS\", \"EEE, dd MMM yyyy HH:mm:ss zzz\", \"yyyy-MM-dd\")
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      StdDateFormat stdDateFormat1 = stdDateFormat0.withLenient((Boolean) null);
      assertSame(stdDateFormat1, stdDateFormat0);
  }
}
