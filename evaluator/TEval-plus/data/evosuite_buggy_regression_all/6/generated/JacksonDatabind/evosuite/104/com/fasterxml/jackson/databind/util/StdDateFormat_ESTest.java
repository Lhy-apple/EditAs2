/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:30:31 GMT 2023
 */

package com.fasterxml.jackson.databind.util;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.util.StdDateFormat;
import java.text.FieldPosition;
import java.text.ParseException;
import java.text.ParsePosition;
import java.time.ZoneId;
import java.time.ZoneOffset;
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
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.CANADA_FRENCH;
      StdDateFormat stdDateFormat0 = new StdDateFormat(timeZone0, locale0);
      assertFalse(stdDateFormat0.isColonIncludedInTimeZone());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      StdDateFormat stdDateFormat1 = stdDateFormat0.clone();
      assertFalse(stdDateFormat1.isColonIncludedInTimeZone());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
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
  public void test03()  throws Throwable  {
      TimeZone timeZone0 = StdDateFormat.getDefaultTimeZone();
      Locale locale0 = Locale.CHINESE;
      StdDateFormat stdDateFormat0 = new StdDateFormat(timeZone0, locale0, (Boolean) null);
      ZoneOffset zoneOffset0 = ZoneOffset.UTC;
      TimeZone timeZone1 = TimeZone.getTimeZone((ZoneId) zoneOffset0);
      StdDateFormat stdDateFormat1 = stdDateFormat0.withTimeZone(timeZone1);
      assertSame(stdDateFormat1, stdDateFormat0);
      assertFalse(stdDateFormat1.isColonIncludedInTimeZone());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      stdDateFormat0.parse("0000-00-00");
      assertTrue(stdDateFormat0.isLenient());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      String string0 = stdDateFormat0.instance.toString();
      assertFalse(stdDateFormat0.isColonIncludedInTimeZone());
      assertEquals("DateFormat com.fasterxml.jackson.databind.util.StdDateFormat: (timezone: null, locale: en_US, lenient: null)", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      TimeZone timeZone0 = stdDateFormat0.getTimeZone();
      assertFalse(stdDateFormat0.isColonIncludedInTimeZone());
      assertNull(timeZone0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
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
  public void test08()  throws Throwable  {
      Locale locale0 = StdDateFormat.DEFAULT_LOCALE;
      MockSimpleDateFormat mockSimpleDateFormat0 = (MockSimpleDateFormat)StdDateFormat.getRFC1123Format((TimeZone) null, locale0);
      assertEquals("EEE, dd MMM yyyy HH:mm:ss zzz", mockSimpleDateFormat0.toLocalizedPattern());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Locale locale0 = Locale.GERMAN;
      MockSimpleDateFormat mockSimpleDateFormat0 = (MockSimpleDateFormat)StdDateFormat.getISO8601Format((TimeZone) null, locale0);
      assertEquals("uuuu-MM-tt'T'HH:mm:ss.SSSZ", mockSimpleDateFormat0.toLocalizedPattern());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      boolean boolean0 = stdDateFormat0.isColonIncludedInTimeZone();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      TimeZone timeZone0 = TimeZone.getDefault();
      stdDateFormat0.instance.setTimeZone(timeZone0);
      assertFalse(stdDateFormat0.isColonIncludedInTimeZone());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      stdDateFormat0.hashCode();
      assertFalse(stdDateFormat0.isColonIncludedInTimeZone());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      ZoneInfo zoneInfo0 = (ZoneInfo)StdDateFormat.DEFAULT_TIMEZONE;
      StdDateFormat stdDateFormat1 = stdDateFormat0.withTimeZone(zoneInfo0);
      try { 
        stdDateFormat1._parseAsISO8601("yyyy-MM-dd'T'HH:mm:ss.SSSZ", (ParsePosition) null);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Cannot parse date \"yyyy-MM-dd'T'HH:mm:ss.SSSZ\": while it seems to fit format 'yyyy-MM-dd'T'HH:mm:ss.SSSZ', parsing fails (leniency? null)
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      StdDateFormat stdDateFormat1 = stdDateFormat0.withTimeZone((TimeZone) null);
      assertNotSame(stdDateFormat1, stdDateFormat0);
      assertFalse(stdDateFormat1.isColonIncludedInTimeZone());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TimeZone timeZone0 = StdDateFormat.getDefaultTimeZone();
      Locale locale0 = Locale.JAPAN;
      StdDateFormat stdDateFormat0 = new StdDateFormat(timeZone0, locale0, (Boolean) null);
      StdDateFormat stdDateFormat1 = stdDateFormat0.withTimeZone(timeZone0);
      assertFalse(stdDateFormat1.isColonIncludedInTimeZone());
      assertSame(stdDateFormat1, stdDateFormat0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      Locale locale0 = Locale.KOREA;
      StdDateFormat stdDateFormat1 = stdDateFormat0.withLocale(locale0);
      assertNotSame(stdDateFormat1, stdDateFormat0);
      assertFalse(stdDateFormat1.isColonIncludedInTimeZone());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      Locale locale0 = Locale.US;
      StdDateFormat stdDateFormat1 = stdDateFormat0.withLocale(locale0);
      assertSame(stdDateFormat1, stdDateFormat0);
      assertFalse(stdDateFormat1.isColonIncludedInTimeZone());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      StdDateFormat stdDateFormat1 = stdDateFormat0.withColonInTimeZone(false);
      assertSame(stdDateFormat1, stdDateFormat0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      Boolean boolean0 = Boolean.valueOf(true);
      StdDateFormat stdDateFormat1 = stdDateFormat0.withLenient(boolean0);
      stdDateFormat1.setLenient(true);
      assertFalse(stdDateFormat1.isColonIncludedInTimeZone());
      assertTrue(stdDateFormat1.isLenient());
      assertNotSame(stdDateFormat1, stdDateFormat0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      stdDateFormat0.setLenient(false);
      assertFalse(stdDateFormat0.isLenient());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
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
  public void test22()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      try { 
        stdDateFormat0.parse("7KZ(DA5g{5`gt-)Ei");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Cannot parse date \"7KZ(DA5g{5`gt-)Ei\": not compatible with any of standard forms (\"yyyy-MM-dd'T'HH:mm:ss.SSSZ\", \"yyyy-MM-dd'T'HH:mm:ss.SSS\", \"EEE, dd MMM yyyy HH:mm:ss zzz\", \"yyyy-MM-dd\")
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      stdDateFormat0.parse("7");
      assertFalse(stdDateFormat0.isColonIncludedInTimeZone());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      try { 
        stdDateFormat0.parse(", ");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Cannot parse date \",\": not compatible with any of standard forms (\"yyyy-MM-dd'T'HH:mm:ss.SSSZ\", \"yyyy-MM-dd'T'HH:mm:ss.SSS\", \"EEE, dd MMM yyyy HH:mm:ss zzz\", \"yyyy-MM-dd\")
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      TimeZone timeZone0 = StdDateFormat.getDefaultTimeZone();
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      Boolean boolean0 = Boolean.FALSE;
      StdDateFormat stdDateFormat0 = new StdDateFormat(timeZone0, locale0, boolean0, true);
      MockDate mockDate0 = new MockDate((-424), (-424), 1523, 1523, 1523, 1);
      StringBuffer stringBuffer0 = new StringBuffer("0000-00-00T00:00");
      FieldPosition fieldPosition0 = new FieldPosition(3341);
      stdDateFormat0.format((Date) mockDate0, stringBuffer0, fieldPosition0);
      assertEquals("0000-00-00T00:001445-01-04T12:23:01.000+00:00", stringBuffer0.toString());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      // Undeclared exception!
      try { 
        stdDateFormat0.instance.format((Date) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.Calendar", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      SimpleTimeZone simpleTimeZone0 = new SimpleTimeZone(677, "yUfOwVsU2n");
      Locale locale0 = Locale.CANADA;
      MockDate mockDate0 = new MockDate(677, 677, 677, 17, 17);
      StringBuffer stringBuffer0 = new StringBuffer();
      stdDateFormat0._format(simpleTimeZone0, locale0, mockDate0, stringBuffer0);
      assertEquals("2635-04-08T17:17:00.677+0000", stringBuffer0.toString());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      MockDate mockDate0 = new MockDate(43, 43, 13);
      StringBuffer stringBuffer0 = new StringBuffer((CharSequence) "yyyy-MM-dd'T'HH:mm:ss.SSSZ");
      SimpleTimeZone simpleTimeZone0 = new SimpleTimeZone((-631), "");
      stdDateFormat0._format(simpleTimeZone0, (Locale) null, mockDate0, stringBuffer0);
      assertEquals("yyyy-MM-dd'T'HH:mm:ss.SSSZ1946-08-12T23:59:59.369-0000", stringBuffer0.toString());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      StdDateFormat stdDateFormat1 = stdDateFormat0.withColonInTimeZone(true);
      MockDate mockDate0 = new MockDate(43, 43, 13);
      StringBuffer stringBuffer0 = new StringBuffer((CharSequence) "yyyy-MM-dd'T'HH:mm:ss.SSSZ");
      SimpleTimeZone simpleTimeZone0 = new SimpleTimeZone((-631), "");
      stdDateFormat1._format(simpleTimeZone0, (Locale) null, mockDate0, stringBuffer0);
      assertEquals("yyyy-MM-dd'T'HH:mm:ss.SSSZ1946-08-12T23:59:59.369-00:00", stringBuffer0.toString());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      Integer integer0 = new Integer(2);
      String string0 = stdDateFormat0.format((Object) integer0);
      assertEquals("1970-01-01T00:00:00.002+0000", string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      Locale locale0 = Locale.JAPANESE;
      MockDate mockDate0 = new MockDate((-1875), (-950), (-456));
      StringBuffer stringBuffer0 = new StringBuffer((CharSequence) "[Uq8IS=a'^Yw");
      TimeZone timeZone0 = TimeZone.getTimeZone("yyyy-MM-dd'T'HH:mm:ss.SSSZ");
      stdDateFormat0._format(timeZone0, locale0, mockDate0, stringBuffer0);
      assertEquals("[Uq8IS=a'^Yw0057-08-01T00:00:00.000+0000", stringBuffer0.toString());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      String string0 = stdDateFormat0.instance.toPattern();
      assertFalse(stdDateFormat0.isColonIncludedInTimeZone());
      assertEquals("[one of: 'yyyy-MM-dd'T'HH:mm:ss.SSSZ', 'EEE, dd MMM yyyy HH:mm:ss zzz' (lenient)]", string0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      boolean boolean0 = stdDateFormat0.equals((Object) null);
      assertFalse(stdDateFormat0.isColonIncludedInTimeZone());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      boolean boolean0 = stdDateFormat0.equals(stdDateFormat0);
      assertFalse(stdDateFormat0.isColonIncludedInTimeZone());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      try { 
        stdDateFormat0.parse("2.2250738585072012e-308");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Cannot parse date \"2.2250738585072012e-308\": not compatible with any of standard forms (\"yyyy-MM-dd'T'HH:mm:ss.SSSZ\", \"yyyy-MM-dd'T'HH:mm:ss.SSS\", \"EEE, dd MMM yyyy HH:mm:ss zzz\", \"yyyy-MM-dd\")
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      try { 
        stdDateFormat0.parse("0000-Zl00-00T00:00");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Cannot parse date \"0000-Zl00-00T00:00\": not compatible with any of standard forms (\"yyyy-MM-dd'T'HH:mm:ss.SSSZ\", \"yyyy-MM-dd'T'HH:mm:ss.SSS\", \"EEE, dd MMM yyyy HH:mm:ss zzz\", \"yyyy-MM-dd\")
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      try { 
        stdDateFormat0.parse("0000-0000");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Cannot parse date \"0000-0000\": while it seems to fit format 'yyyy-MM-dd', parsing fails (leniency? null)
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.CHINA;
      MockSimpleDateFormat mockSimpleDateFormat0 = (MockSimpleDateFormat)StdDateFormat.getISO8601Format(timeZone0, locale0);
      assertEquals("aaaa-nn-jj'T'HH:mm:ss.SSSZ", mockSimpleDateFormat0.toLocalizedPattern());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      try { 
        stdDateFormat0.parse("yyyy-MM-dd'T'HH:mm:ss.SSSZ");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Cannot parse date \"yyyy-MM-dd'T'HH:mm:ss.SSSZ\": not compatible with any of standard forms (\"yyyy-MM-dd'T'HH:mm:ss.SSSZ\", \"yyyy-MM-dd'T'HH:mm:ss.SSS\", \"EEE, dd MMM yyyy HH:mm:ss zzz\", \"yyyy-MM-dd\")
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      StdDateFormat stdDateFormat1 = stdDateFormat0.withLenient((Boolean) null);
      assertSame(stdDateFormat1, stdDateFormat0);
  }
}
