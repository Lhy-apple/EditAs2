/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:47:20 GMT 2023
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
import java.util.Date;
import java.util.Locale;
import java.util.TimeZone;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.System;
import org.evosuite.runtime.mock.java.text.MockSimpleDateFormat;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StdDateFormat_ESTest extends StdDateFormat_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      MockSimpleDateFormat mockSimpleDateFormat0 = (MockSimpleDateFormat)StdDateFormat.getISO8601Format(timeZone0);
      assertEquals("yyyy-MM-dd'T'HH:mm:ss.SSSZ", mockSimpleDateFormat0.toPattern());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      MockSimpleDateFormat mockSimpleDateFormat0 = (MockSimpleDateFormat)StdDateFormat.getRFC1123Format((TimeZone) null);
      assertEquals("EEE, dd MMM yyyy HH:mm:ss zzz", mockSimpleDateFormat0.toLocalizedPattern());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      TimeZone timeZone0 = stdDateFormat0.getTimeZone();
      assertNull(timeZone0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      stdDateFormat0.hashCode();
      stdDateFormat0.instance.toString();
      ParsePosition parsePosition0 = new ParsePosition(0);
      System.setCurrentTimeMillis((-1L));
      StdDateFormat stdDateFormat1 = stdDateFormat0.clone();
      ParsePosition parsePosition1 = new ParsePosition(0);
      stdDateFormat1.parse("5T_\"xI!Ek tkZG?", parsePosition1);
      boolean boolean0 = stdDateFormat0.isLenient();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      ZoneOffset zoneOffset0 = ZoneOffset.ofHoursMinutes(9, 9);
      TimeZone timeZone0 = TimeZone.getTimeZone((ZoneId) zoneOffset0);
      StdDateFormat stdDateFormat1 = stdDateFormat0.withTimeZone(timeZone0);
      assertNotSame(stdDateFormat1, stdDateFormat0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      StdDateFormat stdDateFormat1 = stdDateFormat0.withTimeZone((TimeZone) null);
      StdDateFormat stdDateFormat2 = stdDateFormat1.withTimeZone((TimeZone) null);
      boolean boolean0 = stdDateFormat2.equals(stdDateFormat1);
      assertNotSame(stdDateFormat2, stdDateFormat0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      StdDateFormat stdDateFormat1 = stdDateFormat0.withLocale(locale0);
      try { 
        stdDateFormat1.parse("yyyy-MM-dd'T'HH:mm:ss.SSSZ");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Can not parse date \"yyyy-MM-dd'T'HH:mm:ss.SSSZ\": not compatible with any of standard forms (\"yyyy-MM-dd'T'HH:mm:ss.SSSZ\", \"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'\", \"EEE, dd MMM yyyy HH:mm:ss zzz\", \"yyyy-MM-dd\")
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Locale locale0 = Locale.US;
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      StdDateFormat stdDateFormat1 = stdDateFormat0.withLocale(locale0);
      assertSame(stdDateFormat1, stdDateFormat0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getTimeZone("00");
      Locale locale0 = Locale.GERMANY;
      StdDateFormat stdDateFormat0 = new StdDateFormat(timeZone0, locale0);
      stdDateFormat0.setTimeZone(timeZone0);
      assertEquals("GMT", timeZone0.getID());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      TimeZone timeZone0 = TimeZone.getDefault();
      stdDateFormat0.instance.setTimeZone(timeZone0);
      assertEquals("GMT", timeZone0.getID());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      stdDateFormat0.instance.setLenient(false);
      assertTrue(stdDateFormat0.isLenient());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      boolean boolean0 = stdDateFormat0.isLenient();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TimeZone timeZone0 = StdDateFormat.getDefaultTimeZone();
      Locale locale0 = Locale.US;
      StdDateFormat stdDateFormat0 = new StdDateFormat(timeZone0, locale0);
      try { 
        stdDateFormat0.parse("yyyy-MM-dd'T'HH:mm:ss.SSSZ");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Can not parse date \"yyyy-MM-dd'T'HH:mm:ss.SSSZ\": not compatible with any of standard forms (\"yyyy-MM-dd'T'HH:mm:ss.SSSZ\", \"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'\", \"EEE, dd MMM yyyy HH:mm:ss zzz\", \"yyyy-MM-dd\")
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      try { 
        stdDateFormat0.parse("4{_8-+[C*0H]b");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Can not parse date \"4{_8-+[C*0H]bZ\": while it seems to fit format 'yyyy-MM-dd'T'HH:mm:ss.SSS'Z'', parsing fails (leniency? null)
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      // Undeclared exception!
      try { 
        stdDateFormat0.parse("-");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"-\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      try { 
        stdDateFormat0.parse("2.2250738585072012e-308");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Can not parse date \"2.2250738585072012e-308\": not compatible with any of standard forms (\"yyyy-MM-dd'T'HH:mm:ss.SSSZ\", \"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'\", \"EEE, dd MMM yyyy HH:mm:ss zzz\", \"yyyy-MM-dd\")
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      try { 
        stdDateFormat0.parse("G");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Can not parse date \"G\": not compatible with any of standard forms (\"yyyy-MM-dd'T'HH:mm:ss.SSSZ\", \"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'\", \"EEE, dd MMM yyyy HH:mm:ss zzz\", \"yyyy-MM-dd\")
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      stdDateFormat0.parse("00");
      assertFalse(stdDateFormat0.isLenient());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      Object object0 = stdDateFormat0.parseObject("2014-02-14T20:21:21.320+0000");
      assertEquals("Fri Feb 14 20:21:21 GMT 2014", object0.toString());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      ParsePosition parsePosition0 = new ParsePosition(6);
      // Undeclared exception!
      try { 
        stdDateFormat0.parse("-", parsePosition0);
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"-\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      try { 
        stdDateFormat0.parseObject("09");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Format.parseObject(String) failed
         //
         verifyException("java.text.Format", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      try { 
        stdDateFormat0.parseObject("b");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Format.parseObject(String) failed
         //
         verifyException("java.text.Format", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      MockDate mockDate0 = new MockDate();
      String string0 = stdDateFormat0.instance.format((Date) mockDate0);
      assertEquals("2014-02-14T20:21:21.320+0000", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      String string0 = stdDateFormat0.toString();
      assertEquals("DateFormat com.fasterxml.jackson.databind.util.StdDateFormat(locale: en_US)", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      boolean boolean0 = stdDateFormat0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      ParsePosition parsePosition0 = new ParsePosition(2125);
      try { 
        stdDateFormat0.parseAsISO8601(".000", parsePosition0, false);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Can not parse date \".000\": while it seems to fit format 'yyyy-MM-dd', parsing fails (leniency? null)
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      ZoneOffset zoneOffset0 = ZoneOffset.ofHoursMinutes(9, 9);
      TimeZone.getTimeZone((ZoneId) zoneOffset0);
      ParsePosition parsePosition0 = new ParsePosition(9);
      try { 
        stdDateFormat0.parseAsISO8601("yyyy-MM-dd'T'HH:mm:ss.SSSZ", parsePosition0, true);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Can not parse date \"yyyy-MM-dd'T'HH:mm:ss.SSSZ\": while it seems to fit format 'yyyy-MM-dd'T'HH:mm:ss.SSS'Z'', parsing fails (leniency? null)
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("D}xrskti5Z+v'z+?`", (ParsePosition) null, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("86.H\\-4B", (ParsePosition) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("J8i66.H\r-4B", (ParsePosition) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("J8i66.pH\r-4B", (ParsePosition) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("14jTJW+OE!$S", (ParsePosition) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("2nBF0 |b&dW", (ParsePosition) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601(" y1z,S6>W", (ParsePosition) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601(" !y1z,S6>W", (ParsePosition) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601(")", (ParsePosition) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      try { 
        stdDateFormat0.parse("yyyy-MM-dd'T'HH:mm:ss.SSSZ");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Can not parse date \"yyyy-MM-dd'T'HH:mm:ss.SSSZ\": not compatible with any of standard forms (\"yyyy-MM-dd'T'HH:mm:ss.SSSZ\", \"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'\", \"EEE, dd MMM yyyy HH:mm:ss zzz\", \"yyyy-MM-dd\")
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("yyyy-MM-dd", (ParsePosition) null, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("C41)i-6`T_", (ParsePosition) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getTimeZone("00");
      Locale locale0 = Locale.GERMANY;
      MockSimpleDateFormat mockSimpleDateFormat0 = (MockSimpleDateFormat)StdDateFormat.getRFC1123Format(timeZone0, locale0);
      assertEquals("EEE, tt MMM uuuu HH:mm:ss zzz", mockSimpleDateFormat0.toLocalizedPattern());
  }
}