/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:29:00 GMT 2023
 */

package com.fasterxml.jackson.databind.util;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.util.StdDateFormat;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.ParsePosition;
import java.util.Date;
import java.util.Locale;
import java.util.SimpleTimeZone;
import java.util.TimeZone;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.text.MockSimpleDateFormat;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StdDateFormat_ESTest extends StdDateFormat_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      StdDateFormat stdDateFormat1 = stdDateFormat0.clone();
      assertNotSame(stdDateFormat0, stdDateFormat1);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TimeZone timeZone0 = StdDateFormat.getDefaultTimeZone();
      MockSimpleDateFormat mockSimpleDateFormat0 = (MockSimpleDateFormat)StdDateFormat.getISO8601Format(timeZone0);
      assertEquals("yyyy-MM-dd'T'HH:mm:ss.SSSZ", mockSimpleDateFormat0.toPattern());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      MockSimpleDateFormat mockSimpleDateFormat0 = (MockSimpleDateFormat)StdDateFormat.getRFC1123Format((TimeZone) null);
      assertEquals("EEE, dd MMM yyyy HH:mm:ss zzz", mockSimpleDateFormat0.toLocalizedPattern());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TimeZone timeZone0 = StdDateFormat.getDefaultTimeZone();
      Locale locale0 = Locale.GERMANY;
      Boolean boolean0 = new Boolean(true);
      StdDateFormat stdDateFormat0 = new StdDateFormat(timeZone0, locale0, boolean0);
      TimeZone timeZone1 = stdDateFormat0.getTimeZone();
      assertSame(timeZone1, timeZone0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      stdDateFormat0.setLenient(true);
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("bxTI,$J&", (ParsePosition) null, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      stdDateFormat0.hashCode();
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      StdDateFormat stdDateFormat1 = stdDateFormat0.withTimeZone((TimeZone) null);
      assertNotSame(stdDateFormat1, stdDateFormat0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getTimeZone("");
      TimeZone timeZone1 = TimeZone.getTimeZone("");
      StdDateFormat stdDateFormat0 = new StdDateFormat(timeZone1, (Locale) null);
      StdDateFormat stdDateFormat1 = stdDateFormat0.withTimeZone(timeZone0);
      assertSame(stdDateFormat1, stdDateFormat0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      Locale locale0 = Locale.US;
      StdDateFormat stdDateFormat1 = stdDateFormat0.withLocale(locale0);
      assertSame(stdDateFormat1, stdDateFormat0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      TimeZone timeZone0 = StdDateFormat.getDefaultTimeZone();
      stdDateFormat0.instance.setTimeZone(timeZone0);
      assertTrue(stdDateFormat0.isLenient());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleTimeZone simpleTimeZone0 = new SimpleTimeZone(85, "");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      Boolean boolean0 = Boolean.valueOf(true);
      StdDateFormat stdDateFormat0 = new StdDateFormat(simpleTimeZone0, locale0, boolean0);
      boolean boolean1 = stdDateFormat0.isLenient();
      assertTrue(boolean1);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleTimeZone simpleTimeZone0 = new SimpleTimeZone((-425), "");
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      StdDateFormat stdDateFormat0 = new StdDateFormat(simpleTimeZone0, locale0, (Boolean) null);
      boolean boolean0 = stdDateFormat0.isLenient();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      Date date0 = stdDateFormat0.parse("00");
      assertEquals("Thu Jan 01 00:00:00 GMT 1970", date0.toString());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      try { 
        stdDateFormat0.parse(",");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Can not parse date \",\": not compatible with any of standard forms (\"yyyy-MM-dd'T'HH:mm:ss.SSSZ\", \"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'\", \"EEE, dd MMM yyyy HH:mm:ss zzz\", \"yyyy-MM-dd\")
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
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
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      ParsePosition parsePosition0 = new ParsePosition(1);
      stdDateFormat0.parse("1901-02-01T01:01:00.000+0000", parsePosition0);
      assertEquals("java.text.ParsePosition[index=28,errorIndex=-1]", parsePosition0.toString());
      assertEquals(28, parsePosition0.getIndex());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      Date date0 = stdDateFormat0.parse("00", (ParsePosition) null);
      assertEquals("Thu Jan 01 00:00:00 GMT 1970", date0.toString());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      // Undeclared exception!
      try { 
        stdDateFormat0.parse("yyyy-MM-dd'T'HH:mm:ss.SSSZ", (ParsePosition) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      // Undeclared exception!
      try { 
        stdDateFormat0.parse("-", (ParsePosition) null);
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"-\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      MockDate mockDate0 = new MockDate(1, 1, 1, 1, 1);
      String string0 = stdDateFormat0.format((Date) mockDate0);
      assertEquals("1901-02-01T01:01:00.000+0000", string0);
      
      stdDateFormat0.parse("1901-02-01T01:01:00.000+0000");
      Date date0 = stdDateFormat0.parse("1901-02-01T01:01:00.000+0000");
      assertEquals("Fri Feb 01 01:01:00 GMT 1901", date0.toString());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      String string0 = stdDateFormat0.instance.toString();
      assertEquals("DateFormat com.fasterxml.jackson.databind.util.StdDateFormat(locale: en_US)", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      DateFormat dateFormat0 = DateFormat.getInstance();
      boolean boolean0 = stdDateFormat0.equals(dateFormat0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getTimeZone("");
      StdDateFormat stdDateFormat0 = new StdDateFormat(timeZone0, (Locale) null);
      boolean boolean0 = stdDateFormat0.equals(stdDateFormat0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      try { 
        stdDateFormat0.parse("378D~OPqGNm5");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Can not parse date \"378D~OPqGNm5\": not compatible with any of standard forms (\"yyyy-MM-dd'T'HH:mm:ss.SSSZ\", \"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'\", \"EEE, dd MMM yyyy HH:mm:ss zzz\", \"yyyy-MM-dd\")
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      // Undeclared exception!
      try { 
        stdDateFormat0.parse("00.000", (ParsePosition) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("00", (ParsePosition) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("yyyy-MM-dd'T'HH:mm:ss.SSSZ", (ParsePosition) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("yR%Eqn+1;", (ParsePosition) null, true);
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
        stdDateFormat0.parseAsISO8601("yyyy-MM-dd", (ParsePosition) null, true);
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
        stdDateFormat0.parseAsISO8601("_zSl+?k9]+", (ParsePosition) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("]sED?I u-@^", (ParsePosition) null, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("Y{BXKwq]-0fzq", (ParsePosition) null, true);
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
        stdDateFormat0.parseAsISO8601("3ynn(&eKG}-)5BA", (ParsePosition) null, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("3ynn(&eKG}-)5~BA", (ParsePosition) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("_zS+?9]+", (ParsePosition) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("D>)cstr,[e+\"V9,tw  ", (ParsePosition) null, false);
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
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("QQ$*'MR-", (ParsePosition) null, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("YX@}GIglGf", (ParsePosition) null, false);
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
      try { 
        stdDateFormat0.parse("1g97-b8s/wb");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Can not parse date \"1g97-b8s/wb000Z\": while it seems to fit format 'yyyy-MM-dd'T'HH:mm:ss.SSS'Z'', parsing fails (leniency? null)
         //
         verifyException("com.fasterxml.jackson.databind.util.StdDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      StdDateFormat stdDateFormat0 = new StdDateFormat();
      // Undeclared exception!
      try { 
        stdDateFormat0.parseAsISO8601("0 ", (ParsePosition) null, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      StdDateFormat stdDateFormat0 = StdDateFormat.instance;
      Locale locale0 = Locale.FRENCH;
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
}
