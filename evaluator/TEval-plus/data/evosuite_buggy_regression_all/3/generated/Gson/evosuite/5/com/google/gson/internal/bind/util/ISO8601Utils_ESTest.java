/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:32:23 GMT 2023
 */

package com.google.gson.internal.bind.util;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.gson.internal.bind.util.ISO8601Utils;
import java.text.ParseException;
import java.text.ParsePosition;
import java.util.Date;
import java.util.SimpleTimeZone;
import java.util.TimeZone;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ISO8601Utils_ESTest extends ISO8601Utils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      MockDate mockDate0 = new MockDate(62);
      String string0 = ISO8601Utils.format((Date) mockDate0);
      assertEquals("1970-01-01T00:00:00Z", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ISO8601Utils iSO8601Utils0 = new ISO8601Utils();
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      MockDate mockDate0 = new MockDate();
      String string0 = ISO8601Utils.format((Date) mockDate0, true);
      assertEquals("2014-02-14T20:21:21.320Z", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MockDate mockDate0 = new MockDate(62);
      SimpleTimeZone simpleTimeZone0 = new SimpleTimeZone(62, "1970-01-01T00:00:00Z");
      String string0 = ISO8601Utils.format((Date) mockDate0, true, (TimeZone) simpleTimeZone0);
      assertEquals("1970-01-01T00:00:00.124+00:00", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MockDate mockDate0 = new MockDate();
      SimpleTimeZone simpleTimeZone0 = new SimpleTimeZone((-1), "X{)6tQU\"|a");
      String string0 = ISO8601Utils.format((Date) mockDate0, false, (TimeZone) simpleTimeZone0);
      assertEquals("2014-02-14T20:21:21-00:00", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(1);
      try { 
        ISO8601Utils.parse("6126495-03-01T04:01:01Z", parsePosition0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Failed to parse date [\"6126495-03-01T04:01:01Z']: Mismatching time zone indicator: GMT-01T04:01:01Z given, resolves to GMT
         //
         verifyException("com.google.gson.internal.bind.util.ISO8601Utils", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(0);
      Date date0 = ISO8601Utils.parse("1970-01-01T00:00:00.124+00:00", parsePosition0);
      assertEquals("Fri Feb 14 20:21:21 GMT 2014", date0.toString());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(0);
      try { 
        ISO8601Utils.parse("208770104-12-13T03:22:26Z", parsePosition0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Failed to parse date [\"208770104-12-13T03:22:26Z']: Invalid time zone indicator '4'
         //
         verifyException("com.google.gson.internal.bind.util.ISO8601Utils", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(0);
      Date date0 = ISO8601Utils.parse("2014-02-14T20:21:21-00:00", parsePosition0);
      assertEquals("Fri Feb 14 20:21:21 GMT 2014", date0.toString());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(0);
      Date date0 = ISO8601Utils.parse("2014-02-14T20:21:21.320Z", parsePosition0);
      assertEquals("Fri Feb 14 20:21:21 GMT 2014", date0.toString());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition((-2603));
      try { 
        ISO8601Utils.parse((String) null, parsePosition0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Failed to parse date [null]: (java.lang.NumberFormatException)
         //
         verifyException("com.google.gson.internal.bind.util.ISO8601Utils", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(12);
      try { 
        ISO8601Utils.parse("", parsePosition0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Failed to parse date [\"']: (java.lang.NumberFormatException)
         //
         verifyException("com.google.gson.internal.bind.util.ISO8601Utils", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(1);
      try { 
        ISO8601Utils.parse("+0000", parsePosition0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Failed to parse date [\"+0000']: +0000
         //
         verifyException("com.google.gson.internal.bind.util.ISO8601Utils", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(2147483645);
      try { 
        ISO8601Utils.parse("#K'NPf", parsePosition0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Failed to parse date [\"#K'NPf']: #K'NPf
         //
         verifyException("com.google.gson.internal.bind.util.ISO8601Utils", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(1);
      try { 
        ISO8601Utils.parse("Mismatching time zone indicator: ", parsePosition0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Failed to parse date [\"Mismatching time zone indicator: ']: Invalid number: isma
         //
         verifyException("com.google.gson.internal.bind.util.ISO8601Utils", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(0);
      try { 
        ISO8601Utils.parse("6126494-01-29T04:00:00Z", parsePosition0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Failed to parse date [\"6126494-01-29T04:00:00Z']: Invalid number: 4-
         //
         verifyException("com.google.gson.internal.bind.util.ISO8601Utils", e);
      }
  }
}