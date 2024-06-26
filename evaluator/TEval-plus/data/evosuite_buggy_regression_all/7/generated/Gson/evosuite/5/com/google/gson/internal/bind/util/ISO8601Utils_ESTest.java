/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:43:32 GMT 2023
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
      MockDate mockDate0 = new MockDate();
      String string0 = ISO8601Utils.format((Date) mockDate0);
      assertEquals("2014-02-14T20:21:21Z", string0);
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
      MockDate mockDate0 = new MockDate((-182L));
      SimpleTimeZone simpleTimeZone0 = new SimpleTimeZone(1551, "");
      String string0 = ISO8601Utils.format((Date) mockDate0, false, (TimeZone) simpleTimeZone0);
      assertEquals("1970-01-01T00:00:01+00:00", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MockDate mockDate0 = new MockDate((-182L));
      SimpleTimeZone simpleTimeZone0 = new SimpleTimeZone((-502), "/e1r2zRMou#-m6");
      String string0 = ISO8601Utils.format((Date) mockDate0, false, (TimeZone) simpleTimeZone0);
      assertEquals("1969-12-31T23:59:59-00:00", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(1);
      try { 
        ISO8601Utils.parse("184834560-10-28T01:01:01Z", parsePosition0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Failed to parse date [\"184834560-10-28T01:01:01Z']: Mismatching time zone indicator: GMT-10-28T01:01:01Z given, resolves to GMT
         //
         verifyException("com.google.gson.internal.bind.util.ISO8601Utils", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(0);
      Date date0 = ISO8601Utils.parse("2014-02-14T20:21:21.320Z", parsePosition0);
      assertEquals("Fri Feb 14 20:21:21 GMT 2014", date0.toString());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(0);
      Date date0 = ISO8601Utils.parse("2014-02-14T20:21:21Z", parsePosition0);
      assertEquals("Fri Feb 14 20:21:21 GMT 2014", date0.toString());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(0);
      try { 
        ISO8601Utils.parse("208737700-06-13T06:35:32Z", parsePosition0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Failed to parse date [\"208737700-06-13T06:35:32Z']: Invalid time zone indicator '0'
         //
         verifyException("com.google.gson.internal.bind.util.ISO8601Utils", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition((-113));
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
  public void test10()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(2147483645);
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
  public void test11()  throws Throwable  {
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
  public void test12()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(0);
      try { 
        ISO8601Utils.parse("veTqLK", parsePosition0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Failed to parse date [\"veTqLK']: Invalid number: veTq
         //
         verifyException("com.google.gson.internal.bind.util.ISO8601Utils", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(0);
      try { 
        ISO8601Utils.parse("4*q4", parsePosition0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Failed to parse date [\"4*q4']: Invalid number: 4*q4
         //
         verifyException("com.google.gson.internal.bind.util.ISO8601Utils", e);
      }
  }
}
