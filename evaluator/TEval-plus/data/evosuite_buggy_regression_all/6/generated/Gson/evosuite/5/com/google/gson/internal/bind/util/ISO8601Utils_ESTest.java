/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:16:31 GMT 2023
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
      MockDate mockDate0 = new MockDate(0, 0, 0);
      String string0 = ISO8601Utils.format((Date) mockDate0);
      assertEquals("1899-12-31T00:00:00Z", string0);
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
      MockDate mockDate0 = new MockDate((-874), (-874), (-874));
      SimpleTimeZone simpleTimeZone0 = new SimpleTimeZone((-874), "/:6oC");
      String string0 = ISO8601Utils.format((Date) mockDate0, false, (TimeZone) simpleTimeZone0);
      assertEquals("0950-10-07T23:59:59-00:00", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MockDate mockDate0 = new MockDate();
      SimpleTimeZone simpleTimeZone0 = new SimpleTimeZone(1906, "Qh>`rW3ha!JIddl1o&");
      String string0 = ISO8601Utils.format((Date) mockDate0, true, (TimeZone) simpleTimeZone0);
      assertEquals("2014-02-14T20:21:23.226+00:00", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(0);
      try { 
        ISO8601Utils.parse("246883-10-06T00:00:00Z", parsePosition0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Failed to parse date [\"246883-10-06T00:00:00Z']: Mismatching time zone indicator: GMT-06T00:00:00Z given, resolves to GMT
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
      try { 
        ISO8601Utils.parse("208770139-06-28T06:39:34Z", parsePosition0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Failed to parse date [\"208770139-06-28T06:39:34Z']: Invalid time zone indicator '9'
         //
         verifyException("com.google.gson.internal.bind.util.ISO8601Utils", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(0);
      Date date0 = ISO8601Utils.parse("1899-12-31T00:00:00Z", parsePosition0);
      assertEquals("Fri Feb 14 20:21:21 GMT 2014", date0.toString());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition((-4));
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
      ParsePosition parsePosition0 = new ParsePosition(13);
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
      ParsePosition parsePosition0 = new ParsePosition(2147483645);
      try { 
        ISO8601Utils.parse("<jFC,vF", parsePosition0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Failed to parse date [\"<jFC,vF']: <jFC,vF
         //
         verifyException("com.google.gson.internal.bind.util.ISO8601Utils", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(0);
      try { 
        ISO8601Utils.parse("<jFC,vF", parsePosition0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Failed to parse date [\"<jFC,vF']: Invalid number: <jFC
         //
         verifyException("com.google.gson.internal.bind.util.ISO8601Utils", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ParsePosition parsePosition0 = new ParsePosition(0);
      try { 
        ISO8601Utils.parse("1SP| D[rgS", parsePosition0);
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Failed to parse date [\"1SP| D[rgS']: Invalid number: 1SP|
         //
         verifyException("com.google.gson.internal.bind.util.ISO8601Utils", e);
      }
  }
}