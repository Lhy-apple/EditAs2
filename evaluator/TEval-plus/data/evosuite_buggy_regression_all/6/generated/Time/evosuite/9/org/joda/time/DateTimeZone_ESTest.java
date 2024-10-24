/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 01:02:03 GMT 2023
 */

package org.joda.time;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Locale;
import java.util.Set;
import java.util.SimpleTimeZone;
import java.util.TimeZone;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.joda.time.DateTime;
import org.joda.time.DateTimeZone;
import org.joda.time.LocalDateTime;
import org.joda.time.ReadableInstant;
import org.joda.time.tz.CachedDateTimeZone;
import org.joda.time.tz.FixedDateTimeZone;
import org.joda.time.tz.NameProvider;
import org.joda.time.tz.Provider;
import org.joda.time.tz.UTCProvider;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DateTimeZone_ESTest extends DateTimeZone_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      assertNotNull(dateTimeZone0);
      
      String string0 = dateTimeZone0.getShortName((-9223372036826397795L));
      assertEquals("-07:52:58", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      Object object0 = dateTimeZone0.UTC.writeReplace();
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetHours((-14));
      TimeZone timeZone0 = dateTimeZone0.toTimeZone();
      DateTimeZone dateTimeZone1 = DateTimeZone.forTimeZone(timeZone0);
      assertSame(dateTimeZone1, dateTimeZone0);
      assertEquals("-14:00", dateTimeZone0.getID());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forID("-08:00");
      assertNotNull(dateTimeZone0);
      
      String string0 = dateTimeZone0.getName(1428L, (Locale) null);
      assertEquals("-08:00", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      assertNotNull(dateTimeZone0);
      
      String string0 = dateTimeZone0.getName((-2336L));
      assertEquals("-08:00", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      assertNotNull(dateTimeZone0);
      
      String string0 = dateTimeZone0.toString();
      assertEquals("America/Los_Angeles", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Set<String> set0 = DateTimeZone.getAvailableIDs();
      assertFalse(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      NameProvider nameProvider0 = DateTimeZone.getNameProvider();
      DateTimeZone.setNameProvider(nameProvider0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      DateTime dateTime0 = new DateTime(2478L, dateTimeZone0);
      int int0 = dateTimeZone0.UTC.getOffset((ReadableInstant) dateTime0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Provider provider0 = DateTimeZone.getProvider();
      assertNotNull(provider0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      TimeZone timeZone0 = dateTimeZone0.toTimeZone();
      DateTimeZone dateTimeZone1 = DateTimeZone.forTimeZone(timeZone0);
      assertSame(dateTimeZone0, dateTimeZone1);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      UTCProvider uTCProvider0 = new UTCProvider();
      DateTimeZone.setProvider(uTCProvider0);
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      assertTrue(dateTimeZone0.isFixed());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      DateTimeZone.setDefault(dateTimeZone0);
      assertEquals("America/Los_Angeles", dateTimeZone0.getID());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      // Undeclared exception!
      try { 
        DateTimeZone.setDefault((DateTimeZone) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The datetime zone must not be null
         //
         verifyException("org.joda.time.DateTimeZone", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forID((String) null);
      assertEquals("America/Los_Angeles", dateTimeZone0.getID());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forID("UTC");
      assertTrue(dateTimeZone0.isFixed());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forID("+00:00");
      assertNotNull(dateTimeZone0);
      assertEquals("UTC", dateTimeZone0.getID());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      try { 
        DateTimeZone.forOffsetHoursMinutes(0, (-1334));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Minutes out of range: -1334
         //
         verifyException("org.joda.time.DateTimeZone", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetHours(0);
      assertEquals("UTC", dateTimeZone0.toString());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      try { 
        DateTimeZone.forOffsetHoursMinutes(3219, 3219);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Minutes out of range: 3219
         //
         verifyException("org.joda.time.DateTimeZone", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetHours(2);
      TimeZone timeZone0 = dateTimeZone0.toTimeZone();
      DateTimeZone.forTimeZone(timeZone0);
      DateTimeZone dateTimeZone1 = DateTimeZone.forTimeZone(timeZone0);
      assertEquals("+02:00", dateTimeZone0.getID());
      assertSame(dateTimeZone1, dateTimeZone0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forTimeZone((TimeZone) null);
      assertEquals("America/Los_Angeles", dateTimeZone0.toString());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      FixedDateTimeZone fixedDateTimeZone0 = (FixedDateTimeZone)DateTimeZone.UTC;
      TimeZone timeZone0 = fixedDateTimeZone0.toTimeZone();
      DateTimeZone dateTimeZone0 = DateTimeZone.forTimeZone(timeZone0);
      assertSame(dateTimeZone0, fixedDateTimeZone0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getTimeZone("PRT");
      UTCProvider uTCProvider0 = new UTCProvider();
      DateTimeZone.setProvider(uTCProvider0);
      // Undeclared exception!
      try { 
        DateTimeZone.forTimeZone(timeZone0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The datetime zone id 'PRT' is not recognised
         //
         verifyException("org.joda.time.DateTimeZone", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      SimpleTimeZone simpleTimeZone0 = new SimpleTimeZone((-310), "--");
      // Undeclared exception!
      try { 
        DateTimeZone.forTimeZone(simpleTimeZone0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The datetime zone id '--' is not recognised
         //
         verifyException("org.joda.time.DateTimeZone", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetMillis(0);
      assertEquals("UTC", dateTimeZone0.getID());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      DateTimeZone.setProvider((Provider) null);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      DateTimeZone.setNameProvider((NameProvider) null);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetMillis(1366);
      assertEquals("+00:00:01.366", dateTimeZone0.toString());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      Locale locale0 = Locale.CHINESE;
      String string0 = dateTimeZone0.getShortName((-3626L), locale0);
      assertEquals("-08:00", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetHours(58);
      String string0 = dateTimeZone0.getShortName((long) 58);
      assertEquals("+58:00", string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      Locale locale0 = Locale.CANADA;
      String string0 = dateTimeZone0.getName(1121L, locale0);
      assertNotNull(string0);
      assertEquals("-08:00", string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      int int0 = dateTimeZone0.UTC.getOffset((ReadableInstant) null);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      boolean boolean0 = dateTimeZone0.isStandardOffset(10000000000L);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      boolean boolean0 = dateTimeZone0.isStandardOffset(25267840L);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      int int0 = dateTimeZone0.getOffsetFromLocal((-2114L));
      assertEquals((-28800000), int0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      int int0 = dateTimeZone0.getOffsetFromLocal(9223372036854775807L);
      assertEquals((-28378000), int0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getTimeZone("PLT");
      DateTimeZone dateTimeZone0 = DateTimeZone.forTimeZone(timeZone0);
      int int0 = dateTimeZone0.getOffsetFromLocal(9223372036854775770L);
      assertEquals(18000000, int0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetHours(2);
      CachedDateTimeZone cachedDateTimeZone0 = CachedDateTimeZone.forZone(dateTimeZone0);
      int int0 = cachedDateTimeZone0.getOffsetFromLocal(2);
      assertEquals("+02:00", cachedDateTimeZone0.getID());
      assertEquals(7200000, int0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getTimeZone("Africa/Addis_Ababa");
      DateTimeZone dateTimeZone0 = DateTimeZone.forTimeZone(timeZone0);
      CachedDateTimeZone cachedDateTimeZone0 = CachedDateTimeZone.forZone(dateTimeZone0);
      int int0 = cachedDateTimeZone0.getOffsetFromLocal(9223372036854775807L);
      assertEquals(9320000, int0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      LocalDateTime localDateTime0 = new LocalDateTime(dateTimeZone0);
      boolean boolean0 = dateTimeZone0.UTC.isLocalDateTimeGap(localDateTime0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      // Undeclared exception!
      try { 
        dateTimeZone0.convertUTCToLocal((-9223372036826397809L));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // Adding time zone offset caused overflow
         //
         verifyException("org.joda.time.DateTimeZone", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      long long0 = dateTimeZone0.convertUTCToLocal(2530L);
      assertEquals((-28797470L), long0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      // Undeclared exception!
      try { 
        dateTimeZone0.convertLocalToUTC(9223372036854775807L, true, 9223372036854775807L);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // Subtracting time zone offset caused overflow
         //
         verifyException("org.joda.time.DateTimeZone", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      long long0 = dateTimeZone0.convertLocalToUTC((-1695L), false);
      assertEquals(28798305L, long0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      // Undeclared exception!
      try { 
        dateTimeZone0.convertLocalToUTC(9223372036854775807L, false);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // Subtracting time zone offset caused overflow
         //
         verifyException("org.joda.time.DateTimeZone", e);
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getTimeZone("PRT");
      DateTimeZone dateTimeZone0 = DateTimeZone.forTimeZone(timeZone0);
      long long0 = dateTimeZone0.convertLocalToUTC((-765399600001L), true);
      assertEquals((-765385200001L), long0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      long long0 = dateTimeZone0.UTC.getMillisKeepLocal((DateTimeZone) null, 0L);
      assertEquals(28800000L, long0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      long long0 = dateTimeZone0.getMillisKeepLocal(dateTimeZone0, 9223372036854775793L);
      assertEquals(9223372036854775793L, long0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      // Undeclared exception!
      try { 
        dateTimeZone0.isLocalDateTimeGap((LocalDateTime) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.joda.time.DateTimeZone", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      long long0 = dateTimeZone0.adjustOffset((-5752800003L), true);
      assertEquals((-5752800003L), long0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      long long0 = dateTimeZone0.adjustOffset((-204L), true);
      assertEquals((-204L), long0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      long long0 = dateTimeZone0.adjustOffset((-9223372036854775771L), false);
      assertEquals((-9223372036854775771L), long0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      long long0 = dateTimeZone0.adjustOffset(9223372036854775807L, false);
      assertEquals(9223372036854775807L, long0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      long long0 = dateTimeZone0.adjustOffset((-5756400001L), false);
      assertEquals((-5756400001L), long0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      long long0 = dateTimeZone0.adjustOffset((-5752800003L), false);
      assertEquals((-5756400003L), long0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      long long0 = dateTimeZone0.adjustOffset((-5756400001L), true);
      assertEquals((-5752800001L), long0);
  }
}
