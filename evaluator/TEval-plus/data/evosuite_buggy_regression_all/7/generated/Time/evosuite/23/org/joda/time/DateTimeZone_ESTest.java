/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 20:18:15 GMT 2023
 */

package org.joda.time;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.util.Locale;
import java.util.Set;
import java.util.SimpleTimeZone;
import java.util.TimeZone;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.joda.time.DateTimeZone;
import org.joda.time.LocalDateTime;
import org.joda.time.MutableDateTime;
import org.joda.time.ReadableInstant;
import org.joda.time.chrono.BuddhistChronology;
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
      
      String string0 = dateTimeZone0.getShortName(112702464L);
      assertEquals("-08:00", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CachedDateTimeZone cachedDateTimeZone0 = (CachedDateTimeZone)DateTimeZone.getDefault();
      Object object0 = ((DateTimeZone)cachedDateTimeZone0).writeReplace();
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetHours((-472));
      Locale locale0 = Locale.CHINESE;
      String string0 = dateTimeZone0.getShortName((long) (-472), locale0);
      assertEquals("-472:00", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forID("-08:00");
      assertNotNull(dateTimeZone0);
      assertEquals("-08:00", dateTimeZone0.getID());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetMillis(2068);
      String string0 = dateTimeZone0.getName((long) 2068);
      assertEquals("+00:00:02.068", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstanceUTC();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      String string0 = dateTimeZone0.toString();
      assertEquals("UTC", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Set<String> set0 = DateTimeZone.getAvailableIDs();
      assertEquals(567, set0.size());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      NameProvider nameProvider0 = DateTimeZone.getNameProvider();
      DateTimeZone.setNameProvider(nameProvider0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeZone dateTimeZone0 = buddhistChronology0.getZone();
      long long0 = dateTimeZone0.UTC.getMillisKeepLocal((DateTimeZone) null, 1);
      assertEquals(28800001L, long0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Provider provider0 = DateTimeZone.getProvider();
      DateTimeZone dateTimeZone0 = provider0.getZone("Australia/Sydney");
      int int0 = dateTimeZone0.getOffsetFromLocal(677L);
      assertEquals(36000000, int0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      TimeZone timeZone0 = dateTimeZone0.toTimeZone();
      DateTimeZone dateTimeZone1 = DateTimeZone.forTimeZone(timeZone0);
      assertSame(dateTimeZone1, dateTimeZone0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DateTimeZone.getDefault();
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      assertEquals("America/Los_Angeles", dateTimeZone0.toString());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      UTCProvider uTCProvider0 = new UTCProvider();
      DateTimeZone.setProvider(uTCProvider0);
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      assertEquals("UTC", dateTimeZone0.getID());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      DateTimeZone.setDefault(dateTimeZone0);
      assertEquals("America/Los_Angeles", dateTimeZone0.toString());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
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
  public void test15()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forID((String) null);
      assertEquals("America/Los_Angeles", dateTimeZone0.toString());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forID("UTC");
      assertEquals("UTC", dateTimeZone0.toString());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forID("+00:00");
      assertEquals("UTC", dateTimeZone0.getID());
      assertNotNull(dateTimeZone0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      try { 
        DateTimeZone.forOffsetHoursMinutes(0, 1659);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Minutes out of range: 1659
         //
         verifyException("org.joda.time.DateTimeZone", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetHoursMinutes(0, 0);
      assertEquals("UTC", dateTimeZone0.toString());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      try { 
        DateTimeZone.forOffsetHoursMinutes((-4020), (-4020));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Minutes out of range: -4020
         //
         verifyException("org.joda.time.DateTimeZone", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      try { 
        DateTimeZone.forOffsetHours(3600000);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Offset is too large
         //
         verifyException("org.joda.time.DateTimeZone", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forTimeZone((TimeZone) null);
      assertEquals("America/Los_Angeles", dateTimeZone0.getID());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ZoneOffset zoneOffset0 = ZoneOffset.UTC;
      TimeZone timeZone0 = TimeZone.getTimeZone((ZoneId) zoneOffset0);
      DateTimeZone dateTimeZone0 = DateTimeZone.forTimeZone(timeZone0);
      assertEquals("UTC", dateTimeZone0.getID());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      UTCProvider uTCProvider0 = new UTCProvider();
      DateTimeZone.setProvider(uTCProvider0);
      TimeZone timeZone0 = TimeZone.getTimeZone("America/Los_Angeles");
      // Undeclared exception!
      try { 
        DateTimeZone.forTimeZone(timeZone0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The datetime zone id 'America/Los_Angeles' is not recognised
         //
         verifyException("org.joda.time.DateTimeZone", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ZoneOffset zoneOffset0 = ZoneOffset.MIN;
      TimeZone timeZone0 = TimeZone.getTimeZone((ZoneId) zoneOffset0);
      DateTimeZone dateTimeZone0 = DateTimeZone.forTimeZone(timeZone0);
      DateTimeZone dateTimeZone1 = DateTimeZone.forTimeZone(timeZone0);
      assertSame(dateTimeZone1, dateTimeZone0);
      assertEquals("-18:00", dateTimeZone1.toString());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      SimpleTimeZone simpleTimeZone0 = new SimpleTimeZone(45, "GSY");
      DateTimeZone dateTimeZone0 = DateTimeZone.forTimeZone(simpleTimeZone0);
      assertEquals("UTC", dateTimeZone0.getID());
      assertNotNull(dateTimeZone0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetMillis(0);
      assertEquals("UTC", dateTimeZone0.getID());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      DateTimeZone.setProvider((Provider) null);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      DateTimeZone.setNameProvider((NameProvider) null);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetMillis((-28378000));
      assertEquals("-07:52:58", dateTimeZone0.toString());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      FixedDateTimeZone fixedDateTimeZone0 = (FixedDateTimeZone)DateTimeZone.UTC;
      Locale locale0 = Locale.GERMANY;
      String string0 = fixedDateTimeZone0.getName((-10800599L), locale0);
      assertNotNull(string0);
      assertEquals("+00:00", string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetMillis(1659);
      MutableDateTime mutableDateTime0 = new MutableDateTime(dateTimeZone0);
      dateTimeZone0.UTC.getOffset((ReadableInstant) mutableDateTime0);
      assertEquals("+00:00:01.659", dateTimeZone0.toString());
      assertEquals("+00:00:01.659", dateTimeZone0.getID());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      int int0 = dateTimeZone0.getOffset((ReadableInstant) null);
      assertEquals((-28800000), int0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      boolean boolean0 = dateTimeZone0.isStandardOffset(21859200000L);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetMillis((-473));
      boolean boolean0 = dateTimeZone0.isStandardOffset((-473));
      assertTrue(boolean0);
      assertEquals("-00:00:00.473", dateTimeZone0.getID());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      int int0 = dateTimeZone0.getOffsetFromLocal((-182L));
      assertEquals((-28800000), int0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      int int0 = dateTimeZone0.getOffsetFromLocal(9964000017L);
      assertEquals((-25200000), int0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      int int0 = dateTimeZone0.getOffsetFromLocal(9223372036854775807L);
      assertEquals((-28378000), int0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.forOffsetMillis(45);
      CachedDateTimeZone cachedDateTimeZone0 = CachedDateTimeZone.forZone(dateTimeZone0);
      int int0 = cachedDateTimeZone0.getOffsetFromLocal(45);
      assertEquals(45, int0);
      assertEquals("+00:00:00.045", cachedDateTimeZone0.toString());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      long long0 = dateTimeZone0.adjustOffset((-227L), false);
      assertEquals((-227L), long0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      // Undeclared exception!
      try { 
        dateTimeZone0.adjustOffset(9223372036854775807L, true);
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
      // Undeclared exception!
      try { 
        dateTimeZone0.convertLocalToUTC(9223372036854775807L, false, 9223372036854775807L);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // Subtracting time zone offset caused overflow
         //
         verifyException("org.joda.time.DateTimeZone", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      long long0 = dateTimeZone0.convertLocalToUTC(114920448L, false);
      assertEquals(143720448L, long0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      // Undeclared exception!
      try { 
        dateTimeZone0.convertLocalToUTC(9223372036854775807L, true);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // Subtracting time zone offset caused overflow
         //
         verifyException("org.joda.time.DateTimeZone", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Provider provider0 = DateTimeZone.getProvider();
      DateTimeZone dateTimeZone0 = provider0.getZone("Australia/Sydney");
      // Undeclared exception!
      try { 
        dateTimeZone0.convertLocalToUTC((-9223372036854775808L), false);
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
      Provider provider0 = DateTimeZone.getProvider();
      DateTimeZone dateTimeZone0 = provider0.getZone("Australia/Sydney");
      // Undeclared exception!
      try { 
        dateTimeZone0.convertLocalToUTC((-9223372036854775808L), true);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Illegal instant due to time zone offset transition: -292275055-05-16T16:47:04.192 (Australia/Sydney)
         //
         verifyException("org.joda.time.DateTimeZone", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      long long0 = dateTimeZone0.convertLocalToUTC((-348L), true);
      assertEquals(28799652L, long0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      long long0 = dateTimeZone0.getMillisKeepLocal(dateTimeZone0, 0L);
      assertEquals(0L, long0);
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
      boolean boolean0 = dateTimeZone0.UTC.isLocalDateTimeGap((LocalDateTime) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      long long0 = dateTimeZone0.adjustOffset(1230L, true);
      assertEquals(1230L, long0);
  }
}