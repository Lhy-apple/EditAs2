/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:18:51 GMT 2023
 */

package org.apache.commons.lang3.time;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.text.ParseException;
import java.text.ParsePosition;
import java.util.Date;
import java.util.Locale;
import java.util.TimeZone;
import java.util.regex.Pattern;
import org.apache.commons.lang3.time.FastDateParser;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class FastDateParser_ESTest extends FastDateParser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMAN;
      FastDateParser fastDateParser0 = new FastDateParser("D+|E+|F+|G+|H+|K+|M+|S+|W+|Z+|a+|dE|h+|k+|m+|s+|w+|y+|z+|''|'[^']++(''[^']*+)*+'|[^'A-Za-z]++", timeZone0, locale0);
      boolean boolean0 = fastDateParser0.equals(fastDateParser0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.UK;
      FastDateParser fastDateParser0 = new FastDateParser("] w#WZs3%BJMgw^", timeZone0, locale0);
      assertEquals("] w#WZs3%BJMgw^", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getTimeZone("{mSV6]8`|8ZR]{");
      Locale locale0 = FastDateParser.JAPANESE_IMPERIAL;
      FastDateParser fastDateParser0 = new FastDateParser("{mSV6]8`|8ZR]{", timeZone0, locale0);
      assertEquals("{mSV6]8`|8ZR]{", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.ITALY;
      FastDateParser fastDateParser0 = new FastDateParser("hA}$+%SeY,_", timeZone0, locale0);
      Date date0 = fastDateParser0.parse("9ON|*sSh47}w");
      assertEquals("Fri Feb 14 20:21:21 GMT 2014", date0.toString());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMANY;
      FastDateParser fastDateParser0 = new FastDateParser("|uMT?f", timeZone0, locale0);
      fastDateParser0.hashCode();
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMANY;
      FastDateParser fastDateParser0 = new FastDateParser("GMTBET", timeZone0, locale0);
      String string0 = fastDateParser0.toString();
      assertEquals("FastDateParser[GMTBET,de_DE,GMT]", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      TimeZone timeZone0 = TimeZone.getDefault();
      FastDateParser fastDateParser0 = new FastDateParser("m@ o;X`X%d\"?ATmQLZ", timeZone0, locale0);
      // Undeclared exception!
      try { 
        fastDateParser0.parseObject("m@ o;X`X%d\"?ATmQLZ", (ParsePosition) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.lang3.time.FastDateParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMANY;
      FastDateParser fastDateParser0 = new FastDateParser("|uMT?f", timeZone0, locale0);
      String string0 = fastDateParser0.getPattern();
      assertEquals("|uMT?f", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMANY;
      FastDateParser fastDateParser0 = new FastDateParser("6sJhTRy-<aJ)5MF~", timeZone0, locale0);
      Locale locale1 = fastDateParser0.getLocale();
      assertEquals("", locale1.getVariant());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.JAPAN;
      FastDateParser fastDateParser0 = new FastDateParser(">W&koD6gU}7bw/", timeZone0, locale0);
      try { 
        fastDateParser0.parseObject(">W&koD6gU}7bw/");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // Unparseable date: \">W&koD6gU}7bw/\" does not match >(\\p{IsNd}++)&(\\p{IsNd}++)
         //
         verifyException("org.apache.commons.lang3.time.FastDateParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMANY;
      FastDateParser fastDateParser0 = new FastDateParser("?Zj+p2jO", timeZone0, locale0);
      TimeZone timeZone1 = fastDateParser0.getTimeZone();
      assertSame(timeZone1, timeZone0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMANY;
      FastDateParser fastDateParser0 = new FastDateParser("|uMT?f", timeZone0, locale0);
      Pattern pattern0 = fastDateParser0.getParsePattern();
      assertEquals("\\|", pattern0.toString());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMANY;
      FastDateParser fastDateParser0 = null;
      try {
        fastDateParser0 = new FastDateParser("", timeZone0, locale0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid pattern
         //
         verifyException("org.apache.commons.lang3.time.FastDateParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      TimeZone timeZone0 = TimeZone.getTimeZone("m@ o=X`Xd\"?ATJmQQ");
      FastDateParser fastDateParser0 = new FastDateParser("m@ o=X`Xd\"?ATJmQQ", timeZone0, locale0);
      boolean boolean0 = fastDateParser0.equals(locale0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMAN;
      FastDateParser fastDateParser0 = new FastDateParser("D+|E+|F+|G+|H+|K+|M+|S+|W+|Z+|a+|dE|h+|k+|m+|s+|w+|y+|z+|''|'[^']++(''[^']*+)*+'|[^'A-Za-z]++", timeZone0, locale0);
      FastDateParser fastDateParser1 = new FastDateParser("GMTAET", timeZone0, locale0);
      boolean boolean0 = fastDateParser0.equals(fastDateParser1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMANY;
      FastDateParser fastDateParser0 = new FastDateParser(" is not a supported timezone name", timeZone0, locale0);
      TimeZone timeZone1 = TimeZone.getTimeZone("PNT");
      FastDateParser fastDateParser1 = new FastDateParser(" is not a supported timezone name", timeZone1, locale0);
      boolean boolean0 = fastDateParser0.equals(fastDateParser1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMANY;
      FastDateParser fastDateParser0 = new FastDateParser("|uMT?f", timeZone0, locale0);
      Locale locale1 = Locale.KOREAN;
      FastDateParser fastDateParser1 = new FastDateParser("|uMT?f", timeZone0, locale1);
      boolean boolean0 = fastDateParser0.equals(fastDateParser1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getTimeZone("\"`9i=[-'/?i+}?~");
      Locale locale0 = FastDateParser.JAPANESE_IMPERIAL;
      FastDateParser fastDateParser0 = new FastDateParser("\"`9i=[-'/?i+}?~", timeZone0, locale0);
      try { 
        fastDateParser0.parse("6D");
        fail("Expecting exception: ParseException");
      
      } catch(ParseException e) {
         //
         // (The ja_JP_JP_#u-ca-japanese locale does not support dates before 1868 AD)
         // Unparseable date: \"6D\" does not match \"`9
         //
         verifyException("org.apache.commons.lang3.time.FastDateParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.CANADA_FRENCH;
      FastDateParser fastDateParser0 = new FastDateParser("2$a", timeZone0, locale0);
      assertEquals("2$a", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.ROOT;
      FastDateParser fastDateParser0 = new FastDateParser("^([\bEr;fKQ7y[%d u_", timeZone0, locale0);
      assertEquals("^([\bEr;fKQ7y[%d u_", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.FRANCE;
      FastDateParser fastDateParser0 = new FastDateParser("(dM}TA?eMqY", timeZone0, locale0);
      assertEquals("(dM}TA?eMqY", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMANY;
      FastDateParser fastDateParser0 = new FastDateParser("|uMT?f", timeZone0, locale0);
      int int0 = fastDateParser0.adjustYear(63);
      assertEquals(1963, int0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMANY;
      FastDateParser fastDateParser0 = new FastDateParser("|uMT?f", timeZone0, locale0);
      int int0 = fastDateParser0.adjustYear(0);
      assertEquals(2000, int0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMAN;
      FastDateParser fastDateParser0 = new FastDateParser(")M[Qvkla/`rz6\tZq", timeZone0, locale0);
      assertEquals(")M[Qvkla/`rz6\tZq", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMAN;
      FastDateParser fastDateParser0 = new FastDateParser("*n<EndnlhBKfKPhdyU", timeZone0, locale0);
      assertEquals("*n<EndnlhBKfKPhdyU", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      TimeZone timeZone0 = TimeZone.getTimeZone(",");
      FastDateParser fastDateParser0 = new FastDateParser(",", timeZone0, locale0);
      assertEquals(",", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMAN;
      FastDateParser fastDateParser0 = new FastDateParser("-", timeZone0, locale0);
      assertEquals("-", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.TAIWAN;
      FastDateParser fastDateParser0 = new FastDateParser(".51@0L|ea#n86/a=", timeZone0, locale0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.TAIWAN;
      FastDateParser fastDateParser0 = new FastDateParser("w/!)lns( ?gtXROg", timeZone0, locale0);
      assertEquals("w/!)lns( ?gtXROg", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.TAIWAN;
      FastDateParser fastDateParser0 = new FastDateParser("0^bIIOL80Y9b48", timeZone0, locale0);
      assertEquals("0^bIIOL80Y9b48", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMAN;
      FastDateParser fastDateParser0 = new FastDateParser("1=", timeZone0, locale0);
      assertEquals("1=", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.ROOT;
      FastDateParser fastDateParser0 = new FastDateParser("h4W5e.w", timeZone0, locale0);
      assertEquals("h4W5e.w", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.CANADA_FRENCH;
      FastDateParser fastDateParser0 = new FastDateParser("K7^OkyIRa'D7SW+e&", timeZone0, locale0);
      assertEquals("K7^OkyIRa'D7SW+e&", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.SIMPLIFIED_CHINESE;
      FastDateParser fastDateParser0 = new FastDateParser("8'`U", timeZone0, locale0);
      assertEquals("8'`U", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getTimeZone("9R&BO*p<");
      Locale locale0 = FastDateParser.JAPANESE_IMPERIAL;
      FastDateParser fastDateParser0 = new FastDateParser("9R&BO*p<", timeZone0, locale0);
      assertEquals("9R&BO*p<", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      TimeZone timeZone0 = TimeZone.getTimeZone(":dC*,r(JATnF");
      FastDateParser fastDateParser0 = new FastDateParser(":dC*,r(JATnF", timeZone0, locale0);
      assertEquals(":dC*,r(JATnF", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMAN;
      FastDateParser fastDateParser0 = new FastDateParser(";v80d1xa2<^QR1", timeZone0, locale0);
      assertEquals(";v80d1xa2<^QR1", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.TAIWAN;
      FastDateParser fastDateParser0 = new FastDateParser("<i+;xG `2WCge?jo4i", timeZone0, locale0);
      assertEquals("<i+;xG `2WCge?jo4i", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      TimeZone timeZone0 = TimeZone.getTimeZone("=r;C0xqI[");
      FastDateParser fastDateParser0 = new FastDateParser("=r;C0xqI[", timeZone0, locale0);
      assertEquals("=r;C0xqI[", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.TAIWAN;
      FastDateParser fastDateParser0 = new FastDateParser("F", timeZone0, locale0);
      assertEquals("F", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.GERMAN;
      FastDateParser fastDateParser0 = new FastDateParser("_E nY>^ F", timeZone0, locale0);
      assertEquals("_E nY>^ F", fastDateParser0.getPattern());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      TimeZone timeZone0 = TimeZone.getDefault();
      Locale locale0 = Locale.TAIWAN;
      FastDateParser fastDateParser0 = new FastDateParser("`|54gxe.:8t", timeZone0, locale0);
      assertEquals("`|54gxe.:8t", fastDateParser0.getPattern());
  }
}