/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:51:32 GMT 2023
 */

package org.jsoup.helper;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLStreamHandler;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.ListIterator;
import java.util.function.Predicate;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.net.MockURL;
import org.jsoup.helper.StringUtil;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StringUtil_ESTest extends StringUtil_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      String[] stringArray0 = new String[8];
      // Undeclared exception!
      try { 
        StringUtil.join(stringArray0, "_DV#\"!%7y");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.System", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      StringUtil stringUtil0 = new StringUtil();
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      String string0 = StringUtil.resolve("Ah,7u{t<opvv5C9", "Ah,7u{t<opvv5C9");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      LinkedList<String> linkedList0 = new LinkedList<String>();
      ListIterator<String> listIterator0 = linkedList0.listIterator();
      String string0 = StringUtil.join((Iterator) listIterator0, "     ");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      LinkedList<String> linkedList0 = new LinkedList<String>();
      linkedList0.offer("     ");
      ListIterator<String> listIterator0 = linkedList0.listIterator();
      String string0 = StringUtil.join((Iterator) listIterator0, "     ");
      assertFalse(linkedList0.contains(string0));
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      String string0 = StringUtil.padding(858);
      assertEquals("                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          ", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      // Undeclared exception!
      try { 
        StringUtil.padding((-665));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // width must be > 0
         //
         verifyException("org.jsoup.helper.StringUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      String string0 = StringUtil.padding(13);
      assertEquals("             ", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      boolean boolean0 = StringUtil.isBlank((String) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      boolean boolean0 = StringUtil.isBlank("(Z~KCo");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      boolean boolean0 = StringUtil.isBlank("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      boolean boolean0 = StringUtil.isBlank("       ");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      boolean boolean0 = StringUtil.isNumeric((String) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      boolean boolean0 = StringUtil.isNumeric("1");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      boolean boolean0 = StringUtil.isNumeric("");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      boolean boolean0 = StringUtil.isNumeric("t");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      boolean boolean0 = StringUtil.isWhitespace(9);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      boolean boolean0 = StringUtil.isWhitespace(10);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      boolean boolean0 = StringUtil.isWhitespace(12);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      boolean boolean0 = StringUtil.isWhitespace(13);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      boolean boolean0 = StringUtil.isActuallyWhitespace(9);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      boolean boolean0 = StringUtil.isActuallyWhitespace(10);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      boolean boolean0 = StringUtil.isActuallyWhitespace(12);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      boolean boolean0 = StringUtil.isActuallyWhitespace(13);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      boolean boolean0 = StringUtil.isActuallyWhitespace(160);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      StringBuilder stringBuilder0 = StringUtil.stringBuilder();
      StringUtil.appendNormalisedWhitespace(stringBuilder0, "$Q `.dtGa!s", true);
      assertEquals("$Q `.dtGa!s", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      StringBuilder stringBuilder0 = StringUtil.stringBuilder();
      StringUtil.appendNormalisedWhitespace(stringBuilder0, "                 ", true);
      assertEquals("", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      String[] stringArray0 = new String[1];
      stringArray0[0] = "      ";
      boolean boolean0 = StringUtil.in((String) null, stringArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      String[] stringArray0 = new String[2];
      stringArray0[0] = "      ";
      boolean boolean0 = StringUtil.in("      ", stringArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      String[] stringArray0 = new String[5];
      stringArray0[0] = "Ah,7u{t<opvv5C9";
      stringArray0[2] = "Ah,7u{t<opvv5C9";
      boolean boolean0 = StringUtil.inSorted("", stringArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      String[] stringArray0 = new String[5];
      stringArray0[2] = "Ah,7u{t<opvvsC9";
      boolean boolean0 = StringUtil.inSorted("Ah,7u{t<opvvsC9", stringArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      URLStreamHandler uRLStreamHandler0 = mock(URLStreamHandler.class, new ViolatedAssumptionAnswer());
      URL uRL0 = MockURL.URL(".kkZ[D\u0007@>l", ".kkZ[D\u0007@>l", 14, ".kkZ[D\u0007@>l", uRLStreamHandler0);
      try { 
        StringUtil.resolve(uRL0, ".kkZ[D\u0007@>l");
        fail("Expecting exception: MalformedURLException");
      
      } catch(MalformedURLException e) {
         //
         // unknown protocol: .kkz[d\u0007@>l
         //
         verifyException("java.net.URL", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      URL uRL0 = MockURL.getFtpExample();
      URL uRL1 = StringUtil.resolve(uRL0, "?,zb?~?~([e");
      assertEquals("ftp://ftp.someFakeButWellFormedURL.org/fooExample?,zb?~?~([e", uRL1.toString());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      URL uRL0 = MockURL.getHttpExample();
      URL uRL1 = StringUtil.resolve(uRL0, ".akwkQ]Z[vDl");
      assertEquals("http://www.someFakeButWellFormedURL.org/.akwkQ]Z[vDl", uRL1.toString());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      LinkedList<String> linkedList0 = new LinkedList<String>();
      linkedList0.add("     ");
      linkedList0.addFirst("     ");
      linkedList0.offerLast("     ");
      ListIterator<String> listIterator0 = linkedList0.listIterator();
      StringUtil.join((Iterator) listIterator0, "?XcK1KOmjf`xL?Mf%MTH");
      assertFalse(listIterator0.hasNext());
      
      MockURL.getHttpExample();
      StringBuilder stringBuilder0 = StringUtil.stringBuilder();
      String string0 = StringUtil.normaliseWhitespace("java.lang.String@0000000001?XcK1KOmjf`xL?Mf%MTH     ?XcK1KOmjf`xL?Mf%MTH     ");
      StringBuilder stringBuilder1 = stringBuilder0.insert(1, (CharSequence) "java.lang.String@0000000001?XcK1KOmjf`xL?Mf%MTH     ?XcK1KOmjf`xL?Mf%MTH     ");
      StringBuilder stringBuilder2 = stringBuilder0.append((CharSequence) "java.lang.String@0000000001?XcK1KOmjf`xL?Mf%MTH     ?XcK1KOmjf`xL?Mf%MTH     ");
      StringBuilder stringBuilder3 = stringBuilder0.replace(1, 1, "java.lang.String@0000000001?XcK1KOmjf`xL?Mf%MTH ?XcK1KOmjf`xL?Mf%MTH ");
      stringBuilder2.insert(1, (CharSequence) stringBuilder3);
      Predicate.isEqual((Object) stringBuilder1);
      StringBuilder stringBuilder4 = stringBuilder0.append((CharSequence) stringBuilder0);
      stringBuilder4.append((Object) stringBuilder1);
      stringBuilder0.append("java.lang.String@0000000001?XcK1KOmjf`xL?Mf%MTH     ?XcK1KOmjf`xL?Mf%MTH     ");
      stringBuilder2.append((CharSequence) stringBuilder2);
      String string1 = StringUtil.normaliseWhitespace("java.lang.String@0000000001?XcK1KOmjf`xL?Mf%MTH ?XcK1KOmjf`xL?Mf%MTH ");
      assertTrue(string1.equals((Object)string0));
  }
}
