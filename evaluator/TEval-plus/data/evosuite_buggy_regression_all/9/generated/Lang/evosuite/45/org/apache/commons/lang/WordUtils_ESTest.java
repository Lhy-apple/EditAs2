/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:30:22 GMT 2023
 */

package org.apache.commons.lang;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.lang.WordUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class WordUtils_ESTest extends WordUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      String string0 = WordUtils.capitalize("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      String string0 = WordUtils.initials("A;)uT_j/(.! *X2i/J");
      assertEquals("A*", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      String string0 = WordUtils.uncapitalize(" d;{^Xe;SX!.eb3");
      assertEquals(" d;{^Xe;SX!.eb3", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      WordUtils wordUtils0 = new WordUtils();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      String string0 = WordUtils.wrap((String) null, (-690), "h{M3;]JvYd?W/Bt&", false);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      String string0 = WordUtils.wrap(" h|W)FKbWb.[Lj", (-10), "?", false);
      assertEquals("h|W)FKbWb.[Lj", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      String string0 = WordUtils.wrap("X )`TVQ", 4, "9U9b$\"':", false);
      assertEquals("X9U9b$\"':)`TVQ", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      String string0 = WordUtils.wrap("[irJM*DqueX7hm9Q", (-3135), "[irJM*DqueX7hm9Q", true);
      assertEquals("[[irJM*DqueX7hm9Qi[irJM*DqueX7hm9Qr[irJM*DqueX7hm9QJ[irJM*DqueX7hm9QM[irJM*DqueX7hm9Q*[irJM*DqueX7hm9QD[irJM*DqueX7hm9Qq[irJM*DqueX7hm9Qu[irJM*DqueX7hm9Qe[irJM*DqueX7hm9QX[irJM*DqueX7hm9Q7[irJM*DqueX7hm9Qh[irJM*DqueX7hm9Qm[irJM*DqueX7hm9Q9[irJM*DqueX7hm9QQ", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      String string0 = WordUtils.wrap("JYm X9q%6C=m7/", (-11));
      assertEquals("JYm\nX9q%6C=m7/", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      char[] charArray0 = new char[0];
      String string0 = WordUtils.capitalize("java.runtime.version", charArray0);
      assertEquals("java.runtime.version", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      String string0 = WordUtils.capitalize((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      String string0 = WordUtils.capitalizeFully((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      String string0 = WordUtils.capitalizeFully("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      char[] charArray0 = new char[0];
      String string0 = WordUtils.capitalizeFully("Wi=dowD NT", charArray0);
      assertEquals("Wi=dowD NT", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      char[] charArray0 = new char[0];
      String string0 = WordUtils.uncapitalize("YoUX?3Zk)y", charArray0);
      assertEquals("YoUX?3Zk)y", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      String string0 = WordUtils.uncapitalize((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      String string0 = WordUtils.uncapitalize("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      String string0 = WordUtils.swapCase((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      String string0 = WordUtils.swapCase("A;)uT_j/(.! *X2i/J");
      assertEquals("a;)Ut_J/(.! *x2I/j", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      String string0 = WordUtils.swapCase("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      String string0 = WordUtils.swapCase(" is less than 0: ");
      assertEquals(" IS LESS THAN 0: ", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      String string0 = WordUtils.initials((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      String string0 = WordUtils.initials("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      char[] charArray0 = new char[1];
      String string0 = WordUtils.initials("java.runte.version", charArray0);
      assertEquals("j", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      char[] charArray0 = new char[0];
      String string0 = WordUtils.initials("java.runtime.version", charArray0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      char[] charArray0 = new char[1];
      charArray0[0] = 'l';
      String string0 = WordUtils.capitalizeFully("org.apache.commons.lang.WordUtils", charArray0);
      assertEquals("Org.apache.commons.lAng.wordutilS", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      String string0 = WordUtils.abbreviate("A;)ut_j/(.! *x2i/j", (-1), (-1), "A;)uT_j/(.! *X2i/J");
      assertEquals("A;)ut_j/(.!A;)uT_j/(.! *X2i/J", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      String string0 = WordUtils.abbreviate((String) null, (-1), (-1), (String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      String string0 = WordUtils.abbreviate("", 0, 0, "");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      String string0 = WordUtils.abbreviate("5#'4j5Z-xv#HO", 1, 1, "5#'4j5Z-xv#HO");
      assertEquals("55#'4j5Z-xv#HO", string0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      // Undeclared exception!
      try { 
        WordUtils.abbreviate("5#'4jdZ-xv#0O", 184, 184, "5#'4jdZ-xv#0O");
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      String string0 = WordUtils.abbreviate("JM8:xu~q<{[r", (-1), (-1), "JM8:xu~q<{[r");
      assertEquals("JM8:xu~q<{[r", string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      String string0 = WordUtils.abbreviate("5L#'4j5 -xv#HO", 1, 1, "5L#'4j5 -xv#HO");
      assertEquals("55L#'4j5 -xv#HO", string0);
  }
}