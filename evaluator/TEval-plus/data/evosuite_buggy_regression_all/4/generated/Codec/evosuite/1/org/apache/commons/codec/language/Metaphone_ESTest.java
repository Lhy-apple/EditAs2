/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:26:24 GMT 2023
 */

package org.apache.commons.codec.language;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.codec.language.Metaphone;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Metaphone_ESTest extends Metaphone_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      metaphone0.setMaxCodeLen(354);
      metaphone0.encode("Parameter supplied to Metaphone encode is not of type java.lang.String");
      assertEquals(354, metaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.encode("Parameter supplied to Metaphone encode is not of type java.lang.String");
      assertEquals("PRM0", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      metaphone0.metaphone((String) null);
      assertEquals(4, metaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      metaphone0.encode("");
      assertEquals(4, metaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.encode("{");
      assertEquals(4, metaphone0.getMaxCodeLen());
      assertEquals("{", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.metaphone("AEIOU");
      assertEquals("E", string0);
      assertEquals(4, metaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.metaphone("KXPT");
      assertEquals("KKSP", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.metaphone("WR_:X<P$\"");
      assertEquals("RKSP", string0);
      assertEquals(4, metaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.metaphone("xCk7_)kKJj");
      assertEquals(4, metaphone0.getMaxCodeLen());
      assertEquals("XKKJ", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.encode("CHE");
      assertEquals("K", string0);
      assertEquals(4, metaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.metaphone("GN");
      assertEquals(4, metaphone0.getMaxCodeLen());
      assertEquals("N", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.encode("At");
      assertEquals("AT", string0);
      assertEquals(4, metaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.encode("Wnf?$7M>bqIP:S]Q");
      assertEquals("NFMB", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.metaphone("Wh:BZoo^mW#-[");
      assertEquals(4, metaphone0.getMaxCodeLen());
      assertEquals("BSM", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.encode("G?-i7JlghQa]'BJw[");
      assertEquals("KJLK", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.encode(".#]k`->MB0");
      assertEquals(4, metaphone0.getMaxCodeLen());
      assertEquals("KMB", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      Object object0 = metaphone0.encode((Object) "NFMB");
      assertEquals("NFM", object0);
      assertEquals(4, metaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.encode("SC");
      assertEquals(4, metaphone0.getMaxCodeLen());
      assertEquals("SK", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.encode("XCi|\"@&l#");
      assertEquals(4, metaphone0.getMaxCodeLen());
      assertEquals("XL", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.encode("HYK!dCI/HzH\"!K");
      assertEquals("KTSS", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.metaphone("TCH");
      assertEquals("X", string0);
      assertEquals(4, metaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.metaphone("CH");
      assertEquals(4, metaphone0.getMaxCodeLen());
      assertEquals("X", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.encode("CHV");
      assertEquals("XF", string0);
      assertEquals(4, metaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.metaphone("TD ");
      assertEquals(4, metaphone0.getMaxCodeLen());
      assertEquals("XT", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.encode("FqDGn3~");
      assertEquals("FKTN", string0);
      assertEquals(4, metaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      boolean boolean0 = metaphone0.isMetaphoneEqual("FqDGEn3~", "S7");
      assertEquals(4, metaphone0.getMaxCodeLen());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.encode("$GH");
      assertEquals("", string0);
      assertEquals(4, metaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.metaphone("_%~}(yQi6?)GHA\"");
      assertEquals("K", string0);
      assertEquals(4, metaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      Object object0 = metaphone0.encode((Object) "GE");
      assertEquals(4, metaphone0.getMaxCodeLen());
      assertEquals("J", object0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.encode("i?>hA?");
      assertEquals(4, metaphone0.getMaxCodeLen());
      assertEquals("IH", string0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.encode("p9>j&y");
      assertEquals("PJ", string0);
      assertEquals(4, metaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.encode("fd#\"liyE");
      assertEquals("FTLY", string0);
      assertEquals(4, metaphone0.getMaxCodeLen());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      String string0 = metaphone0.metaphone("fet,-mX{EFPkJoW_c");
      assertEquals("FXMK", string0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Metaphone metaphone0 = new Metaphone();
      Object object0 = new Object();
      try { 
        metaphone0.encode(object0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Parameter supplied to Metaphone encode is not of type java.lang.String
         //
         verifyException("org.apache.commons.codec.language.Metaphone", e);
      }
  }
}
