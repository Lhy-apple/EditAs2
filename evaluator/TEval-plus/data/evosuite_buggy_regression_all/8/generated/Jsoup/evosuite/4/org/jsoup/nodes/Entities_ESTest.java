/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:24:19 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.nio.charset.Charset;
import java.nio.charset.CharsetEncoder;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Entities;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Entities_ESTest extends Entities_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      // Undeclared exception!
      try { 
        Entities.escape("", (Document.OutputSettings) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Entities", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Entities entities0 = new Entities();
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Entities.EscapeMode entities_EscapeMode0 = Entities.EscapeMode.base;
      // Undeclared exception!
      try { 
        Entities.escape("\"NmytfuVf", (CharsetEncoder) null, entities_EscapeMode0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Entities", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Charset charset0 = Charset.defaultCharset();
      CharsetEncoder charsetEncoder0 = charset0.newEncoder();
      Entities.EscapeMode entities_EscapeMode0 = Entities.EscapeMode.extended;
      String string0 = Entities.escape("YLW%bV[i8z%3Q$P", charsetEncoder0, entities_EscapeMode0);
      assertEquals("YLW&percnt;bV&lsqb;i8z&percnt;3Q&dollar;P", string0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      String string0 = Entities.unescape("&#cLc\"y:'WnUBCi;A,");
      assertEquals("&#cLc\"y:'WnUBCi;A,", string0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      String string0 = Entities.unescape("e");
      assertEquals("e", string0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      String string0 = Entities.unescape(".s&le>SB1J/3];.'q3");
      assertEquals(".s\u2264>SB1J/3];.'q3", string0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      String string0 = Entities.unescape("&#xcLc\"y:'WnUBCi;A,");
      assertEquals("\fLc\"y:'WnUBCi;A,", string0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      String string0 = Entities.unescape("j{&E*(X(l`pMiGGJ1Yo");
      assertEquals("j{&E*(X(l`pMiGGJ1Yo", string0);
  }
}