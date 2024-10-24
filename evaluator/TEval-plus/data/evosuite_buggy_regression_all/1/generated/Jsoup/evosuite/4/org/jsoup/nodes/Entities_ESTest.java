/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:45:34 GMT 2023
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
        Entities.escape("XNuG$of*i", (Document.OutputSettings) null);
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
        Entities.escape("SR0P?:N", (CharsetEncoder) null, entities_EscapeMode0);
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
      String string0 = Entities.escape("1G@7*NJVg1!OF2", charsetEncoder0, entities_EscapeMode0);
      assertEquals("1G&commat;7&midast;NJVg1&excl;OF2", string0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      String string0 = Entities.unescape("1G&commat;7&midast;NJVg1&excl;OF2");
      assertEquals("1G@7*NJVg1!OF2", string0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      String string0 = Entities.unescape("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      String string0 = Entities.unescape("Yo-&L#&#0tlO:K");
      assertEquals("Yo-&L#\u0000tlO:K", string0);
  }
}
