/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:16:29 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
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
      Document document0 = Document.createShell("&u2*IL_@K");
      Document.OutputSettings document_OutputSettings0 = document0.new OutputSettings();
      String string0 = Entities.escape("&u2*IL_@K", document_OutputSettings0);
      assertEquals("&amp;u2*IL_@K", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Entities entities0 = new Entities();
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Entities.EscapeMode entities_EscapeMode0 = Entities.EscapeMode.extended;
      // Undeclared exception!
      try { 
        Entities.escape("VIDEO", (CharsetEncoder) null, entities_EscapeMode0);
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
      String string0 = Entities.unescape("3/'v4&#A*tq\"S");
      assertEquals("3/'v4&#A*tq\"S", string0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      String string0 = Entities.unescape("#root");
      assertEquals("#root", string0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      String string0 = Entities.unescape("s$vI{&a_L5-c>r$`[?");
      assertEquals("s$vI{&a_L5-c>r$`[?", string0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      String string0 = Entities.unescape("mi&ast");
      assertEquals("mi*", string0);
  }
}