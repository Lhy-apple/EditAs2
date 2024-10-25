/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:08:05 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.NoSuchElementException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Document;
import org.jsoup.parser.Parser;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Parser_ESTest extends Parser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Document document0 = Parser.parseBodyFragment("qN4<b[SD+(</B", "qN4<b[SD+(</B");
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document document0 = Parser.parseBodyFragment("3<!-", "3<!-");
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = Parser.parse("<![CDATA[", "<![CDATA[");
      assertFalse(document0.isBlock());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = Parser.parseBodyFragment("<!---", "<!---");
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = Parser.parse("<?", "<?");
      assertEquals("<?", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = Parser.parse("<!--", "<!--");
      assertFalse(document0.isBlock());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = Parser.parse("</", "</");
      assertFalse(document0.isBlock());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = Parser.parse("}$<Gi'<[dMdYk;U", "}$<Gi'<[dMdYk;U");
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = Parser.parseBodyFragment("qO`4<bASE+ LUv/M", "qO`4<bASE+ LUv/M");
      assertEquals("qO`4<bASE+ LUv/M", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document document0 = Parser.parse("<g=V 7|WUef;", "<g=V 7|WUef;");
      assertEquals("#root", document0.tagName());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      // Undeclared exception!
      try { 
        Parser.parse("[AX69m<Xc=',C&3", "<!--%s-->");
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = Parser.parse("e3<gX=|W)$i+>>t;f)", "e3<gX=|W)$i+>>t;f)");
      assertFalse(document0.isBlock());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      // Undeclared exception!
      try { 
        Parser.parse("ile3<X=|W)$W+t;\"f)", "ile3<X=|W)$W+t;\"f)");
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = Parser.parse("qM`4<bASe+</M", "FUnvEYmG`");
      assertEquals("FUnvEYmG`", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = Parser.parse("mOdSY84{=Qei<d<Xd", "mOdSY84{=Qei<d<Xd");
      assertEquals("mOdSY84{=Qei<d<Xd", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document document0 = Parser.parseBodyFragment("15b<b</MM6Et7Bp", "15b<b</MM6Et7Bp");
      assertEquals("#root", document0.tagName());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Document document0 = Parser.parse("15b<b</MM6Et7Bp", "15b<b</MM6Et7Bp");
      assertEquals("#document", document0.nodeName());
  }
}
