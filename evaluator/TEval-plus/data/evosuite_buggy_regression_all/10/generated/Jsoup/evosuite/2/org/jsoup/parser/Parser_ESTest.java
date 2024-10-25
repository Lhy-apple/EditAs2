/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:46:39 GMT 2023
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
      Document document0 = Parser.parseBodyFragment("<?", "<?");
      assertEquals("<?", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document document0 = Parser.parseBodyFragment("<![CDATA[", "<![CDATA[");
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = Parser.parse("A@x<M</aVj+M=wW", "A@x<M</aVj+M=wW");
      assertFalse(document0.isBlock());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = Parser.parseBodyFragment("<!---", "<!---");
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = Parser.parse("<!", "<!");
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = Parser.parseBodyFragment("<!--", "<!--");
      assertFalse(document0.isBlock());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = Parser.parseBodyFragment("</", "</");
      assertFalse(document0.isBlock());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = Parser.parse("<s(y@M<q[[RA<=:", "<s(y@M<q[[RA<=:");
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = Parser.parse("h<T=lh)b`q>oB)k0", "h<T=lh)b`q>oB)k0");
      assertEquals("h<T=lh)b`q>oB)k0", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document document0 = Parser.parseBodyFragment("-@M<Garj+M='wW", "-@M<Garj+M='wW");
      assertEquals("-@M<Garj+M='wW", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = Parser.parse(":9<BfvF[w=\"8J</SZ", ":9<BfvF[w=\"8J</SZ");
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = Parser.parseBodyFragment("8E<sI=ssz7/j {sLW", "8E<sI=ssz7/j {sLW");
      assertEquals("8E<sI=ssz7/j {sLW", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      // Undeclared exception!
      try { 
        Parser.parseBodyFragment("h<T=lh)b`qBoB)k0", "h<T=lh)b`qBoB)k0");
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
      Document document0 = Parser.parse("text<area", "text<area");
      assertEquals("text<area", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = Parser.parseBodyFragment("tYxt<area", "tYxt<area");
      assertEquals("tYxt<area", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document document0 = Parser.parse("N:XC</bDX%n", "N:XC</bDX%n");
      assertFalse(document0.isBlock());
  }
}
